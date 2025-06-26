import logging
from dotenv import load_dotenv
from pathlib import Path
import os
import json
import random
import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import string
from groq import APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential
from phase1 import VectorStore
from phase2 import TeacherModelFactory, BaseTeacherModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    query: str
    context: str
    teacher_response: str
    retrieved_docs: List[str]
    teacher_confidence: float
    query_type: str
    difficulty_level: str
    metadata: Dict[str, Any]

class QueryGenerator:
    def __init__(self, teacher_model: BaseTeacherModel, embedding_model_name="all-mpnet-base-v2"):
        self.teacher_model = teacher_model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.query_templates = {
            "factual": {
                "simple": [
                    "What is {concept}?",
                    "Define {concept}.",
                    "What does {concept} mean?"
                ],
                "medium": [
                    "How does {concept} work?",
                    "What are the key features of {concept}?",
                    "Why is {concept} important?",
                    "What is the purpose of {concept}?"
                ],
                "complex": [
                    "What are the advantages and disadvantages of {concept}?",
                    "How has {concept} evolved over time?",
                    "What are the applications of {concept}?"
                ]
            },
            "analytical": {
                "simple": [
                    "List the main components of {concept}.",
                    "What are the types of {concept}?",
                    "Identify the key parts of {concept}."
                ],
                "medium": [
                    "Name the key elements of {concept}.",
                    "How is {concept} implemented?",
                    "What factors affect {concept}?"
                ],
                "complex": [
                    "What factors influence the success of {concept}?",
                    "Analyze the impact of {concept} on its domain.",
                    "Evaluate the effectiveness of {concept}."
                ]
            },
            "summarization": {
                "simple": [
                    "Summarize {concept}.",
                    "Give me an overview of {concept}.",
                    "What is {concept} about?"
                ],
                "medium": [
                    "Explain the main points about {concept}.",
                    "What are the key takeaways from {concept}?",
                    "Provide a comprehensive summary of {concept}."
                ],
                "complex": [
                    "Provide a detailed analysis of {concept} including its benefits and limitations.",
                    "Summarize the challenges and opportunities of {concept}.",
                    "Explain {concept} in the context of its field."
                ]
            },
            "comparative": {
                "medium": [
                    "Compare and contrast {concept} and {concept2}.",
                    "What are the similarities and differences between {concept} and {concept2}?",
                    "How do {concept} and {concept2} differ in application?"
                ],
                "complex": [
                    "Analyze the relationship between {concept} and {concept2}.",
                    "How does {concept} impact {concept2}?",
                    "Evaluate the trade-offs between {concept} and {concept2}."
                ]
            }
        }

    def _fill_template(self, template, concepts):
        formatter = string.Formatter()
        fields = [fname for _, fname, _, _ in formatter.parse(template) if fname]
        vals = {}
        if 'concept' in fields and concepts:
            vals['concept'] = concepts[0]
        if 'concept2' in fields and len(concepts) > 1:
            vals['concept2'] = concepts[1]
        if 'concept' in fields and 'concept' not in vals:
            vals['concept'] = "a given topic"
        if 'concept2' in fields and 'concept2' not in vals:
            vals['concept2'] = "another topic"
        try:
            return template.format(**vals)
        except KeyError as e:
            logger.error(f"Missing key for template formatting: {e} in template {template} with concepts {concepts}")
            return template.replace("{concept}", concepts[0] if concepts else "topic").replace("{concept2}", concepts[1] if len(concepts) > 1 else "another topic")
        except Exception as e:
            logger.error(f"Template formatting error: {e} with template {template} and vals {vals}")
            return template.replace("{", "").replace("}", "")

    async def extract_concepts_from_chunk(self, chunk_text: str) -> List[str]:
        prompt = f"""Extract up to 3 most important *concepts* from the text below.

            Rules:
            - Return as a single line, comma-separated.
            - Never return multiple lines.
            - Never add bullet points, numbering, or explanations.
            - Each concept should be less than 4 words.
            - Focus on specific technical or domain-relevant terms.

            Example:
            Text:
            \"\"\"
            Deep learning is a subfield of machine learning concerned with algorithms inspired by the structure of the brain. It is used in image recognition, natural language processing, and robotics.
            \"\"\"
            Concepts:
            deep learning, image recognition, natural language processing

            Now extract from this:
            Text:
            \"\"\"
            {chunk_text}
            \"\"\"
            Concepts:"""
        try:
            response, _ = await self.teacher_model.generate_response(prompt, max_tokens=1000)
            lines = response.splitlines()
            filtered_lines = [
                line for line in lines
                if not any(phrase in line.lower() for phrase in ["here are", "top", "concepts", "the text", "most important", "i am an ai"])
            ]
            cleaned_response = ' '.join(filtered_lines)
            cleaned_response = re.sub(r"[\n\r\t]+", ",", cleaned_response)
            cleaned_response = re.sub(r"\d+\.\s*|\*\s*|\-\s*", "", cleaned_response)
            STOPWORDS = {'the', 'and', 'or', 'is', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by'}
            concepts = [c.strip().lower() for c in cleaned_response.split(',') if c.strip()]
            concepts = [c for c in concepts if c not in STOPWORDS and len(c.split()) <= 4 and len(c) > 2]
            return concepts[:3]
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []

    async def paraphrase_query(self, query: str) -> str:
        """Paraphrase a query using the teacher model"""
        prompt = f"""Paraphrase the following query without changing its meaning:

            Original Query: {query}

            Paraphrased Query:"""
        try:
            response, _ = await self.teacher_model.generate_response(prompt, max_tokens=200, temperature=0.8)
            paraphrased = response.strip()
            if any(bad_phrase in paraphrased.lower() for bad_phrase in ["here are", "i am an ai"]):
                return query
            return paraphrased
        except Exception as e:
            logger.error(f"Error paraphrasing query: {e}")
            return query

    async def generate_synthetic_queries(self, document_chunks: List[Dict], 
                                        queries_per_chunk: int = 10) -> List[Dict]:
        all_queries = []
        for chunk in tqdm(document_chunks, desc="Generating synthetic queries"):
            chunk_text = chunk['content']
            chunk_id = chunk['id']
            concepts = await self.extract_concepts_from_chunk(chunk_text)
            if not concepts:
                continue
            chunk_queries = []
            for _ in range(queries_per_chunk):
                query_type = random.choice([k for k in self.query_templates.keys() if k != "comparative"])
                difficulty = random.choice(list(self.query_templates[query_type].keys()))
                templates = self.query_templates[query_type][difficulty]
                template = random.choice(templates)
                query = self._fill_template(template, concepts)
                if any(bad_phrase in query.lower() for bad_phrase in ["here are", "top 3", "top 5", "the text", "concepts from the text", "i am an ai"]):
                    logger.warning(f"Skipping malformed query: {query}")
                    continue
                # Add original query
                chunk_queries.append({
                    'query': query,
                    'source_chunk_id': chunk_id,
                    'query_type': query_type,
                    'difficulty_level': difficulty,
                    'concepts': concepts,
                    'is_paraphrased': False
                })
                # Add paraphrased query
                paraphrased_query = await self.paraphrase_query(query)
                if paraphrased_query != query:
                    chunk_queries.append({
                        'query': paraphrased_query,
                        'source_chunk_id': chunk_id,
                        'query_type': query_type,
                        'difficulty_level': difficulty,
                        'concepts': concepts,
                        'is_paraphrased': True
                    })
            all_queries.extend(chunk_queries)
        return all_queries

    async def generate_cross_chunk_queries(self, document_chunks: List[Dict], 
                                          num_queries: int = 100) -> List[Dict]:
        cross_chunk_queries = []
        if len(document_chunks) < 2:
            logger.warning("Not enough chunks for cross-chunk query generation.")
            return []
        for _ in range(num_queries):
            selected_indices = random.sample(range(len(document_chunks)), min(random.randint(2, 3), len(document_chunks)))
            selected_chunks = [document_chunks[i] for i in selected_indices]
            all_concepts = []
            for chunk in selected_chunks:
                concepts = await self.extract_concepts_from_chunk(chunk['content'])
                all_concepts.extend(concepts)
            unique_concepts = list(set(all_concepts))
            if len(unique_concepts) < 2:
                continue
            query_types = ['comparative', 'analytical']
            query_type = random.choice(query_types)
            difficulty = random.choice(['medium', 'complex'])
            templates = self.query_templates[query_type][difficulty]
            template = random.choice(templates)
            if query_type == 'comparative' and len(unique_concepts) >= 2:
                concept1, concept2 = random.sample(unique_concepts, 2)
                concepts_for_template = [concept1, concept2]
            else:
                concepts_for_template = unique_concepts
            query = self._fill_template(template, concepts_for_template)
            if any(bad_phrase in query.lower() for bad_phrase in ["here are", "top 3", "top 5", "the text", "concepts from the text", "i am an ai"]):
                logger.warning(f"Skipping malformed query: {query}")
                continue
            chunk_queries = [{
                'query': query,
                'source_chunk_ids': [chunk['id'] for chunk in selected_chunks],
                'query_type': query_type,
                'difficulty_level': difficulty,
                'concepts': all_concepts,
                'is_paraphrased': False
            }]
            paraphrased_query = await self.paraphrase_query(query)
            if paraphrased_query != query:
                chunk_queries.append({
                    'query': paraphrased_query,
                    'source_chunk_ids': [chunk['id'] for chunk in selected_chunks],
                    'query_type': query_type,
                    'difficulty_level': difficulty,
                    'concepts': all_concepts,
                    'is_paraphrased': True
                })
            cross_chunk_queries.extend(chunk_queries)
        return cross_chunk_queries

    def remove_duplicate_queries(self, queries: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
        if not queries:
            return []
        query_texts = [q['query'] for q in queries]
        query_embeddings = self.embedding_model.encode(query_texts, show_progress_bar=False)
        to_remove_indices = set()
        for i in tqdm(range(len(queries)), desc="Deduplicating queries"):
            if i in to_remove_indices:
                continue
            for j in range(i + 1, len(queries)):
                if j in to_remove_indices:
                    continue
                similarity = cosine_similarity(
                    query_embeddings[i].reshape(1, -1), 
                    query_embeddings[j].reshape(1, -1)
                )[0][0]
                if similarity > similarity_threshold:
                    to_remove_indices.add(j)
        filtered_queries = [query for idx, query in enumerate(queries) if idx not in to_remove_indices]
        logger.info(f"Removed {len(queries) - len(filtered_queries)} duplicate queries (pre-LLM).")
        return filtered_queries

class TeacherResponseCollector:
    def __init__(self, teacher_model: BaseTeacherModel, vector_store: VectorStore, retrieval_k=3):
        self.teacher_model = teacher_model
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

    def retrieve_context(self, query: str) -> Tuple[str, List[str]]:
        try:
            relevant_docs = self.vector_store.search(query, k=self.retrieval_k)
            context = "\n\n".join([doc['content'] for doc in relevant_docs])
            doc_ids = [doc['id'] for doc in relevant_docs]
            return context, doc_ids
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "", []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_teacher_response_async(self, query: str, context: str) -> Tuple[str, float]:
        prompt = f"""You are an expert AI assistant tasked with generating a high-quality, concise, and original answer to the following question.
            Your answer MUST be derived SOLELY from the provided context. Synthesize and rephrase the information; **do NOT copy sentences directly from the context.**
            If the context does not contain the answer, state "I cannot answer this question based on the provided information."
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
        try:
            response_text, _ = await self.teacher_model.generate_response(prompt, max_tokens=512, temperature=0.1)
            confidence = self.estimate_confidence(response_text, context, query)
            return response_text.strip(), confidence
        except Exception as e:
            logger.error(f"Error generating teacher response: {e}")
            return "", 0.0

    def estimate_confidence(self, response: str, context: str, query: str) -> float:
        confidence = 0.5
        if "i cannot answer this question based on the provided information" in response.lower():
            return 0.1
        if len(response) > 100:
            confidence += 0.2
        if len(response) > 200:
            confidence += 0.1
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        if response_words:
            overlap_ratio = len(context_words.intersection(response_words)) / len(response_words)
            confidence += overlap_ratio * 0.3
        uncertain_phrases = ['i am not sure', 'unclear', 'insufficient information', 
                            'cannot determine', 'not enough context', 'it is not clear from the context']
        for phrase in uncertain_phrases:
            if phrase in response.lower():
                confidence -= 0.3
                break
        embeddings = self.embedding_model.encode([query, response])
        query_response_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        confidence += query_response_similarity * 0.2
        return max(0.0, min(1.0, confidence))

    async def collect_responses_batch_async(self, queries: List[Dict], batch_size: int = 10) -> List[TrainingExample]:
        training_examples = []
        tasks = []
        for query_data in queries:
            tasks.append(self._process_single_query_async(query_data))
        for i in tqdm(range(0, len(tasks), batch_size), desc="Collecting teacher responses"):
            batch_tasks = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch processing: {result}")
                    continue
                if result:
                    training_examples.append(result)
        return training_examples

    async def _process_single_query_async(self, query_data: Dict) -> Optional[TrainingExample]:
        query = query_data['query']
        context, retrieved_docs = self.retrieve_context(query)
        if not context:
            return None
        teacher_response, confidence = await self.generate_teacher_response_async(query, context)
        await asyncio.sleep(60)
        if not teacher_response or confidence < 0.2:
            return None
        training_example = TrainingExample(
            query=query,
            context=context,
            teacher_response=teacher_response,
            retrieved_docs=retrieved_docs,
            teacher_confidence=confidence,
            query_type=query_data.get('query_type', 'unknown'),
            difficulty_level=query_data.get('difficulty_level', 'medium'),
            metadata={
                'source_chunk_ids': query_data.get('source_chunk_ids', []),
                'concepts': query_data.get('concepts', []),
                'is_paraphrased': query_data.get('is_paraphrased', False)
            }
        )
        return training_example

class DataQualityController:
    def __init__(self, embedding_model_name="all-mpnet-base-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def filter_by_confidence(self, examples: List[TrainingExample], min_confidence: float = 0.9) -> List[TrainingExample]:
        initial_count = len(examples)
        filtered = [ex for ex in examples if ex.teacher_confidence >= min_confidence]
        logger.info(f"Filtered {initial_count - len(filtered)} examples by confidence (min: {min_confidence}).")
        return filtered

    def filter_by_relevance(self, examples: List[TrainingExample], similarity_threshold: float = 0.6) -> List[TrainingExample]:
        initial_count = len(examples)
        filtered = []
        query_texts = [ex.query for ex in examples]
        response_texts = [ex.teacher_response for ex in examples]
        embeddings = self.embedding_model.encode(query_texts + response_texts, show_progress_bar=False)
        query_embeddings = embeddings[:len(examples)]
        response_embeddings = embeddings[len(examples):]
        for i, ex in enumerate(examples):
            similarity = cosine_similarity(
                query_embeddings[i].reshape(1, -1),
                response_embeddings[i].reshape(1, -1)
            )[0][0]
            if similarity >= similarity_threshold:
                filtered.append(ex)
        logger.info(f"Filtered {initial_count - len(filtered)} examples by relevance (min similarity: {similarity_threshold}).")
        return filtered

    def remove_duplicates(self, examples: List[TrainingExample], similarity_threshold: float = 0.95) -> List[TrainingExample]:
        if not examples:
            return examples
        logger.info(f"Starting post-LLM deduplication of {len(examples)} examples...")
        queries = [ex.query for ex in examples] 
        query_embeddings = self.embedding_model.encode(queries, show_progress_bar=False)
        to_remove = set()
        for i in tqdm(range(len(examples)), desc="Deduplicating post-LLM responses"):
            if i in to_remove:
                continue
            for j in range(i + 1, len(examples)):
                if j in to_remove:
                    continue
                similarity = cosine_similarity(
                    query_embeddings[i].reshape(1, -1), 
                    query_embeddings[j].reshape(1, -1)
                )[0][0]
                if similarity > similarity_threshold:
                    if examples[i].teacher_confidence >= examples[j].teacher_confidence:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        filtered_examples = [ex for i, ex in enumerate(examples) if i not in to_remove]
        logger.info(f"Removed {len(examples) - len(filtered_examples)} duplicate examples (post-LLM).")
        return filtered_examples

    def balance_dataset(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        if not examples:
            logger.warning("No training examples to balance. Returning empty list.")
            return []
        groups = {}
        for ex in examples:
            key = (ex.query_type, ex.difficulty_level)
            if key not in groups:
                groups[key] = []
            groups[key].append(ex)
        if not groups:
            return []
        group_sizes = [len(group) for group in groups.values()]
        target_size = max(30, int(np.median(group_sizes)))
        balanced_examples = []
        for key, group in groups.items():
            if len(group) <= target_size:
                balanced_examples.extend(group)
            else:
                sorted_group = sorted(group, key=lambda x: x.teacher_confidence, reverse=True)
                balanced_examples.extend(sorted_group[:target_size])
        logger.info(f"Balanced dataset: {len(balanced_examples)} examples across {len(groups)} groups with target size {target_size}.")
        return balanced_examples

class TrainingDatasetBuilder:
    def __init__(self, teacher_model: BaseTeacherModel, vector_store: VectorStore, output_dir: str = "training_data"):
        self.teacher_model = teacher_model
        self.vector_store = vector_store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.query_generator = QueryGenerator(teacher_model)
        self.response_collector = TeacherResponseCollector(teacher_model, vector_store)
        self.quality_controller = DataQualityController()

    async def build_training_dataset(self, document_chunks: List[Dict], 
                                    queries_per_chunk: int = 10,
                                    cross_chunk_queries_count: int = 100,
                                    min_confidence: float = 0.9) -> List[TrainingExample]:
        logger.info("Starting training dataset generation...")
        logger.info("Generating synthetic queries (initial pool)...")
        synthetic_queries = await self.query_generator.generate_synthetic_queries(
            document_chunks, queries_per_chunk
        )
        print("\n=== Generated Synthetic Queries (Raw, Sample) ===")
        for i, q in enumerate(synthetic_queries[:5]):
            print(f"- {q['query']} (Paraphrased: {q['is_paraphrased']})")
        if len(synthetic_queries) > 5:
            print(f"... and {len(synthetic_queries) - 5} more.")
        print("=== End of Raw Query Sample ===\n")
        logger.info(f"Generating {cross_chunk_queries_count} cross-chunk queries...")
        cross_chunk_queries = await self.query_generator.generate_cross_chunk_queries(
            document_chunks, num_queries=cross_chunk_queries_count
        )
        logger.info(f"Generated {len(cross_chunk_queries)} cross-chunk queries.")
        all_queries_raw = synthetic_queries + cross_chunk_queries
        logger.info(f"Generated {len(all_queries_raw)} total raw queries (synthetic + cross-chunk).")
        logger.info("Deduplicating queries before sending to teacher...")
        unique_queries = self.query_generator.remove_duplicate_queries(all_queries_raw)
        logger.info(f"After pre-LLM deduplication: {len(unique_queries)} unique queries.")
        logger.info("Collecting teacher responses for unique queries (this may take time)...")
        training_examples = await self.response_collector.collect_responses_batch_async(unique_queries)
        logger.info(f"Collected {len(training_examples)} training examples.")
        logger.info("Applying quality control...")
        training_examples = self.quality_controller.filter_by_confidence(training_examples, min_confidence)
        logger.info(f"After confidence filtering: {len(training_examples)} examples.")
        training_examples = self.quality_controller.filter_by_relevance(training_examples)
        logger.info(f"After relevance filtering: {len(training_examples)} examples.")
        training_examples = self.quality_controller.remove_duplicates(training_examples)
        logger.info(f"After post-LLM deduplication: {len(training_examples)} examples.")
        training_examples = self.quality_controller.balance_dataset(training_examples)
        logger.info(f"Final balanced dataset: {len(training_examples)} examples.")
        return training_examples

    def save_dataset(self, training_examples: List[TrainingExample], 
                     split_ratios: Dict[str, float] = None):
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'validation': 0.2, 'test': 0.1}
        dataset_dicts = []
        for ex in training_examples:
            dataset_dicts.append({
                'query': ex.query,
                'context': ex.context,
                'teacher_response': ex.teacher_response,
                'retrieved_docs': ex.retrieved_docs,
                'teacher_confidence': ex.teacher_confidence,
                'query_type': ex.query_type,
                'difficulty_level': ex.difficulty_level,
                'metadata': ex.metadata
            })
        random.shuffle(dataset_dicts)
        total_size = len(dataset_dicts)
        train_size = int(total_size * split_ratios['train'])
        val_size = int(total_size * split_ratios['validation'])
        splits = {
            'train': dataset_dicts[:train_size],
            'validation': dataset_dicts[train_size:train_size + val_size],
            'test': dataset_dicts[train_size + val_size:]
        }
        for split_name, split_data in splits.items():
            output_file = self.output_dir / f"{split_name}_dataset.jsonl"
            with open(output_file, 'w') as f:
                for example in split_data:
                    f.write(json.dumps(example) + '\n')
            logger.info(f"Saved {len(split_data)} examples to {output_file}")
        self.save_dataset_statistics(training_examples)

    def save_dataset_statistics(self, training_examples: List[TrainingExample]):
        stats = {
            'total_examples': len(training_examples),
            'avg_confidence': float(np.mean([ex.teacher_confidence for ex in training_examples])) if training_examples else 0.0,
            'query_type_distribution': {},
            'difficulty_distribution': {},
            'paraphrase_ratio': sum(1 for ex in training_examples if ex.metadata.get('is_paraphrased', False)) / len(training_examples) if training_examples else 0.0,
            'confidence_distribution': {
                'high_confidence (>0.8)': sum(1 for ex in training_examples if ex.teacher_confidence > 0.8),
                'medium_confidence (0.5-0.8)': sum(1 for ex in training_examples if 0.5 <= ex.teacher_confidence <= 0.8),
                'low_confidence (<0.5)': sum(1 for ex in training_examples if ex.teacher_confidence < 0.5)
            }
        }
        for ex in training_examples:
            query_type = ex.query_type
            stats['query_type_distribution'][query_type] = stats['query_type_distribution'].get(query_type, 0) + 1
            difficulty = ex.difficulty_level
            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved dataset statistics to {stats_file}")
        print("\n=== Dataset Statistics ===")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Average confidence: {stats['avg_confidence']:.3f}")
        print(f"Paraphrase ratio: {stats['paraphrase_ratio']:.3f}")
        print(f"Query types: {stats['query_type_distribution']}")
        print(f"Difficulty levels: {stats['difficulty_distribution']}")
        print(f"Confidence distribution: {stats['confidence_distribution']}")

async def main():
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print("Loading document chunks from ChromaDB vector store (Phase 1)...")
    vector_store = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
    collection_stats = vector_store.get_collection_stats()
    total_chunks = collection_stats['total_chunks']
    if total_chunks == 0:
        print("No chunks found in the vector store. Please run Phase 1 first.")
        return
    max_peek = total_chunks
    print(f"Found {total_chunks} chunks, loading for dataset generation...")
    chunks_raw = vector_store.collection.peek(max_peek)
    document_chunks = []
    for i in range(len(chunks_raw['ids'])):
        document_chunks.append({
            'id': chunks_raw['ids'][i],
            'content': chunks_raw['documents'][i]
        })
    print("\nInitializing teacher model (Groq mixtral-8x7b-32768)...")
    try:
        teacher_model = TeacherModelFactory.create_model(
            'groq',
            model_name='llama3-70b-8192',
            groq_api_key=os.getenv("GROQ_API_KEY") 
        )
        print("   ✓ Groq mixtral-8x7b-32768 model initialized successfully.")
    except Exception as e:
        print(f"   ✗ Groq model initialization failed: {e}")
        return
    dataset_builder = TrainingDatasetBuilder(teacher_model, vector_store)
    training_examples = await dataset_builder.build_training_dataset(
        document_chunks,
        queries_per_chunk=5,
        cross_chunk_queries_count=20,
        min_confidence=0.8
    )
    dataset_builder.save_dataset(training_examples)
    print(f"\nTraining dataset generation complete! Generated {len(training_examples)} examples.")

if __name__ == "__main__":
    asyncio.run(main())