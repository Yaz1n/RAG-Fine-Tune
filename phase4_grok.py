import asyncio
import os
import json
import logging
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
import torch
from phase1 import VectorStore
from phase2 import TeacherModelFactory, RAGPipeline
from phase3 import TrainingExample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    query: str
    student_response: str
    teacher_response: str
    bleu_score: float
    rouge_scores: Dict[str, float]
    cosine_similarity: float
    bert_score: float
    student_response_time: float
    teacher_response_time: float
    metadata: Dict[str, Any]

class StudentModelTrainer:
    def __init__(self, model_name: str = "google/flan-t5-base", output_dir: str = "./student_model"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Initialized student model: {model_name} on {self.device}")

    def load_dataset(self, dataset_path: str) -> Dataset:
        logger.info(f"Loading dataset from {dataset_path}")
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        dataset = Dataset.from_list(data)
        def tokenize_function(examples):
            inputs = [f"question: {q} context: {c}" for q, c in zip(examples['query'], examples['context'])]
            model_inputs = self.tokenizer(
                inputs,
                max_length=1024,
                truncation=True,
                padding="max_length"
            )
            labels = self.tokenizer(
                examples['teacher_response'],
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def train(self, train_dataset_path: str, val_dataset_path: str, epochs: int = 15, batch_size: int = 4):
        logger.info("Starting student model training...")
        train_dataset = self.load_dataset(train_dataset_path)
        val_dataset = self.load_dataset(val_dataset_path)
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=3e-5,
            warmup_steps=200,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            gradient_accumulation_steps=4,
            fp16=torch.cuda.is_available()
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        logger.info(f"Student model training completed. Saved to {self.output_dir / 'final_model'}")

    def generate_response(self, query: str, context: str, max_length: int = 512) -> Tuple[str, float]:
        start_time = time.time()
        input_text = f"question: {query} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=512,
            min_length=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_time = time.time() - start_time
        return response, response_time

class ModelComparator:
    def __init__(self, student_trainer: StudentModelTrainer, teacher_pipeline: RAGPipeline):
        self.student_trainer = student_trainer
        self.teacher_pipeline = teacher_pipeline
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    async def evaluate_single_pair(self, query: str, context: str, teacher_response: str) -> EvaluationResult:
        student_response, student_time = self.student_trainer.generate_response(query, context)
        teacher_response_obj = await self.teacher_pipeline.retrieve_and_generate(query)
        teacher_time = teacher_response_obj.response_time
        reference = [teacher_response.split()]
        candidate = student_response.split()
        bleu_score = sentence_bleu(reference, candidate, weights=(0.4, 0.3, 0.2, 0.1), smoothing_function=SmoothingFunction().method1)
        rouge_scores = self.rouge_scorer.score(teacher_response, student_response)
        embeddings = self.embedding_model.encode([student_response, teacher_response])
        cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        P, R, F1 = score([student_response], [teacher_response], lang="en", verbose=False)
        bert_score = float(F1.mean().item())
        return EvaluationResult(
            query=query,
            student_response=student_response,
            teacher_response=teacher_response,
            bleu_score=float(bleu_score),
            rouge_scores={k: float(v.fmeasure) for k, v in rouge_scores.items()},
            cosine_similarity=float(cosine_sim),
            bert_score=bert_score,
            student_response_time=float(student_time),
            teacher_response_time=float(teacher_time),
            metadata={"bert_score": bert_score}
        )

    async def evaluate_dataset(self, test_dataset_path: str) -> List[EvaluationResult]:
        logger.info(f"Evaluating models on test dataset: {test_dataset_path}")
        test_dataset = self.student_trainer.load_dataset(test_dataset_path)
        results = []
        for example in tqdm(test_dataset, desc="Evaluating test dataset"):
            result = await self.evaluate_single_pair(
                query=example['query'],
                context=example['context'],
                teacher_response=example['teacher_response']
            )
            results.append(result)
        return results

    def generate_comparison_report(self, results: List[EvaluationResult], output_file: str):
        if not results:
            logger.warning("No evaluation results to report.")
            return
        metrics = {
            'avg_bleu_score': float(np.mean([r.bleu_score for r in results])) if results else 0.0,
            'avg_rouge1': float(np.mean([r.rouge_scores['rouge1'] for r in results])) if results else 0.0,
            'avg_rouge2': float(np.mean([r.rouge_scores['rouge2'] for r in results])) if results else 0.0,
            'avg_rougeL': float(np.mean([r.rouge_scores['rougeL'] for r in results])) if results else 0.0,
            'avg_cosine_similarity': float(np.mean([r.cosine_similarity for r in results])) if results else 0.0,
            'avg_bert_score': float(np.mean([r.bert_score for r in results])) if results else 0.0,
            'avg_student_response_time': float(np.mean([r.student_response_time for r in results])) if results else 0.0,
            'avg_teacher_response_time': float(np.mean([r.teacher_response_time for r in results])) if results else 0.0,
            'total_examples': len(results)
        }
        report = {
            'metrics': metrics,
            'examples': [
                {
                    'query': r.query,
                    'student_response': r.student_response,
                    'teacher_response': r.teacher_response,
                    'bleu_score': float(r.bleu_score),
                    'rouge_scores': {k: float(v) for k, v in r.rouge_scores.items()},
                    'cosine_similarity': float(r.cosine_similarity),
                    'bert_score': float(r.bert_score),
                    'student_response_time': float(r.student_response_time),
                    'teacher_response_time': float(r.teacher_response_time)
                } for r in results[:5]
            ]
        }
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Comparison report saved to {output_file}")
        logger.info(f"Average BLEU Score: {metrics['avg_bleu_score']:.4f}")
        logger.info(f"Average ROUGE-1: {metrics['avg_rouge1']:.4f}")
        logger.info(f"Average ROUGE-2: {metrics['avg_rouge2']:.4f}")
        logger.info(f"Average ROUGE-L: {metrics['avg_rougeL']:.4f}")
        logger.info(f"Average Cosine Similarity: {metrics['avg_cosine_similarity']:.4f}")
        logger.info(f"Average BERT Score: {metrics['avg_bert_score']:.4f}")
        logger.info(f"Average Student Response Time: {metrics['avg_student_response_time']:.4f}s")
        logger.info(f"Average Teacher Response Time: {metrics['avg_teacher_response_time']:.4f}s")

async def main():
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print("=== Phase 4: Student Model Training and Comparison ===")
    print("\n1. Loading vector store from Phase 1...")
    try:
        vector_store = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
        stats = vector_store.get_collection_stats()
        print(f"   Vector store loaded: {stats['total_chunks']} chunks from {stats['unique_sources']} sources")
    except Exception as e:
        print(f"   Error loading vector store: {e}")
        return
    print("\n2. Initializing teacher model (Groq mixtral-8x7b-32768)...")
    try:
        teacher_model = TeacherModelFactory.create_model(
            'groq',
            model_name='llama3-70b-8192',
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        print("   ✓ Groq mixtral-8x7b-32768 model initialized")
    except Exception as e:
        print(f"   ✗ Groq initialization failed: {e}")
        return
    print("\n3. Setting up RAG pipeline...")
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        teacher_model=teacher_model,
        retrieval_k=3,
        max_context_tokens=4000
    )
    print("   ✓ RAG pipeline initialized")
    print("\n4. Initializing student model (Flan-T5-base)...")
    student_trainer = StudentModelTrainer(model_name="google/flan-t5-base", output_dir="./student_model")
    print("   ✓ Student model initialized")
    print("\n5. Training student model...")
    train_dataset_path = "./training_data/train_dataset.jsonl"
    val_dataset_path = "./training_data/validation_dataset.jsonl"
    if not os.path.exists(train_dataset_path) or not os.path.exists(val_dataset_path):
        print("   ✗ Training/validation dataset not found. Please run Phase 3 first.")
        return
    student_trainer.train(train_dataset_path, val_dataset_path, epochs=15, batch_size=4)
    print("   ✓ Student model training completed")
    print("\n6. Evaluating student vs teacher...")
    comparator = ModelComparator(student_trainer, rag_pipeline)
    test_dataset_path = "./training_data/test_dataset.jsonl"
    if not os.path.exists(test_dataset_path):
        print("   ✗ Test dataset not found. Please run Phase 3 first.")
        return
    results = await comparator.evaluate_dataset(test_dataset_path)
    comparator.generate_comparison_report(results, "./training_data/comparison_report.json")
    print("   ✓ Evaluation completed. Comparison report generated.")
    print("\n=== Phase 4 Complete ===")

if __name__ == "__main__":
    asyncio.run(main())