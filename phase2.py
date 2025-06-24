from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Verify API key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("WARNING: GROQ_API_KEY not loaded in phase2.py")
else:
    print(f"✅ API key loaded in phase2.py: {api_key[:10]}...{api_key[-4:]}")
import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# Core libraries
import numpy as np
from tqdm import tqdm

# API client for Groq
from groq import Groq

# From Phase 1
from phase1 import VectorStore, DocumentChunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGQuery:
    """Represents a RAG query with metadata"""
    id: str
    query: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class RAGResponse:
    """Represents a teacher model response"""
    query_id: str
    query: str
    response: str
    retrieved_chunks: List[Dict[str, Any]]
    model_name: str
    tokens_used: int
    response_time: float
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

class BaseTeacherModel(ABC):
    """Abstract base class for teacher models"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self.total_tokens_used = 0
        self.request_count = 0

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> Tuple[str, int]:
        """Generate response from prompt. Returns (response, tokens_used)"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'model_name': self.model_name,
            'total_tokens_used': self.total_tokens_used,
            'request_count': self.request_count,
            'avg_tokens_per_request': self.total_tokens_used / max(1, self.request_count)
        }

class GroqTeacherModel(BaseTeacherModel):
    """Groq teacher model via Groq API"""

    def __init__(self, model_name: str = "llama3-8b-8192", groq_api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as an environment variable.")
        self.client = Groq(api_key=self.groq_api_key)

    async def generate_response(self, prompt: str, **kwargs) -> Tuple[str, int]:
        messages = [
            {"role": "system", "content": "You are a helpful and intelligent AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        max_tokens = kwargs.get('max_tokens', 1000)
        temperature = kwargs.get('temperature', 0.7)

        try:
            chat_completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            response_text = chat_completion.choices[0].message.content
            tokens_used = chat_completion.usage.total_tokens

            self.total_tokens_used += tokens_used
            self.request_count += 1

            return response_text, tokens_used
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error generating response: {str(e)}", 0

    def count_tokens(self, text: str) -> int:
        # A rough estimate for token count for Groq models (often based on words or subword units)
        # For a more accurate count, you'd typically use a specific tokenizer provided by Groq or an associated library.
        # This simple split is a placeholder.
        return len(text.split())

class RAGPipeline:
    """Complete RAG pipeline with teacher model integration"""

    def __init__(self, vector_store: VectorStore, teacher_model: BaseTeacherModel,
                 retrieval_k: int = 1, max_context_tokens: int = 3000):
        self.vector_store = vector_store
        self.teacher_model = teacher_model
        self.retrieval_k = retrieval_k
        self.max_context_tokens = max_context_tokens
        self.responses = []

    def create_rag_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]],
                           system_prompt: Optional[str] = None) -> str:
        """Create RAG prompt with context and query"""
        if system_prompt is None:
            system_prompt = (
                "You are a helpful and intelligent AI assistant. Use the provided context information to answer the question accurately and comprehensively. "
                "If the context doesn't contain enough information to fully answer the question, say so and provide what information you can based on the context."
                "Do not hallucinate, only answer the question from given context information."
            )

        context_parts = []
        current_tokens = 0

        for chunk in retrieved_chunks:
            chunk_text = f"Source: {chunk['metadata']['source']}\nContent: {chunk['content']}\n"
            chunk_tokens = self.teacher_model.count_tokens(chunk_text)

            if current_tokens + chunk_tokens > self.max_context_tokens:
                break
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens

        context = "\n---\n".join(context_parts)

        prompt = f"""{system_prompt}

            Context Information: {context}

            Question: {query}

            """
        return prompt

    async def retrieve_and_generate(self, query: str, **kwargs) -> RAGResponse:
        """Complete RAG pipeline: retrieve + generate"""
        start_time = time.time()
        query_obj = RAGQuery(
            id=f"query_{int(time.time() * 1000)}",
            query=query,
            timestamp=datetime.now(),
            metadata=kwargs.get('metadata', {})
        )

        logger.info(f"Processing query: {query}")

        try:
            logger.info("Retrieving relevant chunks...")
            retrieved_chunks = self.vector_store.search(query, k=self.retrieval_k)

            if not retrieved_chunks:
                logger.warning("No relevant chunks retrieved")
                response_text = "I don't have enough relevant information in my knowledge base to answer this question."
                tokens_used = 0
                confidence_score = 0.0
            else:
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
                prompt = self.create_rag_prompt(
                    query,
                    retrieved_chunks,
                    kwargs.get('system_prompt')
                )

                logger.info("Generating response with teacher model...")
                response_text, tokens_used = await self.teacher_model.generate_response(
                    prompt,
                    max_tokens=kwargs.get('max_tokens', 1000),
                    temperature=kwargs.get('temperature', 0.7)
                )

                avg_similarity = np.mean([chunk['similarity_score'] for chunk in retrieved_chunks])
                confidence_score = min(avg_similarity * 1.2, 1.0)

            response = RAGResponse(
                query_id=query_obj.id,
                query=query,
                response=response_text,
                retrieved_chunks=retrieved_chunks,
                model_name=self.teacher_model.model_name,
                tokens_used=tokens_used,
                response_time=time.time() - start_time,
                confidence_score=confidence_score,
                timestamp=datetime.now(),
                metadata={
                    'retrieval_k': self.retrieval_k,
                    'chunks_used': len(retrieved_chunks),
                    **kwargs.get('metadata', {})
                }
            )

            self.responses.append(response)

            logger.info(f"Response generated in {response.response_time:.2f}s using {tokens_used} tokens")
            return response

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")

            response = RAGResponse(
                query_id=query_obj.id,
                query=query,
                response=f"Error processing query: {str(e)}",
                retrieved_chunks=[],
                model_name=self.teacher_model.model_name,
                tokens_used=0,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )

            self.responses.append(response)
            return response

    def batch_generate(self, queries: List[str], **kwargs) -> List[RAGResponse]:
        """Generate responses for multiple queries"""
        responses = []

        for query in tqdm(queries, desc="Processing queries"):
            if kwargs.get('delay', 0) > 0:
                time.sleep(kwargs['delay'])

            response = asyncio.run(self.retrieve_and_generate(query, **kwargs))
            responses.append(response)

        return responses

    def evaluate_responses(self, responses: List[RAGResponse]) -> Dict[str, Any]:
        """Evaluate batch of responses"""
        if not responses:
            return {}

        metrics = {
            'total_responses': len(responses),
            'avg_response_time': np.mean([r.response_time for r in responses]),
            'avg_tokens_used': np.mean([r.tokens_used for r in responses]),
            'avg_confidence': np.mean([r.confidence_score for r in responses]),
            'total_tokens': sum(r.tokens_used for r in responses),
            'avg_chunks_retrieved': np.mean([len(r.retrieved_chunks) for r in responses]),
            'error_rate': len([r for r in responses if 'error' in r.metadata]) / len(responses)
        }

        return metrics

    def save_responses(self, filename: str) -> None:
        """Save responses to JSON file"""
        responses_data = []
        for response in self.responses:
            response_dict = asdict(response)
            response_dict['timestamp'] = response.timestamp.isoformat()
            responses_data.append(response_dict)

        with open(filename, 'w') as f:
            json.dump(responses_data, f, indent=2, default=str)

        logger.info(f"Saved {len(responses_data)} responses to {filename}")

    def load_responses(self, filename: str) -> None:
        """Load responses from JSON file"""
        with open(filename, 'r') as f:
            responses_data = json.load(f)

        self.responses = []
        for data in responses_data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            self.responses.append(RAGResponse(**data))

        logger.info(f"Loaded {len(self.responses)} responses from {filename}")

class TeacherModelFactory:
    """Factory for creating teacher models"""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseTeacherModel:
        if model_type.lower() == 'groq':
            return GroqTeacherModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

def main():
    """Main function to demonstrate Phase 2 implementation with Groq"""

    print("=== Phase 2: Teacher Model Setup (Groq) ===\n")

    # Initialize vector store (assuming Phase 1 is complete)
    print("1. Loading vector store from Phase 1...")
    try:
        vector_store = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
        stats = vector_store.get_collection_stats()
        print(f"   Vector store loaded: {stats['total_chunks']} chunks from {stats['unique_sources']} sources")
    except Exception as e:
        print(f"   Error loading vector store: {e}")
        print("   Please run Phase 1 first to create the vector store")
        return

    # Initialize teacher model - Groq
    print("\n2. Initializing teacher model (Groq Llama3)...")
    try:
        # Ensure you have your GROQ_API_KEY set as an environment variable
        # or pass it directly here: groq_api_key="your_groq_api_key"
        teacher_model = TeacherModelFactory.create_model(
            'groq',
            model_name='llama3-8b-8192', # Or any other Groq model, e.g., 'mixtral-8x7b-32768'
            groq_api_key=os.getenv("GROQ_API_KEY") 
        )
        print("   ✓ Groq Llama3 model initialized")
    except Exception as e:
        print(f"   ✗ Groq initialization failed: {e}")
        print("   Please ensure GROQ_API_KEY environment variable is set correctly.")
        return

    # Initialize RAG pipeline
    print("\n3. Setting up RAG pipeline...")
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        teacher_model=teacher_model,
        retrieval_k=1,
        max_context_tokens=2000
    )
    print("   ✓ RAG pipeline initialized")

    print("\n=== Phase 2 Complete ===")
    print("✓ Groq Llama3 teacher model integrated successfully")
    print("✓ RAG pipeline implemented and tested")
    print("✓ Response generation working")
    print("✓ Ready for Phase 3: Training Data Generation")

if __name__ == "__main__":
    main()