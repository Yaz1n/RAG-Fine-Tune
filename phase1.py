import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm
import PyPDF2
from docx import Document
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

UPLOAD_DIR = "uploaded_files"

@dataclass
class DocumentChunk:
    """Represents a processed document chunk with metadata"""
    id: str
    content: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class DocumentProcessor:
    """Handles document loading, processing, and chunking"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nlp = spacy.load("en_core_web_sm")
        
    def load_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def load_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def load_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def load_web_page(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            logger.error(f"Error loading web page {url}: {e}")
            return ""
    
    def load_document(self, source: str) -> str:
        if source.startswith('http'):
            return self.load_web_page(source)
        elif source.endswith('.pdf'):
            return self.load_pdf(source)
        elif source.endswith('.docx'):
            return self.load_docx(source)
        elif source.endswith('.txt'):
            return self.load_txt(source)
        else:
            logger.error(f"Unsupported file type: {source}")
            return ""
    
    def clean_text(self, text: str) -> str:
        text = ' '.join(text.split())
        text = ''.join(char for char in text if char.isprintable())
        boilerplate = [
            'all rights reserved', 'copyright ©', 'terms of service', 'privacy policy',
            'contact us', 'home | about | contact', 'click here'
        ]
        for phrase in boilerplate:
            text = text.replace(phrase, '')
        return text.strip()
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics using spaCy"""
        doc = self.nlp(text)
        topics = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
        stop_words = set(stopwords.words('english'))
        topics = [t for t in topics if all(w not in stop_words for w in t.split()) and len(t) > 2]
        return list(set(topics))[:3]
    
    def chunk_text_semantic(self, text: str, source: str) -> List[DocumentChunk]:
        chunks = []
        sentences = sent_tokenize(text)
        topics = self.extract_topics(text)
        
        current_chunk = []
        current_word_count = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_words = len(word_tokenize(sentence))
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                start_char = text.find(current_chunk[0]) if current_chunk else 0
                end_char = start_char + len(chunk_text)
                
                chunk_id = hashlib.md5(f"{source}_{chunk_idx}_{chunk_text[:50]}".encode()).hexdigest()
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_text,
                    source=source,
                    chunk_index=chunk_idx,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        'word_count': current_word_count,
                        'char_count': len(chunk_text),
                        'sentence_count': len(current_chunk),
                        'chunk_method': 'semantic',
                        'topics': topics
                    }
                )
                
                chunks.append(chunk)
                overlap_sentences = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_sentences + [sentence]
                current_word_count = sum(len(word_tokenize(s)) for s in current_chunk)
                chunk_idx += 1
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            start_char = text.find(current_chunk[0]) if current_chunk else 0
            end_char = start_char + len(chunk_text)
            
            chunk_id = hashlib.md5(f"{source}_{chunk_idx}_{chunk_text[:50]}".encode()).hexdigest()
            
            chunk = DocumentChunk(
                id=chunk_id,
                content=chunk_text,
                source=source,
                chunk_index=chunk_idx,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    'word_count': current_word_count,
                    'char_count': len(chunk_text),
                    'sentence_count': len(current_chunk),
                    'chunk_method': 'semantic',
                    'topics': topics
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def filter_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        filtered_chunks = []
        stop_words = set(stopwords.words('english'))
        
        for chunk in chunks:
            if len(chunk.content.split()) < 20:
                continue
            special_char_ratio = sum(1 for c in chunk.content if not c.isalnum() and c != ' ') / len(chunk.content)
            if special_char_ratio > 0.25:
                continue
            words = chunk.content.split()
            numeric_ratio = sum(1 for word in words if word.replace('.', '').replace(',', '').isdigit()) / len(words)
            if numeric_ratio > 0.25:
                continue
            stopword_ratio = sum(1 for word in words if word.lower() in stop_words) / len(words)
            if stopword_ratio > 0.5:
                continue
            filtered_chunks.append(chunk)
        
        logger.info(f"Filtered {len(chunks) - len(filtered_chunks)} chunks, retained {len(filtered_chunks)} chunks")
        return filtered_chunks
    
    def process_documents(self, sources: List[str], chunk_method: str = 'semantic') -> List[DocumentChunk]:
        all_chunks = []
        
        for source in tqdm(sources, desc="Processing documents"):
            logger.info(f"Processing: {source}")
            text = self.load_document(source)
            if not text:
                continue
            text = self.clean_text(text)
            chunks = self.chunk_text_semantic(text, source)
            chunks = self.filter_chunks(chunks)
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {source}")
        
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks

class VectorStore:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized vector store with collection: {collection_name}")
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return
        logger.info("Generating embeddings...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {
                'source': chunk.source,
                'chunk_index': chunk.chunk_index,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
                **chunk.metadata
            }
            # Convert topics list to a comma-separated string
            if 'topics' in metadata and isinstance(metadata['topics'], list):
                metadata['topics'] = ', '.join(metadata['topics'])
            metadatas.append(metadata)
        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch_end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        logger.info(f"Added {len(chunks)} chunks to vector store")
        if chunks:
            logger.info(f"Sample chunk content: {chunks[0].content[:100]}...")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        search_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i]
            }
            # Convert topics string back to list for consistency
            if 'topics' in result['metadata'] and isinstance(result['metadata']['topics'], str):
                result['metadata']['topics'] = [t.strip() for t in result['metadata']['topics'].split(',')]
            search_results.append(result)
        return search_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        sample_size = min(100, count)
        if sample_size > 0:
            sample = self.collection.peek(sample_size)
            sources = set()
            word_counts = []
            char_counts = []
            topics = set()
            for metadata in sample['metadatas']:
                sources.add(metadata.get('source', 'unknown'))
                word_counts.append(metadata.get('word_count', 0))
                char_counts.append(metadata.get('char_count', 0))
                # Split topics string into list
                topics_list = metadata.get('topics', '')
                if isinstance(topics_list, str):
                    topics.update(t.strip() for t in topics_list.split(',') if t.strip())
                else:
                    topics.update(topics_list)
            stats = {
                'total_chunks': count,
                'unique_sources': len(sources),
                'sources': list(sources),
                'avg_word_count': np.mean(word_counts) if word_counts else 0,
                'avg_char_count': np.mean(char_counts) if char_counts else 0,
                'unique_topics': len(topics),
                'topics': list(topics),
                'sample_size': sample_size
            }
        else:
            stats = {
                'total_chunks': count,
                'unique_sources': 0,
                'sources': [],
                'avg_word_count': 0,
                'avg_char_count': 0,
                'unique_topics': 0,
                'topics': [],
                'sample_size': 0
            }
        return stats

def main():
    try:
        print("Starting Phase 1: Data Preparation and Infrastructure Setup")
        sources = []
        for f in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, f)
            if os.path.isfile(file_path):
                sources.append(file_path)
        if not sources:
            print(f"No files found in {UPLOAD_DIR}. Please upload documents first.")
            logger.error(f"No files found in {UPLOAD_DIR}. Please upload documents first.")
            return
        
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=200)
        print("Processing documents...")
        chunks = processor.process_documents(sources, chunk_method='semantic')
        print(f"Created {len(chunks)} total chunks")
        
        print("Setting up vector store...")
        vector_store = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
        
        print("Adding chunks to vector store...")
        vector_store.add_chunks(chunks)
        
        print("Collection statistics:")
        stats = vector_store.get_collection_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        print("\nPhase 1 Complete!")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()