import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Core libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# Document processing
import PyPDF2
from docx import Document
import requests
from bs4 import BeautifulSoup

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from sentence_transformers import SentenceTransformer

# Vector storage
import chromadb
from chromadb.config import Settings

# Download required NLTK data
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
    
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nlp = spacy.load("en_core_web_sm")
        
    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def load_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def load_txt(self, file_path: str) -> str:
        """Load text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def load_web_page(self, url: str) -> str:
        """Extract text from web page"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            print(f"Error loading web page {url}: {e}")
            return ""
    
    def load_document(self, source: str) -> str:
        """Load document from various sources"""
        if source.startswith('http'):
            return self.load_web_page(source)
        elif source.endswith('.pdf'):
            return self.load_pdf(source)
        elif source.endswith('.docx'):
            return self.load_docx(source)
        elif source.endswith('.txt'):
            return self.load_txt(source)
        else:
            print(f"Unsupported file type: {source}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        text = ''.join(char for char in text if char.isprintable())
        
        return text.strip()
    
    def chunk_text_fixed_size(self, text: str, source: str) -> List[DocumentChunk]:
        """Split text into fixed-size chunks with overlap"""
        chunks = []
        words = text.split()
        
        start_idx = 0
        chunk_idx = 0
        
        while start_idx < len(words):
            # Get chunk words
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions
            start_char = len(' '.join(words[:start_idx]))
            end_char = start_char + len(chunk_text)
            
            # Create chunk ID
            chunk_id = hashlib.md5(f"{source}_{chunk_idx}_{chunk_text[:50]}".encode()).hexdigest()
            
            # Create chunk object
            chunk = DocumentChunk(
                id=chunk_id,
                content=chunk_text,
                source=source,
                chunk_index=chunk_idx,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    'word_count': len(chunk_words),
                    'char_count': len(chunk_text),
                    'chunk_method': 'fixed_size'
                }
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            if end_idx >= len(words):
                break
            
            start_idx = end_idx - self.chunk_overlap
            chunk_idx += 1
        
        return chunks
    
    def chunk_text_semantic(self, text: str, source: str) -> List[DocumentChunk]:
        """Split text into semantic chunks using sentence boundaries"""
        chunks = []
        sentences = sent_tokenize(text)
        
        current_chunk = []
        current_word_count = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_words = len(word_tokenize(sentence))
            
            # If adding this sentence would exceed chunk size, create a new chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                start_char = text.find(current_chunk[0])
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
                        'chunk_method': 'semantic'
                    }
                )
                
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_sentences + [sentence]
                current_word_count = sum(len(word_tokenize(s)) for s in current_chunk)
                chunk_idx += 1
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            start_char = text.find(current_chunk[0])
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
                    'chunk_method': 'semantic'
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def filter_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Filter out low-quality chunks"""
        filtered_chunks = []
        
        for chunk in chunks:
            # Skip very short chunks
            if len(chunk.content.split()) < 10:
                continue
            
            # Skip chunks with too many special characters
            special_char_ratio = sum(1 for c in chunk.content if not c.isalnum() and c != ' ') / len(chunk.content)
            if special_char_ratio > 0.3:
                continue
            
            # Skip chunks that are mostly numbers
            words = chunk.content.split()
            numeric_ratio = sum(1 for word in words if word.replace('.', '').replace(',', '').isdigit()) / len(words)
            if numeric_ratio > 0.5:
                continue
            
            filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def process_documents(self, sources: List[str], chunk_method: str = 'semantic') -> List[DocumentChunk]:
        """Process multiple documents and return chunks"""
        all_chunks = []
        
        for source in tqdm(sources, desc="Processing documents"):
            print(f" Processing: {source}")
            
            # Load document
            text = self.load_document(source)
            if not text:
                continue
            
            # Clean text
            text = self.clean_text(text)
            
            # Chunk text
            if chunk_method == 'semantic':
                chunks = self.chunk_text_semantic(text, source)
            else:
                chunks = self.chunk_text_fixed_size(text, source)
            
            # Filter chunks
            chunks = self.filter_chunks(chunks)
            
            all_chunks.extend(chunks)
            print(f"Created {len(chunks)} chunks from {source}")
        
        return all_chunks

class VectorStore:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Initialized vector store with collection: {collection_name}")
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to vector store"""
        if not chunks:
            return
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Prepare data for ChromaDB
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
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        search_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            }
            search_results.append(result)
        
        return search_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        # Get sample of documents to analyze
        sample_size = min(100, count)
        if sample_size > 0:
            sample = self.collection.peek(sample_size)
            
            # Analyze metadata
            sources = set()
            word_counts = []
            char_counts = []
            
            for metadata in sample['metadatas']:
                sources.add(metadata.get('source', 'unknown'))
                word_counts.append(metadata.get('word_count', 0))
                char_counts.append(metadata.get('char_count', 0))
            
            stats = {
                'total_chunks': count,
                'unique_sources': len(sources),
                'sources': list(sources),
                'avg_word_count': np.mean(word_counts) if word_counts else 0,
                'avg_char_count': np.mean(char_counts) if char_counts else 0,
                'sample_size': sample_size
            }
        else:
            stats = {
                'total_chunks': count,
                'unique_sources': 0,
                'sources': [],
                'avg_word_count': 0,
                'avg_char_count': 0,
                'sample_size': 0
            }
        
        return stats

def main():
    """Main function to demonstrate Phase 1 implementation"""
    
    sources = []
    for f in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(file_path):
            sources.append(file_path)
    if not sources:
        print(f"No files found in {UPLOAD_DIR}. Please upload documents first.")
        return
    
    print("Phase 1: Data Preparation and Infrastructure Setup\n")
    # Initialize document processor
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Process documents
    print("1.Processing documents...")
    chunks = processor.process_documents(sources, chunk_method='semantic')
    
    # Initialize vector store
    print("2.Setting up vector store...")
    vector_store = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
    
    # Add chunks to vector store
    print("3.Adding chunks to vector store...")
    vector_store.add_chunks(chunks)
    
    # Get collection statistics
    print("4.Collection statistics:")
    stats = vector_store.get_collection_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    #for debugging purpose only
    print("Enter a query to find similar content. Type 'exit' to quit.")
    
    while True:
        user_query = input("Your query: ").strip()
        if user_query.lower() == 'exit':
            break
        
        if not user_query:
            print("Please enter a query.")
            continue

        print(f"\nSearching for: '{user_query}'")
        results = vector_store.search(user_query, k=3) # Get top 3 results
        
        if results:
            print(f"Found {len(results)} relevant chunks:")
            for i, res in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"Similarity Score: {res['similarity_score']:.4f}")
                print(f"Source: {res['metadata'].get('source', 'N/A')}")
                print(f"Chunk Index: {res['metadata'].get('chunk_index', 'N/A')}")
                print(f"Content: {res['content']}")
        else:
            print("No relevant chunks found or an error occurred during search.")
    
    print("\nSearch ended. Exiting....")

if __name__ == "__main__":
    main()