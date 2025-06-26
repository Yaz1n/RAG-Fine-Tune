import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List
import logging
from pathlib import Path
from phase1 import DocumentProcessor, VectorStore
from phase2 import TeacherModelFactory, RAGPipeline
from phase3 import TrainingDatasetBuilder
from phase4_grok import StudentModelTrainer, ModelComparator
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Processing and RAG API")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow React frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

UPLOAD_DIR = "uploaded_files"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

class DatasetGenerationRequest(BaseModel):
    queries_per_chunk: int = 5
    cross_chunk_queries_count: int = 50
    min_confidence: float = 0.9

class TrainAndEvaluateRequest(BaseModel):
    epochs: int = 15
    batch_size: int = 4
    model_name: str = "google/flan-t5-base"
    output_dir: str = "./student_model"

@app.post("/upload-files/")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload multiple files to the server
    """
    try:
        uploaded_files = []
        for file in files:
            # Validate file type
            if not file.filename.endswith(('.pdf', '.docx', '.txt')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type for {file.filename}. Allowed types: .pdf, .docx, .txt"
                )
            
            # Save file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file_path)
            logger.info(f"Uploaded file: {file_path}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded {len(uploaded_files)} files",
                "files": uploaded_files
            }
        )
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")

@app.post("/process-and-embed/")
async def process_and_embed():
    """
    Endpoint to process uploaded files and store embeddings
    """
    try:
        # Initialize processor and vector store
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=200)
        vector_store = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
        
        # Get list of files from upload directory
        sources = [
            os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, f))
        ]
        
        if not sources:
            raise HTTPException(
                status_code=400,
                detail=f"No files found in {UPLOAD_DIR}. Please upload documents first."
            )
        
        # Process documents
        logger.info("Processing documents...")
        chunks = processor.process_documents(sources, chunk_method='semantic')
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No valid chunks generated from the documents."
            )
        
        # Add chunks to vector store
        logger.info("Adding chunks to vector store...")
        vector_store.add_chunks(chunks)
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully processed and embedded {len(chunks)} chunks",
                "chunk_count": len(chunks),
                "collection_stats": stats
            }
        )
    except Exception as e:
        logger.error(f"Error processing and embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing and embedding: {str(e)}")

@app.post("/generate-training-dataset/")
async def generate_training_dataset(request: DatasetGenerationRequest):
    """
    Endpoint to generate a training dataset using synthetic queries and teacher responses
    """
    try:
        # Initialize vector store
        vector_store = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
        
        # Check if vector store has chunks
        collection_stats = vector_store.get_collection_stats()
        total_chunks = collection_stats['total_chunks']
        if total_chunks == 0:
            raise HTTPException(
                status_code=400,
                detail="No chunks found in the vector store. Please run /process-and-embed/ first."
            )
        
        # Load document chunks
        max_peek = total_chunks
        chunks_raw = vector_store.collection.peek(max_peek)
        document_chunks = [
            {
                'id': chunks_raw['ids'][i],
                'content': chunks_raw['documents'][i]
            } for i in range(len(chunks_raw['ids']))
        ]
        
        # Initialize teacher model
        teacher_model = TeacherModelFactory.create_model(
            'groq',
            model_name='llama3-70b-8192',
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize dataset builder
        dataset_builder = TrainingDatasetBuilder(
            teacher_model=teacher_model,
            vector_store=vector_store,
            output_dir="training_data"
        )
        
        # Generate training dataset
        logger.info("Starting training dataset generation...")
        training_examples = await dataset_builder.build_training_dataset(
            document_chunks=document_chunks,
            queries_per_chunk=request.queries_per_chunk,
            cross_chunk_queries_count=request.cross_chunk_queries_count,
            min_confidence=request.min_confidence
        )
        
        # Save dataset
        dataset_builder.save_dataset(training_examples)
        
        # Get dataset statistics
        stats_file = Path("training_data") / "dataset_statistics.json"
        with open(stats_file, 'r') as f:
            dataset_stats = json.load(f)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully generated training dataset with {len(training_examples)} examples",
                "example_count": len(training_examples),
                "dataset_statistics": dataset_stats,
                "output_directory": str(Path("training_data").absolute())
            }
        )
    except Exception as e:
        logger.error(f"Error generating training dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating training dataset: {str(e)}")

@app.post("/train-and-evaluate/")
async def train_and_evaluate(request: TrainAndEvaluateRequest):
    """
    Endpoint to train the student model and evaluate it against the teacher model
    """
    try:
        # Initialize vector store
        vector_store = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
        
        # Check if vector store has chunks
        collection_stats = vector_store.get_collection_stats()
        total_chunks = collection_stats['total_chunks']
        if total_chunks == 0:
            raise HTTPException(
                status_code=400,
                detail="No chunks found in the vector store. Please run /process-and-embed/ first."
            )
        
 
        teacher_model = TeacherModelFactory.create_model(
            'groq',
            model_name='llama3-70b-8192',
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            teacher_model=teacher_model,
            retrieval_k=3,
            max_context_tokens=2000
        )
        
        # Initialize student model trainer
        student_trainer = StudentModelTrainer(
            model_name=request.model_name,
            output_dir=request.output_dir
        )
        
        # Check for required datasets
        train_dataset_path = Path("training_data") / "train_dataset.jsonl"
        val_dataset_path = Path("training_data") / "validation_dataset.jsonl"
        test_dataset_path = Path("training_data") / "test_dataset.jsonl"
        
        if not train_dataset_path.exists() or not val_dataset_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Training/validation dataset not found. Please run /generate-training-dataset/ first."
            )
        
        # Train student model
        logger.info("Starting student model training...")
        student_trainer.train(
            train_dataset_path=str(train_dataset_path),
            val_dataset_path=str(val_dataset_path),
            epochs=request.epochs,
            batch_size=request.batch_size
        )
        
        # Evaluate student vs teacher
        logger.info("Evaluating student model against teacher model...")
        comparator = ModelComparator(student_trainer, rag_pipeline)
        if not test_dataset_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Test dataset not found. Please run /generate-training-dataset/ first."
            )
        
        results = await comparator.evaluate_dataset(str(test_dataset_path))
        output_file =Path("training_data") / "comparison_report.json"
        comparator.generate_comparison_report(results, str(output_file))
        
        # Load comparison report
        with open(output_file, 'r') as f:
            comparison_report = json.load(f)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully trained student model and evaluated against teacher model",
                "model_output_dir": str(Path(request.output_dir) / "final_model"),
                "comparison_report": comparison_report,
                "report_file": str(output_file)
            }
        )
    except Exception as e:
        logger.error(f"Error in training and evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in training and evaluation: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}