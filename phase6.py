"""
Phase 6: Model Evaluation Framework
Compare trained retrieval-aware flan-t5-small (student, Phase 5) against the teacher model from Phase 2.
Includes manual user query evaluation with retrieval.
"""

import json
import torch
import numpy as np
import os
import asyncio
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
import logging

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
from chromadb import PersistentClient
# Import VectorStore and DocumentChunk from phase1.py
from phase1 import VectorStore 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evaluation Utility
class SimpleEvaluator:
    def __init__(self, device: str = "cpu"):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.smooth = SmoothingFunction().method1
        self.device = device
        # Ensure sentence_model is loaded to the correct device if CUDA is available
        if torch.cuda.is_available() and self.device == "cuda":
            self.sentence_model.to(self.device)


    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        bleu_scores = []
        rouge_scores = []
        relevancy_scores = []

        # Ensure predictions and references are not empty to avoid division by zero or errors
        if not predictions or not references:
            logger.warning("No predictions or references for evaluation.")
            return {
                "bleu": 0.0,
                "rougeL": 0.0,
                "relevancy_sim": 0.0,
                "bert_f1": 0.0
            }

        for pred_item, ref_item in zip(predictions, references):
            pred = str(pred_item)
            ref = str(ref_item)

            if not pred.strip() or not ref.strip(): # Skip empty strings for scores that would error
                continue

            bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=self.smooth)
            rouge = self.rouge.score(ref, pred)['rougeL'].fmeasure
            
            pred_emb = self.sentence_model.encode([pred], convert_to_tensor=True, device=self.device)
            ref_emb = self.sentence_model.encode([ref], convert_to_tensor=True, device=self.device)
            sim = util.pytorch_cos_sim(pred_emb, ref_emb).item()

            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            relevancy_scores.append(sim)

        # Handle cases where scores might be empty if all predictions/references were empty
        avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
        avg_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
        avg_relevancy = float(np.mean(relevancy_scores)) if relevancy_scores else 0.0
        
        # BERTScore needs a specific handling for empty lists
        bert_f1 = 0.0
        if predictions and references: # Only compute if there are actual texts
            # Filter out empty strings from predictions and references for bert_score
            non_empty_predictions = [p for p in predictions if p.strip()]
            non_empty_references = [r for r in references if r.strip()]

            if non_empty_predictions and non_empty_references:
                try:
                    # BertScore sometimes struggles with very short or trivial inputs.
                    # It's better to pass non-empty lists.
                    P, R, F1 = bert_score(non_empty_predictions, non_empty_references, lang='en', verbose=False, device=self.device if self.device == "cuda" else None)
                    bert_f1 = float(F1.mean().item())
                except Exception as e:
                    logger.warning(f"BERTScore computation failed: {e}. Setting BERTScore F1 to 0.0.")
                    bert_f1 = 0.0
            else:
                 logger.warning("Skipping BERTScore due to empty non-filtered predictions or references.")
                 bert_f1 = 0.0
        
        return {
            "bleu": avg_bleu,
            "rougeL": avg_rouge,
            "relevancy_sim": avg_relevancy,
            "bert_f1": bert_f1
        }

def load_jsonl(path) -> List[Dict]:
    """Loads a JSONL file into a list of dictionaries."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        logger.error(f"Error: Dataset file not found at {path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {path}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {path}: {e}")
        return []


def print_metrics_table(metrics_dict: Dict[str, Dict[str, float]]):
    """Prints evaluation metrics in a formatted table."""
    print("\n=== Model Evaluation Results ===")
    print(f"{'Model':<18} {'BLEU':>7} {'ROUGE-L':>10} {'RelSim':>10} {'BERTScore F1':>13}")
    print("-" * 58)
    for model, scores in metrics_dict.items():
        print(f"{model:<18} {scores.get('bleu', 0.0):7.4f} {scores.get('rougeL', 0.0):10.4f} {scores.get('relevancy_sim', 0.0):10.4f} {scores.get('bert_f1', 0.0):13.4f}")

# ----------------- Manual Evaluation Function ------------------

async def manual_query_evaluation(student_model, student_tokenizer, teacher_model, student_config, top_k=2):
    """Allows for interactive manual querying and evaluation of models."""
    print("\n--- Manual Query Evaluation ---")
    while True: # Loop for continuous manual queries
        user_query = input("Enter your query (type 'quit' to exit): ").strip()
        if user_query.lower() == 'quit':
            print("Exiting manual query evaluation.")
            break # Exit the loop

        if not user_query:
            print("Please enter a query.")
            continue

        # Embed query
        # Re-initialize or ensure embed_model is available here if it's not a global or passed
        embed_model = SentenceTransformer("all-MiniLM-L6-v2") # Consider initializing once globally if performance is critical
        query_embedding = embed_model.encode([user_query]).tolist()[0]

        # Load vector store - CORRECTED PATH HERE
        try:
            # Use the same persist_directory as defined in phase1.py
            # Ensure VectorStore is correctly imported from phase1.py
            vector_store_instance = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
            collection = vector_store_instance.collection # Access the collection directly
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB vectorstore: {e}")
            logger.error("Please ensure the vectorstore 'rag_demo' exists and is properly initialized (e.g., by running Phase 1).")
            continue # Go back to asking for query

        results = collection.query(
            query_embeddings=[query_embedding], # Use query_embeddings for numeric queries
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Ensure results are not empty
        if not results.get("documents") or not results["documents"][0]:
            print("No relevant context found for the query.")
            context = ""
        else:
            context = "\n\n".join(results["documents"][0])


        prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"

        # Student generation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = student_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=student_config.max_seq_length
        )
        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            outputs = student_model.base_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=128,
                # Optional: Add sampling parameters for more varied responses
                # do_sample=True, 
                # top_k=50, 
                # top_p=0.95, 
                # temperature=0.7 
            )
        student_answer = student_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Teacher generation - CORRECTED HERE TO UNPACK TUPLE
        # teacher_model.generate_response returns (response_text, tokens_used)
        teacher_answer_tuple = await teacher_model.generate_response(prompt)
        teacher_answer = teacher_answer_tuple[0] # Get the response string
        teacher_tokens_used = teacher_answer_tuple[1] # Get tokens used (optional for printing)


        # Print answers
        print("\n=== Retrieved Context ===")
        print(context if context else "[No context retrieved]")
        print("\n=== Student Model Answer ===")
        print(student_answer.strip())
        print("\n=== Teacher Model Answer ===")
        print(teacher_answer.strip()) # Now teacher_answer is a string, .strip() works

        # Evaluation
        print("\n--- Evaluation (Student vs Teacher) ---")
        evaluator = SimpleEvaluator(device=str(device))

        student_answer_str = str(student_answer)
        teacher_answer_str = str(teacher_answer)

        # Evaluate student answer against teacher answer as reference
        # Pass as lists, even if single items
        metrics = evaluator.evaluate([student_answer_str], [teacher_answer_str])

        print(f"BLEU: {metrics.get('bleu', 0.0):.4f}")
        print(f"ROUGE-L: {metrics.get('rougeL', 0.0):.4f}")
        print(f"Relevance Cosine Similarity: {metrics.get('relevancy_sim', 0.0):.4f}")
        print(f"BERTScore F1: {metrics.get('bert_f1', 0.0):.4f}")
        print("\n" + "="*50 + "\n")
    

# ----------------- Main Evaluation ------------------

async def main_async():
    """Main function for comprehensive model evaluation."""
    test_dataset = "training_data/test_dataset.jsonl"
    
    # Check if test_dataset file exists before proceeding
    if not Path(test_dataset).exists():
        logger.error(f"Test dataset not found at {test_dataset}. Please ensure Phase 3 (training data generation) has been run.")
        return None, None, None, None # Return None if dataset is missing

    # Load student model and tokenizer from Phase 4
    from phase4 import get_student_config, load_student_model
    student_config = get_student_config()
    student_model, student_tokenizer = load_student_model(student_config)
    student_ckpt = "./student_flan_t5_small/checkpoints/best_model_epoch3.pt"
    if Path(student_ckpt).exists():
        student_model.load_state_dict(torch.load(student_ckpt, map_location="cpu"))
        logger.info(f"Loaded student checkpoint from {student_ckpt}")
    else:
        logger.warning(f"Student model checkpoint not found at {student_ckpt}. Student model will use initial weights.")
    
    student_model.eval() # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    logger.info(f"Student model loaded and moved to device: {device}")

    # Initialize Teacher Model (Groq) from Phase 2
    from phase2 import TeacherModelFactory, GroqTeacherModel
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set. Please set it before running.")
        return None, None, None, None # Return None if API key is missing

    teacher_model = TeacherModelFactory.create_model(
        'groq',
        model_name='llama3-8b-8192', # Or 'mixtral-8x7b-32768' based on your preference
        groq_api_key=groq_api_key
    )
    if not isinstance(teacher_model, GroqTeacherModel):
        logger.error("Failed to initialize GroqTeacherModel. Check TeacherModelFactory implementation in phase2.py.")
        return None, None, None, None # Return None if teacher model initialization fails

    test_samples = load_jsonl(test_dataset)
    if not test_samples:
        logger.error("Test dataset is empty or failed to load. Cannot proceed with evaluation.")
        return student_model, student_tokenizer, teacher_model, student_config # Return initialized models for manual eval option


    queries = [ex["query"] for ex in test_samples]
    # Ensure contexts are properly retrieved from test_samples if they are part of the dataset
    # If not, you might need to re-retrieve contexts using the VectorStore here for a fair comparison.
    # For now, assuming context is in test_samples as generated by Phase 3.
    contexts = [ex["context"] for ex in test_samples] 
    
    references = list(map(str, [ex["teacher_response"] for ex in test_samples]))

    logger.info("Generating predictions with student model...")
    student_preds = []
    # Ensure a local VectorStore instance is available if contexts aren't pre-saved or need to be dynamic
    # For this current setup (context already in test_samples), no new VectorStore search is needed for student_preds
    with torch.no_grad():
        for query, context in tqdm(zip(queries, contexts), total=len(queries), desc="Student model inference"):
            # Ensure prompt doesn't exceed max_length with context
            # A simple way is to truncate context if it's too long
            # However, `padding="max_length"` and `truncation=True` in tokenizer handle this.
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            batch = student_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding="max_length", # Or 'longest' if you have varying prompt lengths and want to batch
                max_length=student_config.max_seq_length
            )
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = student_model.base_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=128,
                # Optional: Add sampling parameters for more varied responses
                # do_sample=True, 
                # top_k=50, 
                # top_p=0.95, 
                # temperature=0.7 
            )
            answer = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
            student_preds.append(str(answer))

    logger.info("Generating predictions with teacher model...")
    teacher_preds = []
    for query, context in tqdm(zip(queries, contexts), total=len(queries), desc="Teacher model inference"):
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        # CORRECTED HERE TO UNPACK TUPLE
        resp_tuple = await teacher_model.generate_response(prompt)
        resp = resp_tuple[0] # Get the response string
        teacher_preds.append(str(resp)) # Append the string content

    evaluator = SimpleEvaluator(device=str(device))
    student_metrics = evaluator.evaluate(student_preds, references)
    teacher_metrics = evaluator.evaluate(teacher_preds, references)

    metrics_dict = {
        "Student_flan-t5-small": student_metrics,
        "Teacher_Groq_Llama3": teacher_metrics
    }
    print_metrics_table(metrics_dict)

    results = {
        "student_preds": student_preds,
        "teacher_preds": teacher_preds,
        "references": references,
        "metrics": metrics_dict
    }
    output_path = Path("evaluation_result/phase6_eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results to {output_path}")
    
    return student_model, student_tokenizer, teacher_model, student_config

if __name__ == "__main__":
    # Run the main asynchronous evaluation to get models and metrics
    student_model_loaded, student_tokenizer_loaded, teacher_model_loaded, student_config_loaded = asyncio.run(main_async())

    # Only offer manual evaluation if models were successfully loaded/initialized
    if student_model_loaded and student_tokenizer_loaded and teacher_model_loaded and student_config_loaded:
        while True:
            try_manual = input("\nDo you want to perform a manual query evaluation? (yes/no): ").strip().lower()
            if try_manual == 'yes':
                # Pass the loaded models and config to the manual evaluation function
                asyncio.run(manual_query_evaluation(
                    student_model_loaded, 
                    student_tokenizer_loaded, 
                    teacher_model_loaded, 
                    student_config_loaded
                ))
                # The manual_query_evaluation function is now designed to loop internally
                # until 'quit' is typed. Once it exits, we break from this outer loop too.
                break 
            elif try_manual == 'no':
                print("Exiting.")
                break
            else:
                print("Invalid input. Please type 'yes' or 'no'.")
    else:
        print("\nSkipping manual evaluation due to previous errors during model/data loading.")