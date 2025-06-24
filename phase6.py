"""
Phase 6: Model Evaluation Framework
Compare trained retrieval-aware flan-t5-small (student, Phase 5) against the teacher model from Phase 2.
Includes manual user query evaluation with retrieval.
"""

import json
import torch
import torch.nn as nn # <--- ADDED THIS IMPORT
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

# Import VectorStore from phase1.py
from phase1 import VectorStore 

# Import necessary components from the *new* phase4.py
from phase4 import (
    get_normal_student_config,
    load_normal_student_model,
    StudentModelConfig # Import the class for type hinting
)
# Import TeacherModelFactory and GroqTeacherModel from phase2
from phase2 import TeacherModelFactory, GroqTeacherModel


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
        if torch.cuda.is_available() and self.device == "cuda":
            self.sentence_model.to(self.device)


    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        bleu_scores = []
        rouge_scores = []
        relevancy_scores = []

        if not predictions or not references:
            logger.warning("No predictions or references for evaluation.")
            return {
                "bleu": 0.0,
                "rougeL": 0.0,
                "relevancy_sim": 0.0,
                "bert_f1": 0.0
            }

        filtered_pairs = []
        for pred_item, ref_item in zip(predictions, references):
            pred = str(pred_item).strip()
            ref = str(ref_item).strip()
            if pred and ref: # Only include if both are non-empty
                filtered_pairs.append((pred, ref))
            else:
                logger.debug(f"Skipping empty prediction or reference: Pred='{pred}', Ref='{ref}'")

        if not filtered_pairs:
            logger.warning("All prediction-reference pairs were empty after filtering. Returning zero scores.")
            return {
                "bleu": 0.0,
                "rougeL": 0.0,
                "relevancy_sim": 0.0,
                "bert_f1": 0.0
            }

        filtered_predictions = [p for p, _ in filtered_pairs]
        filtered_references = [r for _, r in filtered_pairs]

        for pred, ref in filtered_pairs:
            bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=self.smooth)
            rouge = self.rouge.score(ref, pred)['rougeL'].fmeasure
            
            pred_emb = self.sentence_model.encode([pred], convert_to_tensor=True, device=self.device)
            ref_emb = self.sentence_model.encode([ref], convert_to_tensor=True, device=self.device)
            sim = util.pytorch_cos_sim(pred_emb, ref_emb).item()

            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            relevancy_scores.append(sim)

        avg_bleu = float(np.mean(bleu_scores))
        avg_rouge = float(np.mean(rouge_scores))
        avg_relevancy = float(np.mean(relevancy_scores))
        
        bert_f1 = 0.0
        try:
            P, R, F1 = bert_score(filtered_predictions, filtered_references, lang='en', verbose=False, device=self.device if self.device == "cuda" else None)
            bert_f1 = float(F1.mean().item())
        except Exception as e:
            logger.warning(f"BERTScore computation failed for {len(filtered_predictions)} pairs: {e}. Setting BERTScore F1 to 0.0.")
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

async def manual_query_evaluation(student_model: nn.Module, student_tokenizer, teacher_model: GroqTeacherModel, student_config: StudentModelConfig, top_k=2):
    """Allows for interactive manual querying and evaluation of models."""
    print("\n--- Manual Query Evaluation ---")
    
    embed_model = SentenceTransformer("all-MiniLM-L6-v2") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model.to(device)

    evaluator = SimpleEvaluator(device=str(device))

    while True: 
        user_query = input("Enter your query (type 'quit' to exit): ").strip()
        if user_query.lower() == 'quit':
            print("Exiting manual query evaluation.")
            break 

        if not user_query:
            print("Please enter a query.")
            continue

        query_embedding = embed_model.encode([user_query], convert_to_tensor=True).tolist()[0]

        try:
            vector_store_instance = VectorStore(collection_name="rag_demo", persist_directory="./data/chroma_db")
            collection = vector_store_instance.collection 
            logger.info("Successfully connected to ChromaDB collection 'rag_demo'.")
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB vectorstore: {e}")
            logger.error("Please ensure the vectorstore 'rag_demo' exists and is properly initialized (e.g., by running Phase 1).")
            continue 

        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results.get("documents") or not results["documents"][0]:
            print("No relevant context found for the query.")
            context = ""
        else:
            context = "\n\n".join(results["documents"][0])

        prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"

        # Student generation
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
            outputs = student_model.generate( # Call generate directly on the base model
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=128,
                do_sample=True, 
                temperature=0.7, 
                top_p=0.9, 
                repetition_penalty=1.1,
                num_return_sequences=1,
            )
        student_answer = student_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Teacher generation
        teacher_answer_tuple = await teacher_model.generate_response(prompt)
        teacher_answer = teacher_answer_tuple[0] 

        # Print answers
        print("\n=== Retrieved Context ===")
        print(context if context else "[No context retrieved]")
        print("\n=== Student Model Answer ===")
        print(student_answer.strip())
        print("\n=== Teacher Model Answer ===")
        print(teacher_answer.strip())

        # Evaluation
        print("\n--- Evaluation (Student vs Teacher) ---")
        student_answer_str = str(student_answer)
        teacher_answer_str = str(teacher_answer)

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
    
    if not Path(test_dataset).exists():
        logger.error(f"Test dataset not found at {test_dataset}. Please ensure Phase 3 (training data generation) has been run.")
        return None, None, None, None 

    # Load student model and tokenizer from Phase 4
    student_config = get_normal_student_config()
    student_model, student_tokenizer = load_normal_student_model(student_config)
    
    student_ckpt = "./student_flan_t5_small_normal/checkpoints/best_model_epoch3.pt" 
    
    if Path(student_ckpt).exists():
        try:
            student_model.load_state_dict(torch.load(student_ckpt, map_location="cpu"))
            logger.info(f"Loaded normal student checkpoint from {student_ckpt}")
        except Exception as e:
            logger.error(f"Error loading student model state_dict from {student_ckpt}: {e}")
            logger.warning("Student model will use initial weights. This might affect evaluation results.")
    else:
        logger.warning(f"Student model checkpoint not found at {student_ckpt}. Student model will use initial weights.")
    
    student_model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    logger.info(f"Student model loaded and moved to device: {device}")

    # Initialize Teacher Model (Groq) from Phase 2
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set. Please set it before running.")
        return None, None, None, None 

    teacher_model = TeacherModelFactory.create_model(
        'groq',
        model_name='llama3-8b-8192', 
        groq_api_key=groq_api_key
    )
    if not isinstance(teacher_model, GroqTeacherModel):
        logger.error("Failed to initialize GroqTeacherModel. Check TeacherModelFactory implementation in phase2.py.")
        return None, None, None, None 

    test_samples = load_jsonl(test_dataset)
    if not test_samples:
        logger.error("Test dataset is empty or failed to load. Cannot proceed with evaluation.")
        return student_model, student_tokenizer, teacher_model, student_config 

    queries = [ex["query"] for ex in test_samples]
    contexts = [ex["context"] for ex in test_samples] 
    
    references = list(map(str, [ex["teacher_response"] for ex in test_samples]))

    logger.info("Generating predictions with student model (normal flan-t5-small)...")
    student_preds = []
    with torch.no_grad():
        for query, context in tqdm(zip(queries, contexts), total=len(queries), desc="Student model inference"):
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            batch = student_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding="max_length", 
                max_length=student_config.max_seq_length
            )
            for k in batch:
                batch[k] = batch[k].to(device)

            outputs = student_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=128,
                do_sample=True, 
                temperature=0.7, 
                top_p=0.9, 
                repetition_penalty=1.1,
                num_return_sequences=1,
            )
            answer = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
            student_preds.append(str(answer))

    logger.info("Generating predictions with teacher model...")
    teacher_preds = []
    for query, context in tqdm(zip(queries, contexts), total=len(queries), desc="Teacher model inference"):
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        resp_tuple = await teacher_model.generate_response(prompt)
        resp = resp_tuple[0] 
        teacher_preds.append(str(resp)) 

    evaluator = SimpleEvaluator(device=str(device))
    student_metrics = evaluator.evaluate(student_preds, references)
    teacher_metrics = evaluator.evaluate(teacher_preds, references)

    metrics_dict = {
        "Student_flan-t5-small_normal": student_metrics, 
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
    student_model_loaded, student_tokenizer_loaded, teacher_model_loaded, student_config_loaded = asyncio.run(main_async())

    if student_model_loaded and student_tokenizer_loaded and teacher_model_loaded and student_config_loaded:
        while True:
            try_manual = input("\nDo you want to perform a manual query evaluation? (yes/no): ").strip().lower()
            if try_manual == 'yes':
                asyncio.run(manual_query_evaluation(
                    student_model_loaded, 
                    student_tokenizer_loaded, 
                    teacher_model_loaded, 
                    student_config_loaded
                ))
                break 
            elif try_manual == 'no':
                print("Exiting.")
                break
            else:
                print("Invalid input. Please type 'yes' or 'no'.")
    else:
        print("\nSkipping manual evaluation due to previous errors during model/data loading.")