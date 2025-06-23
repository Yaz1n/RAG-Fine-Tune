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
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm
import logging

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
from chromadb import PersistentClient

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

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        bleu_scores = []
        rouge_scores = []
        relevancy_scores = []

        # It's crucial that predictions and references passed to this method
        # already contain only strings. The str() casts below are a secondary
        # safeguard for individual items, but the main lists should be clean.
        for pred_item, ref_item in zip(predictions, references):
            # These casts are good, but the main issue was the lists themselves
            # containing non-string elements before this loop.
            pred = str(pred_item)
            ref = str(ref_item)

            bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=self.smooth)
            rouge = self.rouge.score(ref, pred)['rougeL'].fmeasure
            
            pred_emb = self.sentence_model.encode([pred], convert_to_tensor=True, device=self.device)
            ref_emb = self.sentence_model.encode([ref], convert_to_tensor=True, device=self.device)
            sim = util.pytorch_cos_sim(pred_emb, ref_emb).item()

            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            relevancy_scores.append(sim)

        # The core problem was here: bert_score expects a list of strings.
        # If any element in predictions or references was still a tuple,
        # it would cause the error within bert_score's internals.
        # We ensure the lists are pure strings *before* calling evaluate.
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False, device=self.device if self.device == "cuda" else None)
        
        return {
            "bleu": float(np.mean(bleu_scores)),
            "rougeL": float(np.mean(rouge_scores)),
            "relevancy_sim": float(np.mean(relevancy_scores)),
            "bert_f1": float(F1.mean().item())
        }

def load_jsonl(path) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def print_metrics_table(metrics_dict: Dict[str, Dict[str, float]]):
    print("\n=== Model Evaluation Results ===")
    print(f"{'Model':<18} {'BLEU':>7} {'ROUge-L':>10} {'RelSim':>10} {'BERTScore F1':>13}")
    print("-" * 58)
    for model, scores in metrics_dict.items():
        print(f"{model:<18} {scores['bleu']:7.4f} {scores['rougeL']:10.4f} {scores['relevancy_sim']:10.4f} {scores['bert_f1']:13.4f}")

# ----------------- Manual Evaluation Function ------------------

async def manual_query_evaluation(student_model, student_tokenizer, teacher_model, student_config, top_k=3):
    from chromadb import Client
    from chromadb.config import Settings

    print("\n--- Manual Query Evaluation ---")
    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("No query entered. Skipping.")
        return

    # Embed query
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embed_model.encode([user_query]).tolist()[0]

    # Load vector store
    client = PersistentClient(path="vectorstore")
    collection = client.get_collection("rag_demo")

    results = collection.query(
        query_texts=[user_query],
        n_results=top_k
    )

    if not results["documents"] or not results["documents"][0]:
        print("No relevant context found for the query.")
        return

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
            max_new_tokens=128
        )
    student_answer = student_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Teacher generation
    teacher_answer = await teacher_model.generate_response(prompt)

    # Print answers
    print("\n=== Retrieved Context ===")
    print(context)
    print("\n=== Student Model Answer ===")
    print(student_answer.strip())
    print("\n=== Teacher Model Answer ===")
    print(teacher_answer.strip())

    # Evaluation
    print("\n--- Evaluation (Student vs Teacher) ---")
    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Ensure answers are strings before splitting
    student_answer_str = str(student_answer)
    teacher_answer_str = str(teacher_answer)

    bleu = sentence_bleu([teacher_answer_str.split()], student_answer_str.split(), smoothing_function=smooth)
    rouge_l = rouge.score(teacher_answer_str, student_answer_str)['rougeL'].fmeasure
    
    # Encode on the correct device for manual evaluation
    embed_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_emb = embed_model.encode([student_answer_str], convert_to_tensor=True, device=embed_model_device)
    teacher_emb = embed_model.encode([teacher_answer_str], convert_to_tensor=True, device=embed_model_device)
    
    rel_sim = util.pytorch_cos_sim(student_emb, teacher_emb).item()
    
    # Pass device to bert_score for manual evaluation
    _, _, F1 = bert_score([student_answer_str], [teacher_answer_str], lang='en', verbose=False, device=embed_model_device if embed_model_device == "cuda" else None)

    print(f"BLEU: {bleu:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")
    print(f"Relevance Cosine Similarity: {rel_sim:.4f}")
    print(f"BERTScore F1: {F1.mean().item():.4f}")

# ----------------- Main Evaluation ------------------

async def main_async():
    test_dataset = "training_data/test_dataset.jsonl"

    from phase4 import get_student_config, load_student_model
    student_config = get_student_config()
    student_model, student_tokenizer = load_student_model(student_config)
    student_ckpt = "./student_flan_t5_small/checkpoints/best_model_epoch3.pt"
    if Path(student_ckpt).exists():
        student_model.load_state_dict(torch.load(student_ckpt, map_location="cpu"))
        logger.info(f"Loaded student checkpoint from {student_ckpt}")
    student_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)

    from phase2 import TeacherModelFactory, GroqTeacherModel
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set. Please set it before running.")
        return

    teacher_model = TeacherModelFactory.create_model(
        'groq',
        model_name='llama3-8b-8192', # Or 'mixtral-8x7b-32768' based on your preference
        groq_api_key=groq_api_key
    )
    if not isinstance(teacher_model, GroqTeacherModel):
        logger.error("Failed to initialize GroqTeacherModel. Check TeacherModelFactory implementation.")
        return

    test_samples = load_jsonl(test_dataset)
    queries = [ex["query"] for ex in test_samples]
    contexts = [ex["context"] for ex in test_samples]
    
    # CRITICAL FIX: Ensure all lists passed to evaluator.evaluate contain only strings
    # The `map(str, ...)` ensures that if for any reason an element became a tuple,
    # it is converted to its string representation before evaluation metrics are computed.
    references = list(map(str, [ex["teacher_response"] for ex in test_samples]))

    logger.info("Generating predictions with student model...")
    student_preds = []
    with torch.no_grad():
        for query, context in tqdm(zip(queries, contexts), total=len(queries)):
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
            outputs = student_model.base_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=128
            )
            answer = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
            student_preds.append(str(answer)) # Ensure student predictions are strings

    logger.info("Generating predictions with teacher model...")
    teacher_preds = []
    for query, context in tqdm(zip(queries, contexts), total=len(queries)):
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        resp = await teacher_model.generate_response(prompt)
        teacher_preds.append(str(resp)) # Ensure teacher predictions are strings

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

if __name__ == "__main__":
    asyncio.run(main_async())