#!/usr/bin/env python3
"""
Phase 5: Distillation Training Pipeline for Flan-T5-Small
Supports both normal and retrieval-aware student models.
Aligns with Phases 1-4; focuses on training a single student model.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from datetime import datetime

# === Phase 1: VectorStore / Document chunks ===
# Assume you have a vector store and document_chunks saved from previous phases

# === Phase 2: Teacher Model ===
# Assume teacher responses have been generated and are available in the Phase 3 dataset

# === Phase 3: Training Data ===
# Use the processed, filtered, and split dataset (query, context, answer triples)

# === Phase 4: Student Model Config ===
from phase4 import (
    # Import normal student functions
    get_normal_student_config,
    load_normal_student_model,
    save_normal_pipeline_config
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def create_output_dirs(output_dir):
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/checkpoints").mkdir(exist_ok=True)
    Path(f"{output_dir}/logs").mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Flan-T5-Small Distillation Training")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training dataset (jsonl)")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to evaluation dataset (jsonl)")
    parser.add_argument("--output-dir", type=str, default="./student_flan_t5_small", help="Directory for checkpoints, logs, configs")
    parser.add_argument("--model-type", type=str, default="normal")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--retrieval-dim", type=int, default=384, help="Only used for retrieval-aware model")
    parser.add_argument("--fusion-method", type=str, default="concat", choices=["concat", "attention"], 
                       help="Only used for retrieval-aware model")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    # === Output Dir Setup ===
    create_output_dirs(args.output_dir)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model type: {args.model_type}")

    # === Phase 4: Student Model Config and Load ===
      # normal
    config = get_normal_student_config()
    config.max_seq_length = args.max_length
        # Save config
    config_path = Path(args.output_dir) / f"{config.model_name}_config.json"
    with open(config_path, "w") as f:
            from dataclasses import asdict
            json.dump(asdict(config), f, indent=2)
    logger.info(f"Saved student config to {config_path}")
        
    model, tokenizer = load_normal_student_model(config)
    save_normal_pipeline_config(config.model_name, tokenizer, args.output_dir)
    logger.info("Loaded normal flan-t5-small student model and configs.")

    # === Load Training and Eval Data ===
    train_data = load_jsonl(args.train_data)
    eval_data = load_jsonl(args.eval_data)
    logger.info(f"Training examples: {len(train_data)}, Eval examples: {len(eval_data)}")

    # === Build PyTorch Dataset and DataLoader ===
    from torch.utils.data import Dataset, DataLoader

    class DistillationDataset(Dataset):
        """
        Supports both normal and retrieval-aware training.
        Expects each example as:
            {
                "query": str,
                "context": str (optional for normal model),
                "teacher_response": str,
                "retrieval_embedding": [float, ...] (optional, only used for retrieval-aware)
            }
        """
        def __init__(self, data, tokenizer, max_length, model_type, retrieval_dim=384):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.model_type = model_type
            self.retrieval_dim = retrieval_dim

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            ex = self.data[idx]
            
            # Format input based on model type
            if self.model_type == "retrieval-aware":
                input_text = f"Context: {ex.get('context', '')}\n\nQuestion: {ex['query']}\n\nAnswer:"
            else:  # normal
                input_text = f"Question: {ex['query']}\n\nAnswer:"
            
            target_text = ex['teacher_response']

            # Tokenize inputs and labels
            tokenized = self.tokenizer(
                input_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            labels = self.tokenizer(
                target_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )["input_ids"]

            result = {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0)
            }

            # Add retrieval embeddings only for retrieval-aware model
            if self.model_type == "retrieval-aware":
                retrieval_embedding = ex.get("retrieval_embedding", None)
                if retrieval_embedding is not None:
                    retrieval_embedding = torch.tensor(retrieval_embedding, dtype=torch.float).unsqueeze(0)
                    # Shape: (1, retrieval_dim)
                else:
                    retrieval_embedding = torch.zeros((1, self.retrieval_dim), dtype=torch.float)
                result["retrieval_embeddings"] = retrieval_embedding

            return result

    train_dataset = DistillationDataset(train_data, tokenizer, args.max_length, args.model_type, args.retrieval_dim)
    eval_dataset = DistillationDataset(eval_data, tokenizer, args.max_length, args.model_type, args.retrieval_dim)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    # === Optimizer/Training Setup ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    logger.info("Starting training loop...")
    best_eval_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            # Prepare model inputs based on model type
            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"]
            }
            
            # Add retrieval embeddings only for retrieval-aware model
            if args.model_type == "retrieval-aware":
                model_inputs["retrieval_embeddings"] = batch["retrieval_embeddings"]

            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**model_inputs)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**model_inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                logger.info(f"Epoch {epoch+1} Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # === Evaluation ===
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Prepare model inputs based on model type
                model_inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"]
                }
                
                # Add retrieval embeddings only for retrieval-aware model
                if args.model_type == "retrieval-aware":
                    model_inputs["retrieval_embeddings"] = batch["retrieval_embeddings"]
                
                outputs = model(**model_inputs)
                eval_loss += outputs.loss.item()
                
        avg_eval_loss = eval_loss / len(eval_loader)
        logger.info(f"Epoch {epoch+1} Eval Loss: {avg_eval_loss:.4f}")

        # === Save best model checkpoint ===
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            save_path = os.path.join(args.output_dir, "checkpoints", f"best_model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best model checkpoint to {save_path}")

    logger.info("Training complete.")
    logger.info(f"Best eval loss: {best_eval_loss:.4f}")
    logger.info(f"Model type trained: {args.model_type}")

if __name__ == "__main__":
    main()