import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import logging
from typing import Dict, Any, Tuple, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StudentModelConfig:
    """Configuration for student model setup"""
    base_model: str
    model_name: str
    max_seq_length: int
    device_map: str
    torch_dtype: str
    custom_components: Dict[str, Any]

class RetrievalAwareStudentModel(nn.Module):
    """
    Retrieval-aware wrapper for sequence-to-sequence models (like T5).
    Fuses external retrieval context embeddings with encoder hidden states.
    """
    def __init__(self, base_model, retrieval_dim: int = 384, fusion_method: str = "concat"):
        super().__init__()
        self.base_model = base_model
        self.retrieval_dim = retrieval_dim
        self.fusion_method = fusion_method

        self.hidden_size = base_model.config.d_model

        # Retrieval integration components
        if fusion_method == "concat":
            self.retrieval_projection = nn.Linear(retrieval_dim, self.hidden_size)
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
        elif fusion_method == "attention":
            self.retrieval_projection = nn.Linear(retrieval_dim, self.hidden_size)
            self.retrieval_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=8,
                batch_first=True
            )
        # Additional methods can be added as needed

    def encode_retrieval_context(self, retrieval_embeddings):
        """Encode retrieval context (project to model hidden size)"""
        if retrieval_embeddings is None:
            return None
        projected = self.retrieval_projection(retrieval_embeddings)
        return projected

    def fuse_retrieval_info(self, encoder_hidden_states, retrieval_context):
        if retrieval_context is None:
            return encoder_hidden_states

        if self.fusion_method == "concat":
            batch_size, seq_len, hidden_dim = encoder_hidden_states.shape
            pooled_context = retrieval_context.mean(dim=1, keepdim=True)
            pooled_context = pooled_context.expand(-1, seq_len, -1)
            combined = torch.cat([encoder_hidden_states, pooled_context], dim=-1)
            gate = self.fusion_gate(combined)
            fused = encoder_hidden_states * gate + pooled_context * (1 - gate)
            return fused

        elif self.fusion_method == "attention":
            attended_context, _ = self.retrieval_attention(
                encoder_hidden_states, retrieval_context, retrieval_context
            )
            fused = encoder_hidden_states + attended_context
            return fused

        else:
            return encoder_hidden_states

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        retrieval_embeddings=None,
        **kwargs
    ):
        # Encode inputs
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Encode retrieval and fuse
        retrieval_context = self.encode_retrieval_context(retrieval_embeddings)
        fused_encoder_hidden = self.fuse_retrieval_info(
            encoder_hidden_states, retrieval_context
        )

        # Pass through decoder
        outputs = self.base_model(
            inputs_embeds=fused_encoder_hidden,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs
        )

        return outputs

def get_student_config() -> StudentModelConfig:
    """
    Returns the configuration for flan-t5-small as retrieval-aware student.
    """
    return StudentModelConfig(
        base_model="google/flan-t5-small",
        model_name="student_flan-t5-small_retrievalaware",
        max_seq_length=512,
        device_map="auto",
        torch_dtype="float32",
        custom_components={
            "retrieval_aware": True,
            "fusion_method": "concat",
            "retrieval_dim": 384
        }
    )

def load_student_model(config: StudentModelConfig) -> Tuple[nn.Module, Any]:
    """
    Loads flan-t5-small as retrieval-aware student.
    """
    logger.info(f"Loading student model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map,
        trust_remote_code=True
    )
    # Retrieval-aware wrapper
    if config.custom_components.get("retrieval_aware", False):
        model = RetrievalAwareStudentModel(
            base_model=base_model,
            retrieval_dim=config.custom_components.get("retrieval_dim", 384),
            fusion_method=config.custom_components.get("fusion_method", "concat")
        )
        logger.info("Loaded retrieval-aware flan-t5-small model.")
    else:
        model = base_model
    tokenizer.model_max_length = config.max_seq_length
    return model, tokenizer

def save_student_config(config: StudentModelConfig, output_dir: str):
    Path(output_dir).mkdir(exist_ok=True)
    config_path = Path(output_dir) / f"{config.model_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Saved student config to {config_path}")

def save_rag_pipeline_config(model_name: str, tokenizer, output_dir: str):
    rag_cfg = {
        "model_name": model_name,
        "retrieval": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "retrieval_k": 5,
            "rerank": True
        },
        "generation": {
            "max_new_tokens": 128,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True
        },
        "prompt_template": {
            "system": "You are a helpful assistant that answers questions based on the provided context.",
            "context_template": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            "max_context_length": tokenizer.model_max_length - 100
        }
    }
    cfg_path = Path(output_dir) / f"{model_name}_rag_pipeline_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(rag_cfg, f)
    logger.info(f"Saved RAG pipeline config to {cfg_path}")

def main():
    # ---- PHASE 1 & 2 & 3 INTEGRATION ----
    # Assume you have run Phase 1 (vector store), Phase 2 (teacher model), Phase 3 (training data)
    # Use the output_dir to save configs for downstream training and evaluation

    output_dir = "./student_flan_t5_small"
    config = get_student_config()
    save_student_config(config, output_dir)
    model, tokenizer = load_student_model(config)
    save_rag_pipeline_config(config.model_name, tokenizer, output_dir)

    print(f"\nStudent model flan-t5-small (retrieval-aware) is ready for training!")
    print(f"Configs and artifacts saved in {output_dir}/")
    print("You can now use this model in your training pipeline with the data from Phase 3.")

if __name__ == "__main__":
    main()