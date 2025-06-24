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
    """Configuration dataclass that stores all the settings for the student model setup."""
    base_model: str
    model_name: str
    max_seq_length: int
    device_map: str
    torch_dtype: str
    custom_components: Dict[str, Any]

def get_normal_student_config() -> StudentModelConfig:
    """
    Returns the configuration for flan-t5-small as a normal student (no retrieval).
    """
    return StudentModelConfig(
        base_model="google/flan-t5-small",
        model_name="student_flan-t5-small_normal",
        max_seq_length=512,
        device_map="auto",
        torch_dtype="float32",
        custom_components={
            "retrieval_aware": False,  # Set to False for normal model
        }
    )

def load_normal_student_model(config: StudentModelConfig) -> Tuple[nn.Module, Any]:
    """
    Loads flan-t5-small as a normal student model (without retrieval awareness).
    """
    logger.info(f"Loading normal student model: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True
    )
    
    # Load the base model directly without any wrapper
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map,
        trust_remote_code=True
    )
    
    tokenizer.model_max_length = config.max_seq_length
    logger.info("Loaded normal flan-t5-small model.")
    
    return model, tokenizer

def save_normal_pipeline_config(model_name: str, tokenizer, output_dir: str):
    """Save configuration for normal (non-RAG) pipeline"""
    normal_cfg = {
        "model_name": model_name,
        "generation": {
            "max_new_tokens": 1000,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True
        },
        "prompt_template": {
            "system": "You are a helpful assistant that answers questions accurately and concisely.",
            "input_template": "Question: {question}\n\nAnswer:",
            "max_input_length": tokenizer.model_max_length - 100
        }
    }
    
    cfg_path = Path(output_dir) / f"{model_name}_pipeline_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(normal_cfg, f)
    logger.info(f"Saved normal pipeline config to {cfg_path}")

def main():
    """Main function to set up normal student model"""
    output_dir = "./student_flan_t5_small_normal"
    
    # Get normal student config (no retrieval)
    config = get_normal_student_config()
    
    # Save config
    Path(output_dir).mkdir(exist_ok=True)
    config_path = Path(output_dir) / f"{config.model_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Saved student config to {config_path}")
    
    # Load model and tokenizer
    model, tokenizer = load_normal_student_model(config)
    
    # Save pipeline config
    save_normal_pipeline_config(config.model_name, tokenizer, output_dir)
    
    print(f"\nNormal student model flan-t5-small is ready!")
    print(f"Configs and artifacts saved in {output_dir}/")
    print("This model works independently without requiring external retrieval.")

if __name__ == "__main__":
    main()