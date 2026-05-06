#!/usr/bin/env python3
"""
Fine-tuning OpenAI GPT-OSS20B for Cybersecurity with PEFT/LoRA

This script implements Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA)
to fine-tune OpenAI's GPT-OSS20B model for cybersecurity applications.

Requirements:
- CUDA-capable GPU with at least 24GB VRAM (recommended)
- Python 3.8+
- See requirements.txt for package dependencies

Usage:
    python fine_tune_gpt_oss_cybersecurity.py --config config.yaml

Date: 2025-08-09
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
import yaml




# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_prompt(example):
    """Format examples for instruction following."""
    # Try different common field combinations
    if "instruction" in example and "response" in example:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    elif "question" in example and "answer" in example:
        prompt = f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
    elif "input" in example and "output" in example:
        prompt = f"### Input:\n{example['input']}\n\n### Output:\n{example['output']}"
    elif "prompt" in example and "completion" in example:
        prompt = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
    elif "text" in example:
        # Use as-is if already formatted
        prompt = example['text']
    else:
        # Fallback: use first two string fields found
        string_fields = [k for k, v in example.items() if isinstance(v, str)]
        if len(string_fields) >= 2:
            prompt = f"### Input:\n{example[string_fields[0]]}\n\n### Output:\n{example[string_fields[1]]}"
        else:
            logger.warning(f"Could not format example: {example.keys()}")
            prompt = str(example)
    
    return {"text": prompt}


class CybersecurityFineTuner:
    """Fine-tuning class for cybersecurity-specific GPT-OSS20B model."""
    
    def __init__(self, config: Dict):
        """Initialize the fine-tuner with configuration."""
        self.config = config
        self.model_name = config.get('model_name', 'unsloth/gpt-oss-20b')
        self.output_dir = config.get('output_dir', './gpt-oss-cybersecurity-lora')
        self.max_length = config.get('max_length', 2048)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def load_model(self) -> None:
        """Load the base model with quantization."""
        logger.info(f"Loading model from {self.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_length,
            dtype = None,
            load_in_4bit = True,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = self.config.get('lora_r', 16),
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_alpha = self.config.get('lora_alpha', 32),
            lora_dropout = self.config.get('lora_dropout', 0.1),
            bias = "none",
            use_gradient_checkpointing = True,
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )
        
        logger.info("Model and tokenizer loaded successfully")

    def load_cybersecurity_dataset(self, dataset_path: str) -> None:
        """Load and prepare the cybersecurity dataset."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        if dataset_path.endswith('.jsonl'):
            self.dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith('.json'):
            self.dataset = load_dataset("json", data_files=dataset_path, split="train")
        else:
            # Try loading from Hugging Face datasets
            self.dataset = load_dataset(dataset_path, split='train')
        
        self.dataset = self.dataset.map(format_prompt)
        logger.info(f"Dataset loaded with {len(self.dataset)} examples")

        if self.config.get('validation_split', 0.1) > 0:
            split_dataset = self.dataset.train_test_split(test_size=self.config.get('validation_split', 0.1))
            self.train_dataset = split_dataset["train"]
            self.eval_dataset = split_dataset["test"]
        else:
            self.train_dataset = self.dataset
            self.eval_dataset = None
    def train(self, dataset_path: str):
        self.load_model()
        self.load_cybersecurity_dataset(dataset_path)

        trainer = UnslothTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.train_dataset,
            eval_dataset = self.eval_dataset,
            args = UnslothTrainingArguments(
                per_device_train_batch_size = self.config.get('batch_size', 2),
                gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 8),
                warmup_steps = self.config.get('warmup_steps', 100),
                num_train_epochs=self.config.get('epochs', 3),
                learning_rate = self.config.get('learning_rate', 2e-4),
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = self.config.get('logging_steps', 1),
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = self.output_dir,
            ),
        )
        trainer.train()

    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Generate response using the fine-tuned model."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded first")
        
        # Format the prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids = inputs["input_ids"].to("cuda"),
                attention_mask = inputs["attention_mask"].to("cuda"),
                max_length=len(inputs['input_ids'][0]) + max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and return response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        generated_text = response[len(formatted_prompt):]
        
        return generated_text.strip()

def load_fine_tuned_model(lora_adapter_path: str) -> Tuple[FastLanguageModel, AutoTokenizer]:
    """Load a fine-tuned model for inference."""
    logger.info(f"Loading fine-tuned model from {lora_adapter_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_adapter_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    return model, tokenizer


def create_sample_config() -> Dict:
    """Create a sample configuration."""
    return {
        "model_name": "unsloth/gpt-oss-20b",
        "output_dir": "./gpt-oss-cybersecurity-lora",
        "max_length": 2048,
        "validation_split": 0.1,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "epochs": 3,
        "learning_rate": 2e-4,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 500,
        "warmup_steps": 100,
    }


def main():
    """Main function to handle command line arguments and execute training."""
    parser = argparse.ArgumentParser(description='Fine-tune GPT-OSS20B for cybersecurity with Unsloth')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--dataset', type=str, help='Path to dataset file or Hugging Face dataset name')
    parser.add_argument('--create-sample-config', action='store_true', 
                       help='Create sample configuration file')
    parser.add_argument('--inference', type=str, help='Path to fine-tuned model for inference')
    parser.add_argument('--prompt', type=str, help='Prompt for inference')
    
    args = parser.parse_args()
    
    # Create sample configuration
    if args.create_sample_config:
        config = create_sample_config()
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info("Sample configuration created as 'config.yaml'")
        return
    
    # Inference mode
    if args.inference:
        if not args.prompt:
            logger.error("--prompt is required for inference")
            sys.exit(1)
        
        model, tokenizer = load_fine_tuned_model(args.inference)
        
        # Create temporary fine-tuner for inference
        config = {"model_name": "unsloth/gpt-oss-20b"}
        fine_tuner = CybersecurityFineTuner(config)
        fine_tuner.model = model
        fine_tuner.tokenizer = tokenizer
        
        response = fine_tuner.generate_response(args.prompt)
        print(f"Query: {args.prompt}")
        print(f"Response: {response}")
        return
    
    # Training mode
    if not args.dataset:
        logger.error("--dataset is required for training")
        sys.exit(1)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_sample_config()
        logger.info("Using default configuration")
    
    # Create and run fine-tuner
    fine_tuner = CybersecurityFineTuner(config)
    fine_tuner.train(args.dataset)

if __name__ == "__main__":
    main()
