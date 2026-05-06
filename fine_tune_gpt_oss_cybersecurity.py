#!/usr/bin/env python3
"""
Fine-tuning Llama 3.2 for Cybersecurity with Unsloth
This script implements Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA)
to fine-tune a Llama 3.2 model for cybersecurity applications.
Requirements:
- CUDA-capable GPU with at least 16GB VRAM (recommended)
- Python 3.8+
- See requirements.txt for package dependencies
Usage:
    python fine_tune_gpt_oss_cybersecurity.py --config config.yaml
Date: 2025-08-10
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Llama-3.2-Instruct template
llama3_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

def format_prompt_llama3(example: Dict) -> Dict:
    """Formats an example for Llama-3.2-Instruct fine-tuning."""
    if "instruction" in example and "response" in example:
        instruction = example['instruction']
        response = example['response']
    elif "question" in example and "answer" in example:
        instruction = example['question']
        response = example['answer']
    elif "input" in example and "output" in example:
        instruction = example['input']
        response = example['output']
    else:
        # Fallback for other formats
        keys = list(example.keys())
        instruction = example[keys[0]]
        response = example[keys[1]]

    return {"text": llama3_template.format(instruction, response)}


class CybersecurityFineTuner:
    """Fine-tuning class for cybersecurity-specific Llama 3.2 model."""
    
    def __init__(self, config: Dict):
        """Initialize the fine-tuner with configuration."""
        self.config = config
        self.model_name = config.get('model_name', 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit')
        self.output_dir = config.get('output_dir', './llama3-cybersecurity-lora')
        self.max_length = config.get('max_length', 1024)
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def load_model(self) -> None:
        """Load the base model with quantization."""
        logger.info(f"Loading model from {self.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_length,
            dtype=None,
            load_in_4bit=True,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.get('lora_r', 16),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.1),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        logger.info("Model and tokenizer loaded successfully")

    def load_and_prepare_datasets(self, downsample_size: int = 5000) -> None:
        """Load, merge, format, and downsample the datasets."""
        dataset_names = [
            "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
            "Cogensec/OptikalLLM_10k_dataset",
            "Mohannadcse/cybersec-reasoning-merged"
        ]
        
        logger.info(f"Loading datasets: {', '.join(dataset_names)}")
        
        datasets = [load_dataset(name, split='train') for name in dataset_names]
        
        # Standardize column names before merging
        def standardize_columns(example: Dict, mapping: Dict) -> Dict:
            return {
                "instruction": example[mapping["instruction"]],
                "response": example[mapping["response"]]
            }

        # Define mappings for each dataset
        mappings = [
            {"instruction": "instruction", "response": "response"},  # Trendyol
            {"instruction": "prompt", "response": "response"},       # OptikalLLM
            {"instruction": "instruction", "response": "response"}   # Cybersec-Reasoning
        ]

        standardized_datasets = []
        for i, ds in enumerate(datasets):
            standardized_datasets.append(ds.map(
                lambda x: standardize_columns(x, mappings[i]),
                remove_columns=ds.column_names
            ))

        merged_dataset = concatenate_datasets(standardized_datasets)
        logger.info(f"Merged dataset size: {len(merged_dataset)}")

        if downsample_size > 0 and len(merged_dataset) > downsample_size:
            merged_dataset = merged_dataset.shuffle(seed=42).select(range(downsample_size))
            logger.info(f"Downsampled dataset to {len(merged_dataset)} samples")

        self.dataset = merged_dataset.map(format_prompt_llama3)
        
        if self.config.get('validation_split', 0.1) > 0:
            split_dataset = self.dataset.train_test_split(test_size=self.config.get('validation_split', 0.1))
            self.train_dataset = split_dataset["train"]
            self.eval_dataset = split_dataset["test"]
        else:
            self.train_dataset = self.dataset
            self.eval_dataset = None
            
        logger.info(f"Final training dataset size: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Final evaluation dataset size: {len(self.eval_dataset)}")

    def train(self):
        self.load_model()
        self.load_and_prepare_datasets(downsample_size=self.config.get('downsample_size', 5000))

        trainer = UnslothTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=UnslothTrainingArguments(
                per_device_train_batch_size=self.config.get('batch_size', 2),
                gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 8),
                warmup_steps=self.config.get('warmup_steps', 100),
                num_train_epochs=self.config.get('epochs', 3),
                learning_rate=self.config.get('learning_rate', 2e-4),
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=self.config.get('logging_steps', 1),
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=self.output_dir,
            ),
        )
        trainer.train()

    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Generate response using the fine-tuned model."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded first")
        
        formatted_prompt = llama3_template.format(prompt, "")
        
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt", truncation=True, max_length=self.max_length).to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        
        return assistant_response


def load_fine_tuned_model(lora_adapter_path: str) -> Tuple[FastLanguageModel, AutoTokenizer]:
    """Load a fine-tuned model for inference."""
    logger.info(f"Loading fine-tuned model from {lora_adapter_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_adapter_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def create_sample_config() -> Dict:
    """Create a sample configuration."""
    return {
        "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "output_dir": "./llama3-cybersecurity-lora",
        "max_length": 1024,
        "validation_split": 0.1,
        "downsample_size": 5000,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "epochs": 3,
        "learning_rate": 2e-4,
        "logging_steps": 10,
        "warmup_steps": 100,
    }


def main():
    """Main function to handle command line arguments and execute training."""
    parser = argparse.ArgumentParser(description='Fine-tune Llama 3.2 for cybersecurity with Unsloth')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--create-sample-config', action='store_true', 
                       help='Create sample configuration file')
    parser.add_argument('--inference', type=str, help='Path to fine-tuned model for inference')
    parser.add_argument('--prompt', type=str, help='Prompt for inference')
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        config = create_sample_config()
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info("Sample configuration created as 'config.yaml'")
        return
    
    if args.inference:
        if not args.prompt:
            logger.error("--prompt is required for inference")
            sys.exit(1)
        
        model, tokenizer = load_fine_tuned_model(args.inference)
        
        config = {"model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"}
        fine_tuner = CybersecurityFineTuner(config)
        fine_tuner.model = model
        fine_tuner.tokenizer = tokenizer
        
        response = fine_tuner.generate_response(args.prompt)
        print(f"Query: {args.prompt}")
        print(f"Response: {response}")
        return
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_sample_config()
        logger.info("Using default configuration for training")
    
    fine_tuner = CybersecurityFineTuner(config)
    fine_tuner.train()

if __name__ == "__main__":
    main()
