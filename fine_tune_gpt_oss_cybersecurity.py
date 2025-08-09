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
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CybersecurityFineTuner:
    """Fine-tuning class for cybersecurity-specific GPT-OSS20B model."""
    
    def __init__(self, config: Dict):
        """Initialize the fine-tuner with configuration."""
        self.config = config
        self.model_name = config.get('model_name', 'openai/gpt-oss-20b')
        self.output_dir = config.get('output_dir', './gpt-oss-cybersecurity-lora')
        self.max_length = config.get('max_length', 512)
        
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("CUDA not available. Training will be very slow on CPU.")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def load_tokenizer(self) -> None:
        """Load and configure the tokenizer."""
        logger.info(f"Loading tokenizer from {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        logger.info("Tokenizer loaded successfully")
    
    def load_model(self) -> None:
        """Load the base model with quantization."""
        logger.info(f"Loading model from {self.model_name}")
        
        # Model loading configuration
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        # Add quantization if CUDA is available
        if self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
            logger.info("Using 8-bit quantization")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        logger.info(f"Model loaded on device: {self.model.device}")
        
    def configure_lora(self) -> None:
        """Configure and apply LoRA to the model."""
        logger.info("Configuring LoRA")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.get('lora_r', 16),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.1),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        logger.info("LoRA configuration applied")
    
    def load_cybersecurity_dataset(self, dataset_path: str) -> None:
        """Load and prepare the cybersecurity dataset."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        if dataset_path.endswith('.jsonl'):
            # Load JSONL file
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            
            self.dataset = Dataset.from_list(data)
            
        elif dataset_path.endswith('.json'):
            # Load JSON file
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                self.dataset = Dataset.from_list(data)
            elif isinstance(data, dict) and 'data' in data:
                self.dataset = Dataset.from_list(data['data'])
            else:
                raise ValueError("Unsupported JSON structure")
                
        else:
            # Try loading from Hugging Face datasets
            self.dataset = load_dataset(dataset_path, split='train')
        
        logger.info(f"Dataset loaded with {len(self.dataset)} examples")
    
    def format_dataset(self) -> None:
        """Format the dataset for instruction following."""
        logger.info("Formatting dataset")
        
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
        
        # Apply formatting
        self.dataset = self.dataset.map(format_prompt)
        logger.info("Dataset formatting completed")
    
    def tokenize_dataset(self) -> Tuple[Dataset, Dataset]:
        """Tokenize the dataset and create train/validation splits."""
        logger.info("Tokenizing dataset")
        
        def tokenize_function(examples):
            """Tokenize the dataset for training."""
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_overflowing_tokens=False,
            )
        
        # Tokenize the dataset
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Create train/validation split
        split_ratio = self.config.get('validation_split', 0.1)
        if split_ratio > 0:
            split_dataset = tokenized_dataset.train_test_split(test_size=split_ratio)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None
        
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> Trainer:
        """Create and configure the trainer."""
        logger.info("Creating trainer")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config.get('batch_size', 2),
            per_device_eval_batch_size=self.config.get('eval_batch_size', 2),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 8),
            num_train_epochs=self.config.get('epochs', 3),
            learning_rate=self.config.get('learning_rate', 2e-4),
            fp16=self.device == "cuda",
            logging_steps=self.config.get('logging_steps', 10),
            eval_steps=self.config.get('eval_steps', 100),
            save_steps=self.config.get('save_steps', 500),
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            warmup_steps=self.config.get('warmup_steps', 100),
            lr_scheduler_type="cosine",
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        return trainer
    
    def train(self, dataset_path: str) -> None:
        """Execute the complete training pipeline."""
        logger.info("Starting training pipeline")
        
        # Load components
        self.load_tokenizer()
        self.load_model()
        self.configure_lora()
        
        # Prepare dataset
        self.load_cybersecurity_dataset(dataset_path)
        self.format_dataset()
        train_dataset, eval_dataset = self.tokenize_dataset()
        
        # Create and run trainer
        trainer = self.create_trainer(train_dataset, eval_dataset)
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training completed successfully!")
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Generate response using the fine-tuned model."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded first")
        
        # Format the prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
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


def load_fine_tuned_model(base_model_path: str, lora_adapter_path: str) -> Tuple[PeftModel, AutoTokenizer]:
    """Load a fine-tuned model for inference."""
    logger.info(f"Loading fine-tuned model from {lora_adapter_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True if torch.cuda.is_available() else False
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    return model, tokenizer


def create_sample_config() -> Dict:
    """Create a sample configuration."""
    return {
        "model_name": "openai/gpt-oss-20b",
        "output_dir": "./gpt-oss-cybersecurity-lora",
        "max_length": 512,
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


def create_sample_dataset(output_path: str) -> None:
    """Create a sample cybersecurity dataset for testing."""
    sample_data = [
        {
            "instruction": "What are the key indicators of a potential SQL injection attack?",
            "response": "Key indicators of SQL injection attacks include: 1) Unusual database queries in logs, 2) Error messages revealing database structure, 3) Unexpected application behavior when special characters are entered, 4) Slow database performance, 5) Authentication bypasses, and 6) Data exfiltration patterns in network traffic."
        },
        {
            "instruction": "How should an organization respond to a ransomware attack?",
            "response": "Ransomware response steps: 1) Immediately isolate affected systems, 2) Activate incident response team, 3) Assess scope of infection, 4) Preserve evidence, 5) Restore from clean backups if available, 6) Report to law enforcement and relevant authorities, 7) Communicate with stakeholders, 8) Conduct post-incident analysis, and 9) Improve security measures based on lessons learned."
        },
        {
            "instruction": "What is the principle of least privilege in cybersecurity?",
            "response": "The principle of least privilege is a security concept where users, applications, and systems are granted only the minimum access rights necessary to perform their functions. This reduces the attack surface by limiting potential damage if credentials are compromised and helps prevent lateral movement in case of a breach."
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Sample dataset created at {output_path}")


def main():
    """Main function to handle command line arguments and execute training."""
    parser = argparse.ArgumentParser(description='Fine-tune GPT-OSS20B for cybersecurity')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--create-sample-config', action='store_true', 
                       help='Create sample configuration file')
    parser.add_argument('--create-sample-dataset', type=str,
                       help='Create sample dataset at specified path')
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
    
    # Create sample dataset
    if args.create_sample_dataset:
        create_sample_dataset(args.create_sample_dataset)
        return
    
    # Inference mode
    if args.inference:
        if not args.prompt:
            logger.error("--prompt is required for inference")
            sys.exit(1)
        
        model, tokenizer = load_fine_tuned_model("openai/gpt-oss-20b", args.inference)
        
        # Create temporary fine-tuner for inference
        config = {"model_name": "openai/gpt-oss-20b"}
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
