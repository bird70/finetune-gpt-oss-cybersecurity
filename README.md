# GPT-OSS20B Cybersecurity Fine-tuning

This repository contains code for fine-tuning OpenAI's GPT-OSS20B model for cybersecurity applications using PEFT/LoRA techniques - an experiment I wanted to make after going through the DeepLearning.ai/AWS training "Generative AI with Large Language Models".


**Note**: This implementation assumes access to appropriate computational resources and datasets. Adjust batch sizes, sequence lengths, and other parameters based on your available hardware and specific requirements.

**Please make sure to follow the permitted uses for the Trendyol dataset and any other datasets you use in your fine-tuning process.**
[https://huggingface.co/Trendyol/Trendyol-Cybersecurity-LLM-Qwen3-32B-Q8_0-GGUF]
This example is for educational purposes and may require further adjustments based on your specific use case and environment.

## Requirements

- CUDA-capable GPU with at least 24GB VRAM (recommended)
- Python 3.8+
- CUDA 11.8+ and compatible PyTorch installation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For CUDA support, ensure you have the correct PyTorch version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Create Sample Configuration
```bash
python fine_tune_gpt_oss_cybersecurity.py --create-sample-config
```

### 2. Create Sample Dataset (for testing)
```bash
python fine_tune_gpt_oss_cybersecurity.py --create-sample-dataset sample_dataset.jsonl
```

### 3. Train the Model
```bash
python fine_tune_gpt_oss_cybersecurity.py --config config.yaml --dataset your_dataset.jsonl
```

### 4. Run Inference
```bash
python fine_tune_gpt_oss_cybersecurity.py --inference ./gpt-oss-cybersecurity-lora --prompt "What are the signs of a phishing attack?"
```

## Dataset Format

The script supports multiple dataset formats:

### JSONL Format (recommended)
```json
{"instruction": "What is a SQL injection?", "response": "SQL injection is a code injection technique..."}
{"question": "How to prevent XSS?", "answer": "To prevent XSS attacks, you should..."}
```

### JSON Format
```json
[
  {"instruction": "...", "response": "..."},
  {"question": "...", "answer": "..."}
]
```

## Configuration Options

Key configuration parameters in `config.yaml`:

- `model_name`: Base model to fine-tune
- `batch_size`: Training batch size (reduce if memory issues)
- `gradient_accumulation_steps`: Effective batch size multiplier
- `lora_r`: LoRA rank (higher = more parameters, better quality)
- `learning_rate`: Training learning rate
- `epochs`: Number of training epochs

## Memory Optimization

For limited GPU memory:

1. Reduce `batch_size` to 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_length` to 256 or 128
4. Use 4-bit quantization instead of 8-bit (modify code)

## Hardware Requirements

- **Minimum**: 16GB GPU memory with optimizations
- **Recommended**: 24GB+ GPU memory
- **CPU fallback**: Possible but extremely slow

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size and max_length
2. **Model loading fails**: Check model name and internet connection
3. **Dataset errors**: Verify dataset format and file paths
4. **Slow training**: Ensure CUDA is properly installed and detected

## Example Output

```
Query: What are the key indicators of a potential SQL injection attack?
Response: Key indicators of SQL injection attacks include: 1) Unusual database queries in logs, 2) Error messages revealing database structure, 3) Unexpected application behavior when special characters are entered...
```

## Files Generated

After training:
- `./gpt-oss-cybersecurity-lora/`: Model files
- `./gpt-oss-cybersecurity-lora/adapter_model.bin`: LoRA weights
- `./gpt-oss-cybersecurity-lora/tokenizer.json`: Tokenizer files
