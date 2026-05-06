# Llama 3.2 3B Cybersecurity Micro-Fine-Tuning with Unsloth

This repository contains code for "Micro-Fine-Tuning" the Llama 3.2 3B model for cybersecurity applications using Unsloth. This architecture is optimized for low-resource environments, specifically targeting consumer-grade GPUs with as little as 4GB VRAM.

**Note**: This implementation is designed for extreme efficiency. It leverages Unsloth's optimizations to enable training on hardware that would otherwise be insufficient for standard fine-tuning.

**Datasets**:
- **Trendyol**: [https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset]
- **OptikalLLM**: [https://huggingface.co/datasets/OptikalLLM/CyberSecurity-Instruction-Dataset]
- **Cybersec-Reasoning**: [https://huggingface.co/datasets/Llama-Factory/Cybersecurity-Reasoning]

This example is for educational purposes only and may require further adjustments based on your specific use case and environment.

## Requirements

- CUDA-capable GPU with at least 4GB VRAM
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

### 2. Train the Model
```bash
python fine_tune_gpt_oss_cybersecurity.py --config config.yaml --dataset "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
```

### 3. Run Inference
```bash
python fine_tune_gpt_oss_cybersecurity.py --inference ./llama-3.2-3b-cybersecurity-lora --prompt "What are the signs of a phishing attack?"
```

## Dataset Format

The script supports multiple dataset formats, including Hugging Face datasets, JSONL, and JSON files. The dataset should contain instruction-response pairs in one of the following formats:

- `instruction` and `response`
- `question` and `answer`
- `input` and `output`
- `prompt` and `completion`
- `text` (already formatted)

## Configuration Options

Key configuration parameters in `config.yaml`:

- `model_name`: Base model to fine-tune (default: `unsloth/Llama-3.2-3B-bnb-4bit`)
- `batch_size`: Training batch size (reduce if memory issues)
- `gradient_accumulation_steps`: Effective batch size multiplier
- `lora_r`: LoRA rank (higher = more parameters, better quality)
- `learning_rate`: Training learning rate
- `epochs`: Number of training epochs

## Memory Optimization with Unsloth

Unsloth significantly reduces VRAM usage, allowing to fine-tune models on consumer GPUs. For further memory optimization:

1. Reduce `batch_size` to 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_length` to a smaller value (e.g., 1024 or 512)

## Hardware Requirements

- **Minimum**: 4GB GPU memory with Unsloth
- **Recommended**: 8GB+ GPU memory
- **CPU fallback**: Possible but extremely slow

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size and max_length.
2. **Model loading fails**: Check model name and internet connection.
3. **Dataset errors**: Verify dataset format and file paths.
4. **Slow training**: Ensure CUDA is properly installed and detected.

## Files Generated

After training:
- `./llama-3.2-3b-cybersecurity-lora/`: Model files, including the LoRA adapter and tokenizer.
