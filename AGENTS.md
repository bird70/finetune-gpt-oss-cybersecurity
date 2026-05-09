# FINE-TUNING KNOWLEDGE BASE

**Scope**: `finetune-gpt-oss-cybersecurity/` - Llama 3.2 3B cybersecurity micro-fine-tuning

---

## OVERVIEW

Low-resource LLM fine-tuning using **Unsloth** (4-bit LoRA) targeting consumer GPUs (4GB VRAM minimum). Trains on cybersecurity instruction datasets (Trendyol, OptikalLLM, Cybersec-Reasoning).

---

## ENTRY POINTS

| File | Role | Command |
|------|------|---------|
| `fine_tune_gpt_oss_cybersecurity.py` | Main script: config generation, training, inference | `python fine_tune_gpt_oss_cybersecurity.py --help` |
| `llama_finetuning.py` | Training orchestration via Unsloth | (Called by main script) |
| `*.ipynb` | Interactive notebooks for experimentation | Open in Jupyter |

---

## WHERE TO LOOK

| Task | Location |
|------|----------|
| **Training configuration** | `config.yaml` |
| **Model selection** | `fine_tune_gpt_oss_cybersecurity.py` lines ~60-80 (model_name) |
| **Dataset handling** | `fine_tune_gpt_oss_cybersecurity.py` (supports HF datasets, JSONL, JSON) |
| **Memory optimization** | `llama_finetuning.py` (batch_size, gradient_accumulation, max_length) |
| **Output models** | `./llama-3.2-3b-cybersecurity-lora/` (after training) |

---

## CONVENTIONS

### Configuration (YAML-Based)

```yaml
model_name: unsloth/Llama-3.2-3B-bnb-4bit  # 4-bit quantized base
batch_size: 2                               # Reduce to 1 if OOM
gradient_accumulation_steps: 4              # Maintains effective batch
max_length: 1024                            # 512-1024 for consumer GPU
lora_r: 16                                  # LoRA rank (quality vs speed)
learning_rate: 2e-4
epochs: 3
```

### Dataset Format

Accepts instruction-response pairs with flexible naming:
- `instruction` + `response`
- `question` + `answer`
- `input` + `output`
- `prompt` + `completion`
- `text` (pre-formatted)

---

## ANTI-PATTERNS (THIS SUBDOMAIN)

1. ❌ `load_best_model_at_end=True` → causes "strategy mismatch" with LoRA
2. ❌ `batch_size > 2` on 4GB GPU → guaranteed OOM
3. ❌ `max_length > 2048` on consumer GPU → memory intensive
4. ❌ Omit `use_safetensors=True` → security risk when loading remote models
5. ❌ CPU fallback mode → "extremely slow", use GPU only

---

## MEMORY TROUBLESHOOTING

| Issue | Fix |
|-------|-----|
| CUDA OOM | Reduce `batch_size` to 1, cut `max_length` by 50% |
| Slow loading | Ensure CUDA detected: `python -c "import torch; print(torch.cuda.is_available())"` |
| Dataset errors | Verify column names match supported format (instruction/question/input/prompt) |

---

## COMMANDS

```bash
# Generate config template
python fine_tune_gpt_oss_cybersecurity.py --create-sample-config

# Train (requires GPU + CUDA)
python fine_tune_gpt_oss_cybersecurity.py --config config.yaml \
  --dataset "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"

# Inference with trained adapter
python fine_tune_gpt_oss_cybersecurity.py --inference ./llama-3.2-3b-cybersecurity-lora \
  --prompt "Explain SQL injection attacks"
```

---

## FILES GENERATED

After successful training:
- `llama-3.2-3b-cybersecurity-lora/adapter_config.json`
- `llama-3.2-3b-cybersecurity-lora/adapter_model.bin`
- `llama-3.2-3b-cybersecurity-lora/tokenizer.model`
- `llama-3.2-3b-cybersecurity-lora/training_args.bin`

---

## KEY DEPENDENCIES

- **torch**: PyTorch 2.0+ (CUDA 11.8+)
- **transformers**: 4.30+
- **peft**: LoRA adapter library
- **unsloth**: Unsloth + xformers for memory optimization
- **datasets**: HuggingFace dataset loading
- **accelerate**: Distributed training orchestration

See `requirements.txt` for pinned versions.

---

## RESOURCES

- **Unsloth**: https://github.com/unslothai/unsloth (memory-efficient training)
- **Llama 3.2 3B**: https://huggingface.co/meta-llama/Llama-3.2-3B
- **PEFT/LoRA**: https://github.com/huggingface/peft
- **Cybersecurity Datasets**:
  - Trendyol: https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset
  - OptikalLLM: https://huggingface.co/datasets/OptikalLLM/CyberSecurity-Instruction-Dataset
