# Memory Requirements Guide

Guide for GPU memory requirements when running Llama and Qwen models for PII verification.

---

## Quick Reference Table

| Model | Parameters | FP16 VRAM | 4-bit VRAM | Memory Saved | Recommended GPU |
|-------|-----------|-----------|------------|--------------|-----------------|
| **Qwen 2.5 0.5B** | 0.5B | ~1.5 GB | ~0.5 GB | 67% | GTX 1650+ (4GB) |
| **Llama 3.2 1B** | 1B | ~2.5 GB | ~1.0 GB | 60% | GTX 1650+ (4GB) |
| **Qwen 2.5 1.5B** | 1.5B | ~3.5 GB | ~1.2 GB | 66% | GTX 1660+ (6GB) |
| **Qwen 2.5 3B** | 3B | ~6.5 GB | ~2.3 GB | 65% | RTX 3060 (8GB+) |
| **Llama 3.2 3B** | 3B | ~6.8 GB | ~2.5 GB | 63% | RTX 3060 (8GB+) |

---

## Memory Calculation Formula

### FP16 (Half Precision)
```
Memory (GB) = (Parameters × 2 bytes) + Overhead
```

- **2 bytes** per parameter (FP16 = 16 bits = 2 bytes)
- **Overhead:** ~500-800 MB for:
  - KV cache
  - Activation memory
  - CUDA context
  - Model metadata

### 4-bit Quantization (NF4 via BitsAndBytes)
```
Memory (GB) = (Parameters × 0.5 bytes) + Overhead
```

- **0.5 bytes** per parameter (4 bits = 0.5 bytes)
- **Overhead:** ~300-500 MB for:
  - Quantization constants
  - KV cache (still in FP16)
  - CUDA context

---

## Detailed Breakdown by Model

### Qwen 2.5 0.5B Instruct

| Configuration | Calculation | VRAM Required |
|--------------|-------------|---------------|
| **FP16** | (0.5B × 2) + 0.5GB = 1.5GB | ~1.5 GB |
| **4-bit** | (0.5B × 0.5) + 0.3GB = 0.55GB | ~0.5 GB |

**Use case:** Fastest inference, lowest memory, good for basic PII verification

---

### Llama 3.2 1B Instruct

| Configuration | Calculation | VRAM Required |
|--------------|-------------|---------------|
| **FP16** | (1B × 2) + 0.5GB = 2.5GB | ~2.5 GB |
| **4-bit** | (1B × 0.5) + 0.3GB = 0.8GB | ~1.0 GB |

**Use case:** Good balance of speed and accuracy for PII tasks

---

### Qwen 2.5 1.5B Instruct

| Configuration | Calculation | VRAM Required |
|--------------|-------------|---------------|
| **FP16** | (1.5B × 2) + 0.5GB = 3.5GB | ~3.5 GB |
| **4-bit** | (1.5B × 0.5) + 0.3GB = 1.05GB | ~1.2 GB |

**Use case:** Better reasoning capability than 0.5B/1B models

---

### Qwen 2.5 3B Instruct

| Configuration | Calculation | VRAM Required |
|--------------|-------------|---------------|
| **FP16** | (3B × 2) + 0.5GB = 6.5GB | ~6.5 GB |
| **4-bit** | (3B × 0.5) + 0.3GB = 1.8GB | ~2.3 GB |

**Use case:** High accuracy, complex PII context understanding

---

### Llama 3.2 3B Instruct

| Configuration | Calculation | VRAM Required |
|--------------|-------------|---------------|
| **FP16** | (3B × 2) + 0.8GB = 6.8GB | ~6.8 GB |
| **4-bit** | (3B × 0.5) + 0.5GB = 2.0GB | ~2.5 GB |

**Use case:** Best accuracy for complex edge cases and false positive filtering

---

## Batch Size Impact

Batch size affects memory through activation caching:

| Batch Size | Additional Memory (4-bit) | Additional Memory (FP16) |
|-----------|---------------------------|-------------------------|
| 1 | +0 MB | +0 MB |
| 2 | +50-100 MB | +100-200 MB |
| 4 | +100-200 MB | +200-400 MB |
| 8 | +200-400 MB | +400-800 MB |

**Recommendations:**
- **4-bit mode:** Batch size 4-8 is safe on 8GB GPUs
- **FP16 mode:** Batch size 2-4 on 8GB GPUs
- **CPU mode:** Batch size 1 (very slow, not recommended)

---

## System Requirements

### Minimum Requirements

**For 4-bit quantized models (default):**
- **GPU:** NVIDIA GPU with 4GB+ VRAM (GTX 1650, RTX 2060, etc.)
- **CUDA:** 11.8 or higher
- **RAM:** 8GB+ system RAM
- **Storage:** 5-15GB for model downloads

**For FP16 models:**
- **GPU:** NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 4060, etc.)
- **CUDA:** 11.8 or higher
- **RAM:** 16GB+ system RAM
- **Storage:** 5-15GB for model downloads

### Recommended Setup

**For production workloads:**
- **GPU:** RTX 3060 (12GB), RTX 3080 (10GB), RTX 4070 (12GB), or better
- **CUDA:** 12.1 or higher
- **RAM:** 32GB+ system RAM
- **Storage:** NVMe SSD with 20GB+ free space

---

## Quantization Methods Compared

### 4-bit NF4 (BitsAndBytes) - **DEFAULT**
- **Memory:** 65-75% reduction vs FP16
- **Accuracy:** ~98-99% of FP16 quality
- **Speed:** Slightly slower than FP16 (1.1-1.3x)
- **Use case:** Best for most scenarios - great balance

### FP16 (Half Precision)
- **Memory:** Baseline (100%)
- **Accuracy:** Baseline (100%)
- **Speed:** Fastest inference
- **Use case:** When you have enough VRAM and need maximum speed

### 8-bit Quantization
- **Memory:** 50% reduction vs FP16
- **Accuracy:** ~99-100% of FP16 quality
- **Speed:** Similar to FP16
- **Status:** Not currently implemented (can be added)

---

## vLLM Memory Considerations

vLLM is an optional faster inference engine but has different memory characteristics:

| Model | vLLM Memory (FP16) | Transformers + 4-bit | Winner |
|-------|-------------------|---------------------|---------|
| Qwen 0.5B | ~2.0 GB | ~0.5 GB | 4-bit |
| Llama 1B | ~3.0 GB | ~1.0 GB | 4-bit |
| Qwen 1.5B | ~4.0 GB | ~1.2 GB | 4-bit |
| Qwen 3B | ~7.5 GB | ~2.3 GB | 4-bit |
| Llama 3B | ~8.0 GB | ~2.5 GB | 4-bit |

**Key points:**
- vLLM only supports FP16 (no quantization)
- vLLM is 2-4x faster for batch inference
- Default 4-bit mode is better for memory-constrained environments
- Use `--use-vllm` flag if you have enough VRAM and need speed

---

## Example Commands by GPU

### GTX 1650 / RTX 2060 (4-6GB VRAM)
```bash
# Qwen 0.5B - fastest option
python runners/run_qwen.py \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 4 \
  --input data/training_all_elements.jsonl \
  --output results/qwen_0.5b_results.jsonl

# Llama 1B - good balance
python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --batch-size 4 \
  --input data/training_all_elements.jsonl \
  --output results/llama_1b_results.jsonl
```

### RTX 3060 / RTX 4060 (8-12GB VRAM)
```bash
# Qwen 3B - best accuracy with 4-bit
python runners/run_qwen.py \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --batch-size 8 \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b_results.jsonl

# Llama 3B - best accuracy with 4-bit
python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --batch-size 6 \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_results.jsonl

# Or use FP16 if you want maximum speed
python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --batch-size 4 \
  --disable-quantization \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_fp16_results.jsonl
```

### RTX 3080+ / RTX 4070+ (10-16GB VRAM)
```bash
# Use vLLM for maximum throughput
python runners/run_qwen.py \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --batch-size 16 \
  --use-vllm \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b_vllm_results.jsonl
```

---

## Performance vs Memory Tradeoffs

| Configuration | Memory | Speed | Accuracy | Best For |
|--------------|--------|-------|----------|----------|
| **Qwen 0.5B 4-bit** | 0.5GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Quick scans, low memory |
| **Llama 1B 4-bit** | 1.0GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced performance |
| **Qwen 3B 4-bit** | 2.3GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High accuracy, memory efficient |
| **Llama 3B 4-bit** | 2.5GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Best accuracy |
| **Llama 3B FP16** | 6.8GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High-end GPUs, max speed |
| **Qwen 3B vLLM** | 7.5GB | ⭐⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Production, batch processing |

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size: `--batch-size 2` or `--batch-size 1`
2. Use smaller model: Try 0.5B or 1B instead of 3B
3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
4. Check GPU memory before starting:
   ```bash
   nvidia-smi
   ```

### Slow Inference

**If inference is slower than expected:**
1. Ensure GPU is being used (not CPU)
2. Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try reducing batch size (counterintuitive but can help)
4. Consider vLLM for batch processing

---

## Monitoring GPU Memory

### Real-time Monitoring
```bash
# Watch GPU memory usage
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

### In Python
```python
import torch

# Check available memory
free_mem = torch.cuda.mem_get_info()[0] / 1024**3  # GB
total_mem = torch.cuda.mem_get_info()[1] / 1024**3  # GB
print(f"Free: {free_mem:.2f}GB / Total: {total_mem:.2f}GB")
```

---

## Default Configuration (4-bit Quantization)

As of this version, **4-bit NF4 quantization is enabled by default** for all LLM runners.

This provides:
- ✅ 65-75% memory reduction
- ✅ Fits 3B models on 8GB GPUs
- ✅ Near-identical accuracy to FP16
- ✅ Reasonable inference speed

To use FP16 instead, add the `--disable-quantization` flag.

---

**Last Updated:** 2025-11-18
**Tested With:** CUDA 12.1, PyTorch 2.0+, Transformers 4.35+
