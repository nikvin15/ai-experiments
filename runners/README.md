# PII Verifier Runners

This directory contains runner scripts for evaluating different PII verification models.

## Overview

The runners implement a **batch PII verification** approach where:
1. CPU-based models detect potential PIIs in input text/JSON
2. LLM-based models filter out false positives by analyzing context
3. Results include verified PIIs, latency metrics, and optional reasoning

---

## Available Runners

### GPU-Based (LLM) Runners

| Runner | Models | Memory (4-bit) | Use Case |
|--------|--------|---------------|----------|
| **run_llama.py** | Llama 3.2 1B/3B | 1.0GB / 2.5GB | High accuracy, Meta's Llama models |
| **run_qwen.py** | Qwen 2.5 0.5B/1.5B/3B | 0.5GB / 1.2GB / 2.3GB | Fast inference, Alibaba's Qwen models |

**Key Features:**
- **4-bit quantization enabled by default** (65-75% memory reduction)
- **Batch verification** - verify multiple PIIs per input
- **Two output modes** - simple (list) or reasoning (detailed)
- **vLLM support** - optional faster inference engine
- **Auto-download** - models download from HuggingFace if not present

### CPU-Based Runners

| Runner | Model | Use Case |
|--------|-------|----------|
| **run_distilbert.py** | DistilBERT (ai4privacy) | General PII detection |
| **run_gliner.py** | GLiNER | Named entity recognition |
| **run_phibert.py** | PHI BERT | Healthcare-specific PII |

---

## Quick Start

### Basic Usage (Default 4-bit Quantization)

```bash
# Llama 3.2 3B (2.5GB VRAM)
python runners/run_llama.py \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_results.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct

# Qwen 2.5 3B (2.3GB VRAM)
python runners/run_qwen.py \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b_results.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct
```

### With Reasoning (Detailed Explanations)

```bash
python runners/run_llama.py \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_reasoning.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --with-reasoning
```

### Disable 4-bit (Use FP16)

```bash
python runners/run_qwen.py \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b_fp16.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --disable-quantization
```

### Use vLLM (Faster Inference)

```bash
python runners/run_llama.py \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_vllm.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --use-vllm
```

---

## Input Format

The runners expect JSONL files with the **simplified 3-key format**:

```json
{
  "recordId": "de_0001_01",
  "input": "Patient SSN 123-45-6789 requires authorization",
  "PIIs": ["Social Security Number"]
}
```

**Fields:**
- `recordId` - Unique identifier
- `input` - Text or JSON string to analyze
- `PIIs` - Array of detected PII names (from CPU models)

---

## Output Format

### Simple Mode (Default)

```json
{
  "recordId": "de_0001_01",
  "input": "Patient SSN 123-45-6789 requires authorization",
  "detected_piis": ["Social Security Number"],
  "verified_piis": ["Social Security Number"],
  "latency_ms": 45.2,
  "model": "llama_3.2_3b_4bit"
}
```

### Reasoning Mode (--with-reasoning)

```json
{
  "recordId": "de_0001_01",
  "input": "Patient SSN 123-45-6789 requires authorization",
  "detected_piis": ["Social Security Number"],
  "verified_piis": [
    {
      "pii": "Social Security Number",
      "verified": true,
      "reason": "Valid SSN pattern used in medical context for patient identification"
    }
  ],
  "latency_ms": 52.8,
  "model": "llama_3.2_3b_4bit_reasoning"
}
```

---

## Command-Line Arguments

### Common Arguments (All Runners)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | - | Input JSONL file path |
| `--output` | Yes | - | Output JSONL file path |
| `--model-path` | Yes | - | Path or HuggingFace model name |
| `--device` | No | `cuda` | Device: `cuda` or `cpu` |

### LLM-Specific Arguments (Llama/Qwen)

| Argument | Default | Description |
|----------|---------|-------------|
| `--disable-quantization` | False (4-bit ON) | Disable 4-bit quantization, use FP16 |
| `--use-vllm` | False | Use vLLM for faster inference (forces FP16) |
| `--with-reasoning` | False | Include detailed reasoning in output |
| `--batch-size` | 4 | Batch size (not applicable for batch verification) |

---

## Model Selection Guide

### By GPU Memory

#### 4GB VRAM (GTX 1650, RTX 2060)
```bash
# Qwen 0.5B (fastest, 0.5GB)
python runners/run_qwen.py \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/qwen_0.5b.jsonl

# Llama 1B (balanced, 1.0GB)
python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/llama_1b.jsonl
```

#### 8GB VRAM (RTX 3060, RTX 4060)
```bash
# Qwen 3B (best accuracy, 2.3GB)
python runners/run_qwen.py \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b.jsonl

# Llama 3B (Meta flagship, 2.5GB)
python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b.jsonl

# Or use FP16 for maximum speed (6.8GB)
python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --disable-quantization \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_fp16.jsonl
```

#### 12GB+ VRAM (RTX 3080, RTX 4070+)
```bash
# Use vLLM for maximum throughput
python runners/run_qwen.py \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --use-vllm \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b_vllm.jsonl
```

### By Use Case

#### Fast Screening (Low Memory)
- **Model:** Qwen 2.5 0.5B
- **Memory:** 0.5GB
- **Speed:** ⭐⭐⭐⭐⭐
- **Accuracy:** ⭐⭐⭐

```bash
python runners/run_qwen.py \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/qwen_0.5b.jsonl
```

#### Balanced Performance
- **Model:** Llama 3.2 1B
- **Memory:** 1.0GB
- **Speed:** ⭐⭐⭐⭐
- **Accuracy:** ⭐⭐⭐⭐

```bash
python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/llama_1b.jsonl
```

#### High Accuracy (Production)
- **Model:** Llama 3.2 3B or Qwen 2.5 3B
- **Memory:** 2.3-2.5GB
- **Speed:** ⭐⭐⭐
- **Accuracy:** ⭐⭐⭐⭐⭐

```bash
python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --with-reasoning \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_reasoning.jsonl
```

---

## Memory Requirements

See [MEMORY_GUIDE.md](../MEMORY_GUIDE.md) for detailed memory calculations.

**Quick Reference:**

| Model | FP16 | 4-bit (Default) | Memory Saved |
|-------|------|-----------------|--------------|
| Qwen 2.5 0.5B | 1.5GB | 0.5GB | 67% |
| Llama 3.2 1B | 2.5GB | 1.0GB | 60% |
| Qwen 2.5 1.5B | 3.5GB | 1.2GB | 66% |
| Qwen 2.5 3B | 6.5GB | 2.3GB | 65% |
| Llama 3.2 3B | 6.8GB | 2.5GB | 63% |

---

## Examples

### Example 1: Basic Verification (Qwen 3B)

```bash
python runners/run_qwen.py \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b_results.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct
```

**Output:**
```
2025-11-18 10:30:15 - INFO - Loading Qwen 2.5 model from Qwen/Qwen2.5-3B-Instruct
2025-11-18 10:30:15 - INFO - Device: cuda, 4-bit: True, vLLM: False, Reasoning: False
2025-11-18 10:30:18 - INFO - Qwen 2.5 model loaded successfully
2025-11-18 10:30:18 - INFO - Processing 484 records
2025-11-18 10:30:18 - INFO - Using batch verification: each input analyzed for multiple PIIs
...
============================================================
Performance Summary - qwen_2.5_3b_4bit
============================================================
Total Processed: 484
Verified: 462 (95.5%)
Rejected: 22
Errors: 0

Latency:
  Mean: 45.2ms
  P50:  42.1ms
  P95:  68.3ms
  P99:  89.5ms
...
```

### Example 2: With Reasoning (Llama 3B)

```bash
python runners/run_llama.py \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_reasoning.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --with-reasoning
```

**Sample Output Record:**
```json
{
  "recordId": "de_0042_01",
  "input": "Background check flagged criminal history for applicant John Doe",
  "detected_piis": ["Criminal Records History", "First Name", "Last Name"],
  "verified_piis": [
    {
      "pii": "Criminal Records History",
      "verified": true,
      "reason": "Background check context clearly indicates criminal records are being referenced"
    },
    {
      "pii": "First Name",
      "verified": true,
      "reason": "John is a personal name in applicant context"
    },
    {
      "pii": "Last Name",
      "verified": true,
      "reason": "Doe is a surname following first name"
    }
  ],
  "latency_ms": 58.4,
  "model": "llama_3.2_3b_4bit_reasoning"
}
```

### Example 3: FP16 for Maximum Speed (8GB+ GPU)

```bash
python runners/run_llama.py \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_fp16.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --disable-quantization
```

**Performance Impact:**
- Memory: 6.8GB (vs 2.5GB with 4-bit)
- Speed: ~30% faster inference
- Accuracy: Identical to 4-bit

### Example 4: vLLM for Batch Processing (12GB+ GPU)

```bash
python runners/run_qwen.py \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b_vllm.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --use-vllm
```

**Performance Impact:**
- Memory: 7.5GB (vLLM requires FP16)
- Speed: 2-4x faster than Transformers
- Best for: Large-scale batch processing

---

## Troubleshooting

### Out of Memory (OOM)

**Error:** `CUDA out of memory`

**Solutions:**
1. Use smaller model (0.5B or 1B instead of 3B)
2. Ensure 4-bit quantization is enabled (default)
3. Check GPU memory:
   ```bash
   nvidia-smi
   ```
4. Close other GPU processes

### Slow Inference

**Issue:** Inference slower than expected

**Solutions:**
1. Verify GPU is being used (not CPU):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Try vLLM if you have enough VRAM:
   ```bash
   pip install vllm
   python runners/run_qwen.py --use-vllm ...
   ```
3. Use FP16 instead of 4-bit (trades memory for speed):
   ```bash
   python runners/run_llama.py --disable-quantization ...
   ```

### HuggingFace Authentication

**Error:** `Repository not found` or `Authentication required`

**Solution for Llama models:**
```bash
# Set HuggingFace token
export HF_TOKEN=your_huggingface_token

# Accept Llama license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

### Model Download Issues

**Issue:** Model download fails or takes too long

**Solution:**
1. Check internet connection
2. Download manually:
   ```bash
   huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
   ```
3. Use local path:
   ```bash
   python runners/run_llama.py --model-path /path/to/local/model ...
   ```

---

## Performance Comparison

Run all models and compare results:

```bash
# Run all models
python runners/run_qwen.py \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/qwen_0.5b.jsonl

python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/llama_1b.jsonl

python runners/run_qwen.py \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b.jsonl

python runners/run_llama.py \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b.jsonl

# Compare all results
python analyze_results.py --results-dir results/
```

---

## Notes

- **4-bit quantization is enabled by default** for optimal memory usage
- All runners auto-download models from HuggingFace if not present locally
- Batch verification verifies ALL PIIs in a single input at once (not single-entity verification)
- vLLM requires FP16 (no quantization support)
- Reasoning mode adds 10-20% latency but provides explainability
- See [MEMORY_GUIDE.md](../MEMORY_GUIDE.md) for detailed memory calculations

---

**Last Updated:** 2025-11-18
**Compatible With:** PyTorch 2.0+, Transformers 4.35+, CUDA 11.8+
