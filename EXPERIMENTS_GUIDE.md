# PII Verifier Model Experiments - Complete Guide

**Created:** 2025-11-18
**Purpose:** Test and compare CPU and GPU models for PII verification

---

## What Was Created

### 📁 Directory Structure

```
ai-experiments/
├── README.md                        # Main documentation
├── EXPERIMENTS_GUIDE.md             # This file
├── requirements.txt                 # Python dependencies
├── quickstart.sh                    # Setup script
├── generate_test_data.py            # Test data generator
├── analyze_results.py               # Results analyzer
├── runners/                         # Model runners
│   ├── common.py                    # Shared utilities
│   ├── run_distilbert.py            # DistilBERT runner (CPU)
│   ├── run_gliner.py                # GLiNER runner (CPU)
│   ├── run_phibert.py               # PHI BERT runner (CPU)
│   ├── run_llama.py                 # Llama 3.2 runner (GPU)
│   └── run_qwen.py                  # Qwen 2.5 runner (GPU)
├── data/                            # Input JSONL files
│   └── sample_test.jsonl            # 10 sample test cases
├── results/                         # Output JSONL files (created by runners)
└── models/                          # Downloaded model weights (you create)
```

### 🎯 Models Supported

**CPU Models (Fast, Cheap):**
1. **DistilBERT** - `ai4privacy/distilbert_finetuned_ai4privacy_v2` (66M params)
   - General PII verification
   - Target: 9.5ms p95 latency
   - Best for: EMAIL, PHONE, PERSON, SSN

2. **GLiNER** - `urchade/gliner_base` (400M params)
   - Financial entity verification
   - Target: 75ms p95 latency
   - Best for: CRYPTO, IBAN, SWIFT, ROUTING_NUMBER

3. **PHI BERT** - `obi/deid_bert_i2b2` (110M params)
   - Medical PHI verification
   - Target: 25ms p95 latency
   - Best for: MRN, PATIENT, DIAGNOSIS, MEDICATION

**GPU Models (Slower, More Accurate):**
4. **Llama 3.2** - `meta-llama/Llama-3.2-1B-Instruct` or `Llama-3.2-3B-Instruct`
   - General PII with reasoning
   - Target: 50-120ms p95 latency
   - Requires: 2-3GB VRAM (4-bit quantization)

5. **Qwen 2.5** - `Qwen/Qwen2.5-0.5B-Instruct`, `1.5B-Instruct`, or `3B-Instruct`
   - General PII with reasoning
   - Target: 40-100ms p95 latency
   - Requires: 1.5-2GB VRAM (4-bit quantization)

---

## Quick Start (3 Minutes)

### Step 1: Setup Virtual Environment

```bash
cd ai-experiments

# Create virtual environment
python3 -m venv env

# Activate virtual environment
source env/bin/activate  # Linux/Mac
# OR
env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU models with CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Run setup script
bash quickstart.sh

# This will:
# - Create directories
# - Generate sample test data (200 test cases)
```

### Step 2: Run First Experiment

Models are **automatically downloaded** on first run!

```bash
# Run DistilBERT on sample data
# Model will be downloaded automatically on first run
python3 runners/run_distilbert.py \
  --input data/sample_test.jsonl \
  --output results/distilbert_sample.jsonl \
  --model-path models/distilbert_ai4privacy \
  --batch-size 32
```

**Note:** First run takes a few minutes to download the model. Subsequent runs are instant.

### Step 3: View Results

**Expected Output:**
```
Loading DistilBERT model from models/distilbert_ai4privacy
Device: cpu, ONNX: False
DistilBERT model loaded successfully
Processing 10 records
Batch size: 32

============================================================
Performance Summary - distilbert_ai4privacy_v2
============================================================
Total Processed: 10
Verified: 7 (70.0%)
Rejected: 3
Errors: 0

Latency:
  Mean: 15.23ms
  P50:  14.87ms
  P95:  18.42ms
  P99:  18.42ms

Confidence:
  Mean: 0.842
  Median: 0.879

Throughput:
  Total Time: 0.18s
  Items/sec: 55.56
============================================================

Results saved to: results/distilbert_sample.jsonl
```

```bash
# View first result
head -1 results/distilbert_sample.jsonl | python3 -m json.tool
```

**Example Output:**
```json
{
  "recordId": "sample_001",
  "input": "Contact john.doe@example.com for support",
  "entityType": "EMAIL",
  "entityValue": "john.doe@example.com",
  "metadata": {
    "source": "sample",
    "is_true_positive": true
  },
  "result": {
    "verified": true,
    "confidence": 0.9234,
    "reason": "High confidence PII detection (EMAIL)",
    "latencyMs": 12.45
  },
  "model": {
    "name": "distilbert_ai4privacy_v2",
    "type": "cpu",
    "parameters": "66M"
  },
  "timestamp": "2025-11-18T10:30:45.123Z"
}
```

---

## Running All CPU Models

### 1. DistilBERT (General PII)

```bash
# Generate email test data
python3 generate_test_data.py --output data/test_emails.jsonl --count 100 --type email

# Run DistilBERT
python3 runners/run_distilbert.py \
  --input data/test_emails.jsonl \
  --output results/distilbert_emails.jsonl \
  --model-path models/distilbert_ai4privacy \
  --batch-size 32

# With ONNX optimization (3-10x faster)
python3 runners/run_distilbert.py \
  --input data/test_emails.jsonl \
  --output results/distilbert_emails_onnx.jsonl \
  --model-path models/distilbert_ai4privacy \
  --use-onnx \
  --batch-size 32
```

### 2. GLiNER (Financial Entities)

**First, download GLiNER:**
```bash
pip3 install gliner

python3 -c "from gliner import GLiNER; \
  model = GLiNER.from_pretrained('urchade/gliner_base'); \
  model.save_pretrained('./models/gliner_base')"
```

**Run experiments:**
```bash
# Generate financial test data
python3 generate_test_data.py --output data/test_financial.jsonl --count 100 --type financial

# Run GLiNER
python3 runners/run_gliner.py \
  --input data/test_financial.jsonl \
  --output results/gliner_financial.jsonl \
  --model-path models/gliner_base \
  --batch-size 16
```

### 3. PHI BERT (Medical Entities)

**First, download PHI BERT:**
```bash
python3 -c "from transformers import AutoModelForTokenClassification, AutoTokenizer; \
  AutoModelForTokenClassification.from_pretrained('obi/deid_bert_i2b2').save_pretrained('./models/phi_bert'); \
  AutoTokenizer.from_pretrained('obi/deid_bert_i2b2').save_pretrained('./models/phi_bert')"
```

**Run experiments:**
```bash
# Generate medical test data
python3 generate_test_data.py --output data/test_medical.jsonl --count 100 --type medical

# Run PHI BERT
python3 runners/run_phibert.py \
  --input data/test_medical.jsonl \
  --output results/phibert_medical.jsonl \
  --model-path models/phi_bert \
  --batch-size 32
```

---

## Running GPU Models (Advanced)

### Prerequisites

**Check GPU:**
```bash
# Check if CUDA is available
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Install GPU dependencies:**
```bash
# For CUDA 11.8
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# For 4-bit quantization
pip3 install bitsandbytes

# For fast inference (optional)
pip3 install vllm
```

### Llama 3.2 3B

**Download model:**
```bash
# Requires HuggingFace account and acceptance of Llama license
export HF_TOKEN=your_token_here

python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct').save_pretrained('./models/llama_3.2_3b'); \
  AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct').save_pretrained('./models/llama_3.2_3b')"
```

**Run experiments:**
```bash
# Basic (no quantization, ~6GB VRAM)
python3 runners/run_llama.py \
  --input data/test_emails.jsonl \
  --output results/llama_3b_emails.jsonl \
  --model-path models/llama_3.2_3b \
  --batch-size 2

# With 4-bit quantization (2-3GB VRAM)
python3 runners/run_llama.py \
  --input data/test_emails.jsonl \
  --output results/llama_3b_4bit_emails.jsonl \
  --model-path models/llama_3.2_3b \
  --batch-size 4 \
  --use-4bit

# With vLLM (fastest, if installed)
python3 runners/run_llama.py \
  --input data/test_emails.jsonl \
  --output results/llama_3b_vllm_emails.jsonl \
  --model-path models/llama_3.2_3b \
  --batch-size 8 \
  --use-vllm
```

### Qwen 2.5 3B

**Download model:**
```bash
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', trust_remote_code=True).save_pretrained('./models/qwen_2.5_3b'); \
  AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', trust_remote_code=True).save_pretrained('./models/qwen_2.5_3b')"
```

**Run experiments:**
```bash
# With 4-bit quantization (1.5-2GB VRAM)
python3 runners/run_qwen.py \
  --input data/test_emails.jsonl \
  --output results/qwen_3b_4bit_emails.jsonl \
  --model-path models/qwen_2.5_3b \
  --batch-size 4 \
  --use-4bit

# With vLLM (fastest)
python3 runners/run_qwen.py \
  --input data/test_emails.jsonl \
  --output results/qwen_3b_vllm_emails.jsonl \
  --model-path models/qwen_2.5_3b \
  --batch-size 8 \
  --use-vllm
```

---

## Analyzing Results

### Compare All Models

```bash
# After running multiple experiments
python3 analyze_results.py --results-dir results/
```

**Example Output:**
```
================================================================================
Model Performance Comparison
================================================================================
Model                     Type   Params   P95 Latency  Verified   Avg Conf
--------------------------------------------------------------------------------
distilbert_ai4privacy_v2  cpu    66M          12.45ms      85.0%      0.891
phibert_i2b2              cpu    110M         28.73ms      78.0%      0.867
gliner_base               cpu    400M         78.92ms      92.0%      0.823
qwen_2.5_3b_4bit_vllm     gpu    3B           95.23ms      88.0%      0.845
llama_3.2_3b_4bit         gpu    3B          145.67ms      90.0%      0.872
================================================================================

Detailed Statistics:

distilbert_ai4privacy_v2:
  Total: 100
  Verified: 85 (85.0%)
  Rejected: 15
  Latency:
    Mean: 10.23ms
    P50:  9.87ms
    P95:  12.45ms
    P99:  14.32ms
  Confidence:
    Mean:   0.891
    Median: 0.902

...

================================================================================
Best Performers:
================================================================================
Fastest (P95): distilbert_ai4privacy_v2 - 12.45ms
Highest Confidence: distilbert_ai4privacy_v2 - 0.891
Most Selective: phibert_i2b2 - 78.0% verified
```

### Match Input/Output by recordId

```bash
# View specific record's result
grep '"recordId": "sample_001"' results/distilbert_sample.jsonl | python3 -m json.tool
```

---

## Understanding Results

### Input Format

```json
{
  "recordId": "unique-id",              // Unique identifier
  "input": "context text",              // Text containing entity
  "entityType": "EMAIL",                // Type of entity
  "entityValue": "john@example.com",    // Extracted value
  "metadata": {
    "source": "presidio",               // Where it was detected
    "confidence": 0.95                  // Original detection confidence
  }
}
```

### Output Format

```json
{
  "recordId": "unique-id",              // SAME as input (for matching)
  "input": "context text",              // Preserved from input
  "entityType": "EMAIL",                // Preserved from input
  "entityValue": "john@example.com",    // Preserved from input
  "metadata": {...},                    // Preserved from input
  "result": {
    "verified": true,                   // ✅ TRUE = confirmed PII, ❌ FALSE = false positive
    "confidence": 0.92,                 // Model's confidence (0-1)
    "reason": "High confidence PII",    // Human-readable explanation
    "latencyMs": 12.5                   // Verification time
  },
  "model": {
    "name": "distilbert_ai4privacy_v2", // Model used
    "type": "cpu",                      // cpu or gpu
    "parameters": "66M"                 // Model size
  },
  "timestamp": "2025-11-18T10:30:45.123Z"
}
```

### Matching Input to Output

```python
import json

# Load input
with open('data/test_emails.jsonl') as f:
    inputs = {json.loads(line)['recordId']: json.loads(line) for line in f}

# Load output
with open('results/distilbert_emails.jsonl') as f:
    outputs = {json.loads(line)['recordId']: json.loads(line) for line in f}

# Match by recordId
for record_id in inputs:
    input_record = inputs[record_id]
    output_record = outputs[record_id]

    print(f"RecordID: {record_id}")
    print(f"  Input Entity: {input_record['entityValue']}")
    print(f"  Verified: {output_record['result']['verified']}")
    print(f"  Confidence: {output_record['result']['confidence']:.3f}")
    print(f"  Latency: {output_record['result']['latencyMs']:.2f}ms")
    print()
```

---

## Performance Targets

Based on PRESIDIO_VERIFIER_ANALYSIS.md recommendations:

| Model | Target P95 | Expected F1 | Cost/Month | Use Case |
|-------|------------|-------------|------------|----------|
| **DistilBERT** | <10ms | 0.97 | $80 | General PII (EMAIL, PHONE, PERSON) |
| **GLiNER** | <80ms | 0.98 | $85 | Financial (CRYPTO, IBAN, SWIFT) |
| **PHI BERT** | <30ms | 0.94 | $80 | Medical (MRN, PATIENT, DIAGNOSIS) |
| **Llama 3.2 3B** | <120ms | 0.89 | $95 | Complex cases with reasoning |
| **Qwen 2.5 3B** | <100ms | 0.87 | $95 | Complex cases (most efficient) |

**Recommended Approach:**
- **Phase 1:** CPU-only (DistilBERT + GLiNER + PHI BERT) = $245/month
- **Phase 2:** Hybrid (90% CPU, 10% GPU) = $435/month

---

## Troubleshooting

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# 1. Use 4-bit quantization
--use-4bit

# 2. Reduce batch size
--batch-size 1

# 3. Use smaller model
# Instead of Llama 3.2 3B, use 1B:
--model-path models/llama_3.2_1b

# 4. Use CPU (much slower)
--device cpu
```

### Slow CPU Inference

**Problem:** CPU models are slow

**Solutions:**
```bash
# 1. Use ONNX optimization (3-10x speedup)
--use-onnx

# 2. Increase batch size
--batch-size 64

# 3. Convert to ONNX INT8 quantization
python3 -m optimum.onnxruntime.export onnx \
  --model models/distilbert_ai4privacy \
  --task sequence-classification \
  --quantize int8 \
  models/distilbert_ai4privacy_onnx
```

### Model Download Fails

**Problem:** Can't download models from HuggingFace

**Solutions:**
```bash
# 1. Set token (for gated models like Llama)
export HF_TOKEN=your_token_here

# 2. Use mirror (in restricted regions)
export HF_ENDPOINT=https://hf-mirror.com

# 3. Download manually and place in models/
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'gliner'`

**Solution:**
```bash
# Install missing dependencies
pip3 install gliner

# Or reinstall all dependencies
pip3 install -r requirements.txt
```

---

## Next Steps

### 1. Run Full Experiments

```bash
# Generate large test datasets
python3 generate_test_data.py --output data/test_all.jsonl --count 1000 --type all

# Run all CPU models
python3 runners/run_distilbert.py --input data/test_all.jsonl --output results/distilbert_1000.jsonl --model-path models/distilbert_ai4privacy --batch-size 32
python3 runners/run_gliner.py --input data/test_all.jsonl --output results/gliner_1000.jsonl --model-path models/gliner_base --batch-size 16
python3 runners/run_phibert.py --input data/test_all.jsonl --output results/phibert_1000.jsonl --model-path models/phi_bert --batch-size 32

# Run GPU models (if available)
python3 runners/run_llama.py --input data/test_all.jsonl --output results/llama_1000.jsonl --model-path models/llama_3.2_3b --batch-size 4 --use-4bit
python3 runners/run_qwen.py --input data/test_all.jsonl --output results/qwen_1000.jsonl --model-path models/qwen_2.5_3b --batch-size 4 --use-4bit

# Compare all results
python3 analyze_results.py --results-dir results/
```

### 2. Test with Real Data

Replace `data/test_all.jsonl` with your actual Presidio detection results:

```json
{"recordId": "scan-123-entity-1", "input": "...", "entityType": "EMAIL", "entityValue": "...", "metadata": {"source": "presidio", "confidence": 0.95}}
```

### 3. Deploy to Production

Once you identify the best model:
1. Integrate into `piiVerifier` service (already implemented in `andromeda-on-premise`)
2. Configure in `config/piiVerifierModels.json`
3. Deploy lightweight worker
4. Monitor performance metrics

---

## Summary

✅ **What You Have:**
- 5 model runners (CPU and GPU)
- Test data generator
- Results analyzer
- Sample test data
- Complete documentation

✅ **What You Can Do:**
- Test DistilBERT (66M params) - recommended starting point
- Test GLiNER for financial entities
- Test PHI BERT for medical entities
- Test Llama/Qwen LLMs (if you have GPU)
- Compare all models side-by-side
- Match results to inputs via recordId

✅ **Next Actions:**
1. Run `bash quickstart.sh` to setup
2. Download DistilBERT model
3. Run first experiment on sample data
4. Review results and latency
5. Scale to larger datasets
6. Compare models and choose best fit

---

**Need Help?** Check:
- `README.md` - Main documentation
- `PRESIDIO_VERIFIER_ANALYSIS.md` - Research and recommendations
- `PII_VERIFIER_IMPLEMENTATION_PLAN.md` - Implementation details

**Have a GPU?** Try Llama 3.2 3B or Qwen 2.5 3B with 4-bit quantization!

**Don't have a GPU?** DistilBERT on CPU is excellent (9.5ms latency, F1=0.97)!
