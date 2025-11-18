# PII Verifier Model Experiments

Experiment framework for testing CPU and GPU-based PII verification models.

## Directory Structure

```
ai-experiments/
├── runners/              # Model runner scripts
│   ├── run_distilbert.py       # CPU: DistilBERT PII verification
│   ├── run_gliner.py           # CPU: GLiNER financial entities
│   ├── run_phibert.py          # CPU: PHI BERT medical entities
│   ├── run_llama.py            # GPU: Llama 3.2 (1B/3B)
│   ├── run_qwen.py             # GPU: Qwen 2.5 (0.5B/1.5B/3B)
│   └── common.py               # Shared utilities
├── data/                 # Input JSONL files
│   ├── test_emails.jsonl
│   ├── test_persons.jsonl
│   ├── test_financial.jsonl
│   └── test_medical.jsonl
├── results/              # Output JSONL files
│   └── [timestamp]_[model]_results.jsonl
├── models/               # Downloaded model weights
└── requirements.txt      # Python dependencies
```

## Input Format

Each line in input JSONL file should have:

```json
{
  "recordId": "unique-id-123",
  "input": "Contact john@example.com for support",
  "entityType": "EMAIL",
  "entityValue": "john@example.com",
  "metadata": {
    "source": "presidio",
    "confidence": 0.95
  }
}
```

## Output Format

Each line in output JSONL file will have:

```json
{
  "recordId": "unique-id-123",
  "input": "Contact john@example.com for support",
  "entityType": "EMAIL",
  "entityValue": "john@example.com",
  "result": {
    "verified": true,
    "confidence": 0.92,
    "reason": "Valid email format in contact context",
    "latencyMs": 12.5
  },
  "model": {
    "name": "distilbert_ai4privacy_v2",
    "type": "cpu",
    "parameters": "66M"
  },
  "timestamp": "2025-11-18T10:30:45.123Z"
}
```

## Setup

### 1. Virtual Environment Setup (Recommended)

Create and activate a virtual environment to isolate dependencies:

**Linux/Mac:**
```bash
cd ai-experiments
python3 -m venv env
source env/bin/activate
```

**Windows:**
```bash
cd ai-experiments
python3 -m venv env
env\Scripts\activate
```

### 2. Install Dependencies

**For CPU models:**
```bash
pip install -r requirements.txt
```

**For GPU models with CUDA 11.8:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install vllm  # Optional: For faster LLM inference
```

### 3. Model Auto-Download

**Models are downloaded automatically when you run a runner for the first time.** The models will be cached in the `models/` directory you specify with `--model-path`.

For example:
```bash
# First run will download the model automatically
python runners/run_distilbert.py \
  --input data/test_emails.jsonl \
  --output results/distilbert_results.jsonl \
  --model-path models/distilbert_ai4privacy \
  --batch-size 32
```

**Note**: Llama models require HuggingFace authentication. Set your token:
```bash
export HF_TOKEN=your_token_here
```

**Manual Download (Optional)**: If you prefer to pre-download models:
```bash
# DistilBERT AI4Privacy (66M params)
python -c "from transformers import AutoModel, AutoTokenizer; \
  AutoModel.from_pretrained('ai4privacy/distilbert_finetuned_ai4privacy_v2').save_pretrained('./models/distilbert_ai4privacy'); \
  AutoTokenizer.from_pretrained('ai4privacy/distilbert_finetuned_ai4privacy_v2').save_pretrained('./models/distilbert_ai4privacy')"

# GLiNER (400M params)
python -c "from gliner import GLiNER; \
  model = GLiNER.from_pretrained('urchade/gliner_base'); \
  model.save_pretrained('./models/gliner_base')"

# PHI BERT (110M params)
python -c "from transformers import AutoModelForTokenClassification, AutoTokenizer; \
  AutoModelForTokenClassification.from_pretrained('obi/deid_bert_i2b2').save_pretrained('./models/phi_bert'); \
  AutoTokenizer.from_pretrained('obi/deid_bert_i2b2').save_pretrained('./models/phi_bert')"

# Llama 3.2 3B (GPU)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct').save_pretrained('./models/llama_3.2_3b'); \
  AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct').save_pretrained('./models/llama_3.2_3b')"

# Qwen 2.5 3B (GPU)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct').save_pretrained('./models/qwen_2.5_3b'); \
  AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct').save_pretrained('./models/qwen_2.5_3b')"
```

## Running Experiments

### CPU Models

#### DistilBERT (General PII)
```bash
python runners/run_distilbert.py \
  --input data/test_emails.jsonl \
  --output results/distilbert_results.jsonl \
  --model-path models/distilbert_ai4privacy \
  --batch-size 32

# With ONNX optimization
python runners/run_distilbert.py \
  --input data/test_emails.jsonl \
  --output results/distilbert_onnx_results.jsonl \
  --model-path models/distilbert_ai4privacy \
  --use-onnx \
  --batch-size 32
```

#### GLiNER (Financial Entities)
```bash
python runners/run_gliner.py \
  --input data/test_financial.jsonl \
  --output results/gliner_results.jsonl \
  --model-path models/gliner_base \
  --batch-size 16
```

#### PHI BERT (Medical Entities)
```bash
python runners/run_phibert.py \
  --input data/test_medical.jsonl \
  --output results/phibert_results.jsonl \
  --model-path models/phi_bert \
  --batch-size 32
```

### GPU Models

#### Llama 3.2 3B
```bash
python runners/run_llama.py \
  --input data/test_emails.jsonl \
  --output results/llama_3b_results.jsonl \
  --model-path models/llama_3.2_3b \
  --batch-size 4 \
  --use-4bit  # 4-bit quantization

# With vLLM (faster)
python runners/run_llama.py \
  --input data/test_emails.jsonl \
  --output results/llama_3b_vllm_results.jsonl \
  --model-path models/llama_3.2_3b \
  --use-vllm \
  --batch-size 8
```

#### Qwen 2.5 3B
```bash
python runners/run_qwen.py \
  --input data/test_emails.jsonl \
  --output results/qwen_3b_results.jsonl \
  --model-path models/qwen_2.5_3b \
  --batch-size 4 \
  --use-4bit

# With vLLM (faster)
python runners/run_qwen.py \
  --input data/test_emails.jsonl \
  --output results/qwen_3b_vllm_results.jsonl \
  --model-path models/qwen_2.5_3b \
  --use-vllm \
  --batch-size 8
```

## Analyzing Results

```bash
# Generate comparison report
python analyze_results.py \
  --results-dir results/ \
  --output comparison_report.html

# Show latency statistics
python analyze_results.py \
  --results-dir results/ \
  --metric latency

# Show accuracy comparison
python analyze_results.py \
  --results-dir results/ \
  --metric accuracy
```

## Sample Test Data

Create test data files:

```bash
# Generate sample test data
python generate_test_data.py \
  --output data/test_emails.jsonl \
  --count 100 \
  --type email

python generate_test_data.py \
  --output data/test_financial.jsonl \
  --count 100 \
  --type financial

python generate_test_data.py \
  --output data/test_medical.jsonl \
  --count 100 \
  --type medical
```

## Performance Targets

Based on PRESIDIO_VERIFIER_ANALYSIS.md recommendations:

| Model | Type | Memory | Latency (p95) | F1 Score | Cost/Month |
|-------|------|--------|---------------|----------|------------|
| DistilBERT | CPU | <100MB | 9.5ms | 0.97 | $80/month |
| GLiNER | CPU | 200MB | 75ms | 0.98 | $85/month |
| PHI BERT | CPU | 100MB | 25ms | 0.94 | $80/month |
| Llama 3.2 3B | GPU | 2-3GB | 50-120ms | 0.89 | $95/month |
| Qwen 2.5 3B | GPU | 1.5-2GB | 40-100ms | 0.87 | $95/month |

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 1

# Use 4-bit quantization
--use-4bit

# Use CPU for LLMs (slower)
--device cpu
```

### Slow CPU Inference
```bash
# Use ONNX optimization
--use-onnx

# Increase batch size
--batch-size 64

# Use INT8 quantization
--quantize int8
```

### Model Download Issues
```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Use mirror
export HF_ENDPOINT=https://hf-mirror.com
```

## Notes

- **Recommended**: Start with CPU models (DistilBERT, GLiNER, PHI BERT) as they're faster and cheaper
- **GPU Requirements**: NVIDIA GPU with >=8GB VRAM for LLMs
- **Batch Processing**: Larger batches = better throughput but higher latency per item
- **ONNX**: 3-10x speedup for CPU models, highly recommended
- **vLLM**: 2-4x speedup for GPU LLMs, use when available
