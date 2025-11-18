# PII Verifier Model Experiments

Experiment framework for testing CPU and GPU-based PII verification models with **batch verification** approach.

## Key Features

- **4-bit Quantization by Default** - 65-75% memory reduction for GPU models
- **Batch PII Verification** - LLMs filter false positives from CPU model detections
- **Two Output Modes** - Simple (list) or reasoning (detailed explanations)
- **Auto-Download Models** - Models download from HuggingFace automatically
- **Comprehensive Training Data** - 484 records covering 121 data elements

## Directory Structure

```
ai-experiments/
├── runners/              # Model runner scripts
│   ├── run_llama.py            # GPU: Llama 3.2 (1B/3B) - 4-bit default
│   ├── run_qwen.py             # GPU: Qwen 2.5 (0.5B/1.5B/3B) - 4-bit default
│   ├── run_distilbert.py       # CPU: DistilBERT PII verification
│   ├── run_gliner.py           # CPU: GLiNER financial entities
│   ├── run_phibert.py          # CPU: PHI BERT medical entities
│   ├── common.py               # Shared utilities
│   └── README.md               # Detailed runner documentation
├── data/                 # Training and test JSONL files
│   ├── training_all_elements.jsonl  # 484 records, 121 data elements
│   ├── training_summary.md          # Dataset documentation
│   └── README.md                    # Data format guide
├── results/              # Output JSONL files
│   └── [model]_results.jsonl
├── models/               # Downloaded model weights (auto-created)
├── MEMORY_GUIDE.md       # GPU memory requirements guide
└── requirements.txt      # Python dependencies
```

## Input Format (Simplified 3-Key Format)

**GPU models (Llama/Qwen)** use the new simplified format:

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
- `PIIs` - Array of PII names detected by CPU models (empty for false positives)

See [data/README.md](data/README.md) for the complete dataset documentation.

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
      "reason": "Valid SSN pattern used in medical context for patient authorization"
    }
  ],
  "latency_ms": 52.8,
  "model": "llama_3.2_3b_4bit_reasoning"
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

## Quick Start

### Basic Usage (4-bit Default, Simple Mode)

```bash
# Llama 3.2 3B (2.5GB VRAM, 4-bit enabled by default)
python runners/run_llama.py \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_results.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct

# Qwen 2.5 3B (2.3GB VRAM, 4-bit enabled by default)
python runners/run_qwen.py \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b_results.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct
```

### With Reasoning (Detailed Explanations)

```bash
# Get detailed reasoning for each PII verification
python runners/run_llama.py \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_reasoning.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --with-reasoning
```

### Advanced Options

```bash
# Disable 4-bit quantization (use FP16, requires 6.8GB VRAM)
python runners/run_llama.py \
  --input data/training_all_elements.jsonl \
  --output results/llama_3b_fp16.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --disable-quantization

# Use vLLM for faster inference (requires 7.5GB VRAM)
python runners/run_qwen.py \
  --input data/training_all_elements.jsonl \
  --output results/qwen_3b_vllm.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --use-vllm

# Smaller models for low memory GPUs (4GB VRAM)
python runners/run_qwen.py \
  --input data/training_all_elements.jsonl \
  --output results/qwen_0.5b.jsonl \
  --model-path Qwen/Qwen2.5-0.5B-Instruct

python runners/run_llama.py \
  --input data/training_all_elements.jsonl \
  --output results/llama_1b.jsonl \
  --model-path meta-llama/Llama-3.2-1B-Instruct
```

**For detailed usage examples, see [runners/README.md](runners/README.md)**

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

## Memory Requirements

**4-bit quantization is enabled by default** for optimal memory usage.

| Model | FP16 Memory | 4-bit Memory (Default) | Memory Saved |
|-------|-------------|------------------------|--------------|
| Qwen 2.5 0.5B | 1.5GB | **0.5GB** | 67% |
| Llama 3.2 1B | 2.5GB | **1.0GB** | 60% |
| Qwen 2.5 1.5B | 3.5GB | **1.2GB** | 66% |
| Qwen 2.5 3B | 6.5GB | **2.3GB** | 65% |
| Llama 3.2 3B | 6.8GB | **2.5GB** | 63% |

**See [MEMORY_GUIDE.md](MEMORY_GUIDE.md) for detailed memory calculations and GPU recommendations.**

## Troubleshooting

### CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
1. Use smaller model (0.5B or 1B instead of 3B)
2. Ensure 4-bit quantization is enabled (it's default - don't use `--disable-quantization`)
3. Check GPU memory: `nvidia-smi`
4. Close other GPU processes

### Slow Inference

**Issue:** Inference slower than expected

**Solutions:**
1. Verify GPU is being used:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Try vLLM if you have enough VRAM:
   ```bash
   python runners/run_qwen.py --use-vllm ...
   ```
3. Use FP16 instead of 4-bit (trades memory for speed):
   ```bash
   python runners/run_llama.py --disable-quantization ...
   ```

### HuggingFace Authentication

**Error:** `Repository not found` for Llama models

**Solution:**
```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Accept Llama license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

### Model Download Issues

**Issue:** Model download fails or takes too long

**Solutions:**
1. Check internet connection
2. Download manually using HuggingFace CLI
3. Use local model path after download

## Documentation

- **[MEMORY_GUIDE.md](MEMORY_GUIDE.md)** - Detailed GPU memory requirements and calculations
- **[runners/README.md](runners/README.md)** - Comprehensive runner documentation and examples
- **[data/README.md](data/README.md)** - Training data format and statistics
- **[data/training_summary.md](data/training_summary.md)** - Dataset overview and usage

## Notes

- **4-bit quantization is enabled by default** - Saves 65-75% memory with minimal accuracy loss
- **Batch verification** - LLMs verify multiple PIIs per input (not single-entity verification)
- **Two output modes** - Simple (list of strings) or reasoning (detailed explanations)
- **Auto-download** - Models download from HuggingFace automatically on first run
- **GPU Requirements**: NVIDIA GPU with 4GB+ VRAM (8GB+ recommended for 3B models)
- **vLLM**: 2-4x speedup but requires more memory (no quantization support)
