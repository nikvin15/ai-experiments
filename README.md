# PII Verifier Model Experiments

Experiment framework for testing CPU and GPU-based PII verification models with **batch verification** approach.

## Key Features

- **4-bit Quantization by Default** - 65-75% memory reduction for GPU models (all ≤5GB RAM)
- **Batch PII Verification** - LLMs verify multiple PIIs per input with 3-test framework
- **121 Data Element Descriptions** - Context-aware verification using element definitions
- **Two Output Modes** - Simple (comma-separated) or reasoning (JSON with explanations)
- **4 Optimized Runners** - Phi-3, Gemma-2, Llama 3.2, Qwen 2.5 with universal prompts
- **Auto-Download Models** - Models download from HuggingFace automatically
- **Comprehensive Training Data** - 484 records covering all 121 data elements

## Directory Structure

```
ai-experiments/
├── runners/              # Model runner scripts
│   ├── run_phi3.py             # GPU: Phi-3-mini (3.8B) - 4-bit default
│   ├── run_gemma.py            # GPU: Gemma-2-2B (2B) - 4-bit default
│   ├── run_llama.py            # GPU: Llama 3.2 (1B/3B) - 4-bit default
│   ├── run_qwen.py             # GPU: Qwen 2.5 (0.5B/1.5B/3B) - 4-bit default
│   ├── run_distilbert.py       # CPU: DistilBERT PII verification
│   ├── run_gliner.py           # CPU: GLiNER financial entities
│   ├── run_phibert.py          # CPU: PHI BERT medical entities
│   ├── common.py               # Shared utilities & prompt templates
│   └── README.md               # Detailed runner documentation
├── data/                 # Training and test JSONL files
│   ├── training_all_elements.jsonl  # 484 records, 121 data elements
│   ├── test_sample.jsonl            # 3 records for quick testing
│   ├── training_summary.md          # Dataset documentation
│   └── README.md                    # Data format guide
├── results/              # Output JSONL files
│   └── [model]_results.jsonl
├── models/               # Downloaded model weights (auto-created)
├── default_data_elements.json  # 121 PII element descriptions
├── MEMORY_GUIDE.md       # GPU memory requirements guide
├── EXPERIMENTS_GUIDE.md  # Experiment workflow guide
└── requirements.txt      # Python dependencies
```

## Input Format (Simplified 3-Key Format)

**GPU models (Phi-3/Gemma/Llama/Qwen)** use the simplified format:

```json
{
  "recordId": "test_001",
  "input": "Mr. Adolphus Reagan Ziemann, as a Central Principal Applications Executive at McLaughlin, Nader and Purdy, your knowledge of change management is vital for our company's transformation.",
  "PIIs": ["First Name", "Past Roles or Positions"]
}
```

**Fields:**
- `recordId` - Unique identifier
- `input` - Text or JSON string to analyze
- `PIIs` - Array of PII names detected by CPU models (empty for false positives)

**Test Samples** (`data/test_sample.jsonl`):
- **test_001**: Professional context with name and job title
- **test_002**: Technical context with name and IP address
- **test_003**: Medical context with multiple sensitive PIIs (name, nationality, address, medical condition, etc.)

See [data/README.md](data/README.md) for the complete dataset documentation.

## Output Format

### Simple Mode (Default)

```json
{
  "recordId": "test_001",
  "input": "Mr. Adolphus Reagan Ziemann, as a Central Principal Applications Executive...",
  "detected_piis": ["First Name", "Past Roles or Positions"],
  "verified_piis": ["First Name", "Past Roles or Positions"],
  "latency_ms": 845.2,
  "model": "phi3_mini_4bit"
}
```

### Reasoning Mode (--with-reasoning)

```json
{
  "recordId": "test_002",
  "input": "Hi Alberta, there have been indications of compromised data linked with our server IP address 66.168.73.147...",
  "detected_piis": ["First Name", "IP Address"],
  "verified_piis": [
    {
      "pii": "First Name",
      "verified": true,
      "reason": "Alberta is a personal first name used to address an individual in business context"
    },
    {
      "pii": "IP Address",
      "verified": false,
      "reason": "Server IP address belongs to organization infrastructure, not personal device"
    }
  ],
  "latency_ms": 952.8,
  "model": "phi3_mini_4bit_reasoning"
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

## Available Models

### Recommended Models (≤5GB RAM with 4-bit quantization)

| Model | Parameters | FP16 RAM | 4-bit RAM | Latency | Accuracy | Status | Recommendation |
|-------|------------|----------|-----------|---------|----------|--------|----------------|
| **Phi-3-mini** | 3.8B | 7.6GB | **2.8GB** | 800-1200ms | 85-92% | ⭐⭐⭐⭐⭐ | **Best Reasoning** |
| **Llama 3.2 3B** | 3B | 6.8GB | **2.5GB** | 1500-2100ms | 90%+ | ⭐⭐⭐⭐⭐ | Accurate but slow |
| **Qwen 2.5 3B** | 3B | 6.5GB | **2.3GB** | 1200-1800ms | 88-93% | ⭐⭐⭐⭐⭐ | Balanced |
| **Gemma-2-2B** | 2B | 4.5GB | **1.5GB** | 600-900ms | 80-88% | ⭐⭐⭐⭐ | Fastest viable |
| **Qwen 2.5 1.5B** | 1.5B | 3.5GB | **1.2GB** | 500-800ms | 75-82% | ⭐⭐⭐ | Budget option |
| **Llama 3.2 1B** | 1B | 2.5GB | **1.0GB** | 500-800ms | 70-78% | ⭐⭐ | Too small |
| **Qwen 2.5 0.5B** | 0.5B | 1.5GB | **0.5GB** | 300-500ms | <70% | ❌ | Failed - too small |

### Models NOT Recommended (>5GB RAM after quantization)

| Model | Parameters | FP16 RAM | 4-bit RAM | Why Not Recommended |
|-------|------------|----------|-----------|---------------------|
| Llama 3.1 8B | 8B | 16GB | **6.5GB** | Exceeds 5GB limit |
| Mistral 7B | 7B | 14GB | **5.8GB** | Exceeds 5GB limit |
| Qwen 2.5 7B | 7B | 14GB | **5.5GB** | Exceeds 5GB limit |
| Gemma-2-9B | 9B | 18GB | **7.2GB** | Exceeds 5GB limit |

## Quick Start - All Commands

### Test Data

All commands use `data/test_sample.jsonl` (6 test records):
- **test_001**: Professional context - First Name, Past Roles or Positions
- **test_002**: Technical context - First Name, IP Address (server IP should be rejected)
- **test_003**: Medical context - Multiple sensitive PIIs
- **test_004**: SSN validation - Social Security Number
- **test_005**: Email/Phone false positive - Email Address, Phone Number (toll-free should be rejected)
- **test_006**: Empty PII test - Social Security Number (ticket ID, should be rejected)

---

### 1. Phi-3-mini (3.8B) - Best Reasoning ⭐⭐⭐⭐⭐

**Model name format**: `phi3_mini_4bit` | `phi3_mini_4bit_reasoning` | `phi3_mini_vllm` | `phi3_mini_4bit_vllm_reasoning`

```bash
# Default: 4-bit quantization, simple mode
# Output model name: phi3_mini_4bit
python3 runners/run_phi3.py \
  --input data/test_sample.jsonl \
  --output results/phi3_mini_4bit.jsonl \
  --model-path microsoft/Phi-3-mini-4k-instruct

# 4-bit + reasoning mode
# Output model name: phi3_mini_4bit_reasoning
python3 runners/run_phi3.py \
  --input data/test_sample.jsonl \
  --output results/phi3_mini_4bit_reasoning.jsonl \
  --model-path microsoft/Phi-3-mini-4k-instruct \
  --with-reasoning

# vLLM mode (no quantization, faster)
# Output model name: phi3_mini_vllm
python3 runners/run_phi3.py \
  --input data/test_sample.jsonl \
  --output results/phi3_mini_vllm.jsonl \
  --model-path microsoft/Phi-3-mini-4k-instruct \
  --use-vllm

# vLLM + reasoning mode
# Output model name: phi3_mini_vllm_reasoning
python3 runners/run_phi3.py \
  --input data/test_sample.jsonl \
  --output results/phi3_mini_vllm_reasoning.jsonl \
  --model-path microsoft/Phi-3-mini-4k-instruct \
  --use-vllm \
  --with-reasoning

# FP16 mode (no quantization, no vLLM)
# Output model name: phi3_mini
python3 runners/run_phi3.py \
  --input data/test_sample.jsonl \
  --output results/phi3_mini_fp16.jsonl \
  --model-path microsoft/Phi-3-mini-4k-instruct \
  --disable-quantization
```

---

### 2. Gemma-2-2B (2B) - Fastest Viable ⭐⭐⭐⭐

**Model name format**: `gemma_2b_4bit` | `gemma_2b_4bit_reasoning` | `gemma_2b_vllm` | `gemma_2b_4bit_vllm_reasoning`

```bash
# Default: 4-bit quantization, simple mode
# Output model name: gemma_2b_4bit
python3 runners/run_gemma.py \
  --input data/test_sample.jsonl \
  --output results/gemma_2b_4bit.jsonl \
  --model-path google/gemma-2-2b-it

# 4-bit + reasoning mode
# Output model name: gemma_2b_4bit_reasoning
python3 runners/run_gemma.py \
  --input data/test_sample.jsonl \
  --output results/gemma_2b_4bit_reasoning.jsonl \
  --model-path google/gemma-2-2b-it \
  --with-reasoning

# vLLM mode (no quantization, faster)
# Output model name: gemma_2b_vllm
python3 runners/run_gemma.py \
  --input data/test_sample.jsonl \
  --output results/gemma_2b_vllm.jsonl \
  --model-path google/gemma-2-2b-it \
  --use-vllm

# vLLM + reasoning mode
# Output model name: gemma_2b_vllm_reasoning
python3 runners/run_gemma.py \
  --input data/test_sample.jsonl \
  --output results/gemma_2b_vllm_reasoning.jsonl \
  --model-path google/gemma-2-2b-it \
  --use-vllm \
  --with-reasoning

# FP16 mode (no quantization, no vLLM)
# Output model name: gemma_2b
python3 runners/run_gemma.py \
  --input data/test_sample.jsonl \
  --output results/gemma_2b_fp16.jsonl \
  --model-path google/gemma-2-2b-it \
  --disable-quantization
```

---

### 3. Llama 3.2 3B - Most Accurate ⭐⭐⭐⭐⭐

**Model name format**: `llama_3.2_3b_4bit` | `llama_3.2_3b_4bit_reasoning` | `llama_3.2_3b_vllm` | `llama_3.2_3b_4bit_vllm_reasoning`

```bash
# Default: 4-bit quantization, simple mode
# Output model name: llama_3.2_3b_4bit
python3 runners/run_llama.py \
  --input data/test_sample.jsonl \
  --output results/llama_3.2_3b_4bit.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct

# 4-bit + reasoning mode
# Output model name: llama_3.2_3b_4bit_reasoning
python3 runners/run_llama.py \
  --input data/test_sample.jsonl \
  --output results/llama_3.2_3b_4bit_reasoning.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --with-reasoning

# vLLM mode (no quantization, faster)
# Output model name: llama_3.2_3b_vllm
python3 runners/run_llama.py \
  --input data/test_sample.jsonl \
  --output results/llama_3.2_3b_vllm.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --use-vllm

# vLLM + reasoning mode
# Output model name: llama_3.2_3b_vllm_reasoning
python3 runners/run_llama.py \
  --input data/test_sample.jsonl \
  --output results/llama_3.2_3b_vllm_reasoning.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --use-vllm \
  --with-reasoning

# FP16 mode (no quantization, no vLLM)
# Output model name: llama_3.2_3b
python3 runners/run_llama.py \
  --input data/test_sample.jsonl \
  --output results/llama_3.2_3b_fp16.jsonl \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --disable-quantization
```

---

### 4. Qwen 2.5 3B - Balanced ⭐⭐⭐⭐⭐

**Model name format**: `qwen_2.5_3b_4bit` | `qwen_2.5_3b_4bit_reasoning` | `qwen_2.5_3b_vllm` | `qwen_2.5_3b_4bit_vllm_reasoning`

```bash
# Default: 4-bit quantization, simple mode
# Output model name: qwen_2.5_3b_4bit
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_3b_4bit.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct

# 4-bit + reasoning mode
# Output model name: qwen_2.5_3b_4bit_reasoning
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_3b_4bit_reasoning.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --with-reasoning

# vLLM mode (no quantization, faster)
# Output model name: qwen_2.5_3b_vllm
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_3b_vllm.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --use-vllm

# vLLM + reasoning mode
# Output model name: qwen_2.5_3b_vllm_reasoning
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_3b_vllm_reasoning.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --use-vllm \
  --with-reasoning

# FP16 mode (no quantization, no vLLM)
# Output model name: qwen_2.5_3b
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_3b_fp16.jsonl \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --disable-quantization
```

---

### 5. Qwen 2.5 1.5B - Budget Option ⭐⭐⭐

**Model name format**: `qwen_2.5_1.5b_4bit` | `qwen_2.5_1.5b_4bit_reasoning` | `qwen_2.5_1.5b_vllm` | `qwen_2.5_1.5b_4bit_vllm_reasoning`

```bash
# Default: 4-bit quantization, simple mode
# Output model name: qwen_2.5_1.5b_4bit
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_1.5b_4bit.jsonl \
  --model-path Qwen/Qwen2.5-1.5B-Instruct

# 4-bit + reasoning mode
# Output model name: qwen_2.5_1.5b_4bit_reasoning
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_1.5b_4bit_reasoning.jsonl \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --with-reasoning

# vLLM mode (no quantization, faster)
# Output model name: qwen_2.5_1.5b_vllm
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_1.5b_vllm.jsonl \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --use-vllm

# vLLM + reasoning mode
# Output model name: qwen_2.5_1.5b_vllm_reasoning
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_1.5b_vllm_reasoning.jsonl \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --use-vllm \
  --with-reasoning

# FP16 mode (no quantization, no vLLM)
# Output model name: qwen_2.5_1.5b
python3 runners/run_qwen.py \
  --input data/test_sample.jsonl \
  --output results/qwen_2.5_1.5b_fp16.jsonl \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --disable-quantization
```

---

### 6. Llama 3.2 1B - Small Model ⭐⭐

**Model name format**: `llama_3.2_1b_4bit` | `llama_3.2_1b_4bit_reasoning` | `llama_3.2_1b_vllm`

```bash
# Default: 4-bit quantization, simple mode
# Output model name: llama_3.2_1b_4bit
python3 runners/run_llama.py \
  --input data/test_sample.jsonl \
  --output results/llama_3.2_1b_4bit.jsonl \
  --model-path meta-llama/Llama-3.2-1B-Instruct

# 4-bit + reasoning mode
# Output model name: llama_3.2_1b_4bit_reasoning
python3 runners/run_llama.py \
  --input data/test_sample.jsonl \
  --output results/llama_3.2_1b_4bit_reasoning.jsonl \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --with-reasoning
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

## Memory Requirements & GPU Selection

**4-bit quantization is enabled by default** for optimal memory usage (65-75% reduction).

### Recommended Models by GPU Memory

| Your GPU VRAM | Recommended Models | Best Choice |
|---------------|-------------------|-------------|
| **3GB** | Gemma-2-2B (1.5GB), Qwen 1.5B (1.2GB) | Gemma-2-2B |
| **4GB** | + Llama 3.2 3B (2.5GB), Qwen 3B (2.3GB) | Qwen 3B |
| **6GB+** | + Phi-3-mini (2.8GB) | Phi-3-mini |
| **8GB+** | All models + vLLM mode | Phi-3 + vLLM |

### Memory Usage Table

| Model | Parameters | FP16 RAM | 4-bit RAM | Memory Saved | Min GPU |
|-------|------------|----------|-----------|--------------|---------|
| Qwen 2.5 0.5B | 0.5B | 1.5GB | **0.5GB** | 67% | 2GB |
| Llama 3.2 1B | 1B | 2.5GB | **1.0GB** | 60% | 2GB |
| Qwen 2.5 1.5B | 1.5B | 3.5GB | **1.2GB** | 66% | 3GB |
| **Gemma-2-2B** | 2B | 4.5GB | **1.5GB** | 67% | **3GB** |
| Qwen 2.5 3B | 3B | 6.5GB | **2.3GB** | 65% | 4GB |
| **Llama 3.2 3B** | 3B | 6.8GB | **2.5GB** | 63% | **4GB** |
| **Phi-3-mini** | 3.8B | 7.6GB | **2.8GB** | 63% | **4GB** |

**See [MEMORY_GUIDE.md](MEMORY_GUIDE.md) for detailed memory calculations and optimization tips.**

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

## Model Selection Guide

Choose the right model for your use case:

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Best Overall Reasoning** | Phi-3-mini (3.8B) | Superior reasoning at 800-1200ms, only 2.8GB RAM |
| **Fastest Response** | Gemma-2-2B (2B) | 600-900ms latency, 1.5GB RAM, good accuracy |
| **Most Accurate** | Llama 3.2 3B | 90%+ accuracy, slower at 1500-2100ms |
| **Balanced Speed/Accuracy** | Qwen 2.5 3B | 88-93% accuracy, 1200-1800ms, 2.3GB RAM |
| **Low VRAM (3GB)** | Gemma-2-2B | Only needs 1.5GB, still viable performance |
| **Budget Option** | Qwen 2.5 1.5B | 75-82% accuracy, 500-800ms, 1.2GB RAM |

### Quick Decision Tree

```
Do you have 4GB+ VRAM?
├─ YES: Use Phi-3-mini (best reasoning)
│   └─ Need faster? Use Gemma-2-2B
│
└─ NO (only 3GB):
    └─ Use Gemma-2-2B (fastest + viable accuracy)
```

## Notes

- **4-bit quantization is enabled by default** - Saves 65-75% memory with minimal accuracy loss
- **Batch verification** - LLMs verify multiple PIIs per input using 3-test framework (OWNERSHIP, SPECIFICITY, CONTEXT)
- **Element descriptions** - Each PII type has definition/examples from 121 data elements JSON
- **Two output modes** - Simple (comma-separated) or reasoning (JSON with explanations)
- **Auto-download** - Models download from HuggingFace automatically on first run
- **GPU Requirements**: NVIDIA GPU with 3GB+ VRAM (4GB+ recommended)
- **vLLM**: 2-4x speedup but requires more memory (no quantization support)
- **All models ≤5GB RAM** - Compliant with memory constraints
