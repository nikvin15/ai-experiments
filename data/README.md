# Training Data Directory

This directory contains the comprehensive training dataset for PII verification models.

## Files

### 1. training_all_elements.jsonl
**Main training dataset** - 484 records covering 121 unique data elements

- **Format:** JSON Lines (one JSON object per line) - **SIMPLIFIED 3-KEY FORMAT**
- **Encoding:** UTF-8
- **Records:** 484
- **True Positives:** 462 (95.5%)
- **False Positives:** 22 (4.5%)
- **JSON Examples:** 121 (25.0%) with realistic structures
- **Multi-PII JSON:** 85% of JSON examples contain multiple PII types

### 2. training_summary.md
**Detailed documentation** including:
- Dataset statistics and distribution
- Format structure with examples
- Realistic JSON templates overview
- Usage guidelines for training/evaluation
- Model experiment commands

### 3. sample_test.jsonl
**Sample test data** - 10 manually curated examples for quick testing

## Quick Start

### View a Sample Record (New Simplified Format)
```bash
head -1 training_all_elements.jsonl | python3 -m json.tool
```

**Output:**
```json
{
  "recordId": "de_0001_01",
  "input": "Background check flagged criminal history...",
  "PIIs": ["Criminal Records History"]
}
```

### Count Records
```bash
wc -l training_all_elements.jsonl
```

### Filter by True Positives (records with PIIs)
```bash
cat training_all_elements.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if len(r['PIIs']) > 0:
        print(json.dumps(r))
"
```

### Filter by False Positives (empty PIIs array)
```bash
cat training_all_elements.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if len(r['PIIs']) == 0:
        print(json.dumps(r))
"
```

### Filter JSON Format Examples
```bash
cat training_all_elements.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if r['input'].strip().startswith('{'):
        print(json.dumps(r))
"
```

### Find Multi-PII JSON Examples
```bash
cat training_all_elements.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if r['input'].strip().startswith('{') and len(r['PIIs']) > 3:
        print(json.dumps(r, indent=2))
        break
"
```

## Running Model Experiments

### DistilBERT (All data elements)
```bash
python3 ../runners/run_distilbert.py \
  --input training_all_elements.jsonl \
  --output ../results/distilbert_training_results.jsonl \
  --model-path ../models/distilbert_ai4privacy \
  --batch-size 32
```

### GLiNER (Financial entities only)
```bash
# Filter for financial data elements
cat training_all_elements.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if 'Financial' in r['metadata']['category']:
        print(line.strip())
" > training_financial.jsonl

python3 ../runners/run_gliner.py \
  --input training_financial.jsonl \
  --output ../results/gliner_training_results.jsonl \
  --model-path ../models/gliner_base \
  --batch-size 16
```

### PHI BERT (Health data only)
```bash
# Filter for health data elements
cat training_all_elements.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if 'Health' in r['metadata']['category']:
        print(line.strip())
" > training_health.jsonl

python3 ../runners/run_phibert.py \
  --input training_health.jsonl \
  --output ../results/phibert_training_results.jsonl \
  --model-path ../models/phi_bert \
  --batch-size 32
```

### Llama 3.2 (Full dataset)
```bash
python3 ../runners/run_llama.py \
  --input training_all_elements.jsonl \
  --output ../results/llama_training_results.jsonl \
  --model-path ../models/llama_3.2_3b \
  --batch-size 4 \
  --use-4bit
```

### Qwen 2.5 (Full dataset)
```bash
python3 ../runners/run_qwen.py \
  --input training_all_elements.jsonl \
  --output ../results/qwen_training_results.jsonl \
  --model-path ../models/qwen_2.5_3b \
  --batch-size 4 \
  --use-4bit
```

## Analyze Results

After running experiments, compare all models:

```bash
python3 ../analyze_results.py --results-dir ../results/
```

This will generate:
- Performance comparison table
- Latency statistics (p50, p95, p99)
- Confidence distribution
- Verification rate analysis

## Dataset Statistics (Updated)

| Metric | Value |
|--------|-------|
| Total Records | 484 |
| Data Elements Covered | 121 |
| True Positives | 462 (95.5%) |
| False Positives | 22 (4.5%) |
| JSON Format | 121 (25.0%) |
| Text Format | 363 (75.0%) |
| Multi-PII JSON | 103 (85.1% of JSON) |

### Format Breakdown
- **Text Examples:** 363 records (3 per data element)
  - Healthcare domain: ~121 records
  - Finance domain: ~121 records
  - Mixed domains: ~121 records
- **JSON Examples:** 121 records (1 per data element)
  - User profiles, employee records, CRM data
  - E-commerce orders, healthcare records
  - HubSpot contacts, Slack messages, resumes
  - 85% contain multiple PII types

### Realistic JSON Templates
The JSON examples use 8 realistic templates:
1. **User Profile** - Tech/SaaS applications
2. **Employee Record** - HR systems with nested data
3. **E-commerce Order** - Shopping platforms
4. **HubSpot Contact** - CRM data structure
5. **Slack Message** - Collaboration tools
6. **Resume JSON** - Applicant tracking systems
7. **Healthcare Record** - EMR/patient data
8. **Financial Account** - Banking systems

## Data Generation

To regenerate or modify the training data:

```bash
python3 ../generate_training_data.py \
  --input-json /path/to/default_data_elements.json \
  --output training_all_elements.jsonl
```

## Notes

- **Simplified Format**: Only 3 keys per record (recordId, input, PIIs)
- **PIIs as Strings**: Array of data element names (not objects with values)
- **Binary Classification**: Empty array for false positives, populated array for true positives
- **Realistic JSON**: 8 domain-specific templates (not obvious data_element/value patterns)
- **Multi-PII Detection**: 85% of JSON examples contain multiple PII types
- **Synthetic Data**: All PII values are synthetically generated
- **Edge Cases**: False positives test patterns that look like PII but aren't
- **Compatible**: Works with all runners in `../runners/` directory

## Support

For issues or questions:
- Check `training_summary.md` for detailed documentation
- Review `training_schema.json` for format specification
- See parent directory README for experiment setup

---

**Generated:** 2025-11-18
**Generator:** `../generate_training_data.py`
**Source:** Privado default_data_elements.json (121 elements)
