# Training Data Summary

**Generated:** 2025-11-18
**Total Records:** 484
**Data Elements Covered:** 121

---

## Overview

This training dataset contains **484 test cases** covering **121 unique data elements** from the Privado data classification catalog. Each data element has **4 examples** across different domains and formats.

### Key Features

- **Simplified Format**: Only 3 keys per record (recordId, input, PIIs)
- **Realistic JSON**: Uses real-world structures (user profiles, employee records, CRM data)
- **Multi-PII Detection**: 85% of JSON examples contain multiple PII types
- **Binary Classification**: PIIs array for true positives, empty array for false positives

### Distribution

- **True Positives:** 462 (95.5%)
- **False Positives:** 22 (4.5%)
- **JSON Format:** 121 (25.0%)
- **Text Format:** 363 (75.0%)

---

## Format Structure

Each record has exactly 3 keys:

```json
{
  "recordId": "de_0001_01",
  "input": "Text or JSON containing data",
  "PIIs": ["Email Address", "Phone Number"]
}
```

- **recordId**: Unique identifier (e.g., de_0001_01)
- **input**: Plain text or realistic JSON string
- **PIIs**: Array of data element names found (empty array for false positives)

### Example Records

**Text Example:**
```json
{
  "recordId": "de_0001_01",
  "input": "Patient SSN 123-45-6789 requires authorization for surgery",
  "PIIs": ["Social Security Number"]
}
```

**False Positive:**
```json
{
  "recordId": "de_0042_02",
  "input": "Bug ticket ID SSN-123-456 assigned to security team",
  "PIIs": []
}
```

**JSON with Multiple PIIs:**
```json
{
  "recordId": "de_0041_04",
  "input": "{\"employeeId\": \"EMP-8491\", \"personalInfo\": {\"fullName\": \"Sarah Williams\", \"ssn\": \"123-45-6789\", \"email\": \"sarah.williams@company.com\"}}",
  "PIIs": ["Employee Code", "First Name", "Social Security Number", "Email Address", "Phone Number", "Date of Birth", "Salary", "Bank Account Details"]
}
```

---

## Realistic JSON Templates

The dataset uses 8 domain-specific JSON templates:

1. **User Profile** - SaaS/Tech applications with user data
2. **Employee Record** - HR systems with nested payroll/personal info
3. **E-commerce Order** - Shopping platforms with customer/shipping/payment
4. **HubSpot Contact** - CRM contact records with properties
5. **Slack Message** - Collaboration tools with user profiles
6. **Resume JSON** - Applicant tracking systems with experience/education
7. **Healthcare Record** - EMR systems with patient demographics/insurance
8. **Financial Account** - Banking systems with account/credit info

---

## Data Breakdown

### By Format
- Healthcare text: ~121 records
- Finance text: ~121 records
- Mixed domains text: ~121 records (e-commerce, HR, education, government)
- Realistic JSON: ~121 records (technology, marketing, slack, hubspot, resume)

### Top Data Element Categories
1. Financial Data - Bank accounts, credit cards, salary, tax info
2. Contact Data - Email, phone, address
3. Personal Identification - Name, DOB, gender, SSN
4. Account Data - Account ID, username, password
5. Biometric Data - Fingerprints, facial recognition, voice, iris scan
6. Professional & Employment - Work history, past employers, experience
7. Health Data - Medical conditions, disabilities, certificates
8. Online Identifiers - IP address, cookies, device ID, MAC address
9. Location Data - Precise location, coordinates
10. Usage Data - Click stream, search history, browsing history

---

## Usage Guidelines

### Filtering Examples

**Filter true positives:**
```python
records_with_pii = [r for r in data if len(r['PIIs']) > 0]
```

**Filter false positives:**
```python
false_positives = [r for r in data if len(r['PIIs']) == 0]
```

**Filter JSON examples:**
```python
json_examples = [r for r in data if r['input'].strip().startswith('{')]
```

**Filter multi-PII JSON:**
```python
multi_pii = [r for r in data if r['input'].startswith('{') and len(r['PIIs']) > 3]
```

### Model Evaluation Metrics

Track these metrics during experiments:
- **Precision**: How many detected PIIs are correct?
- **Recall**: How many true PIIs were found?
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: Non-PIIs incorrectly flagged
- **Latency**: Processing time per record

---

## Running Experiments

```bash
# DistilBERT
python runners/run_distilbert.py --input data/training_all_elements.jsonl \
  --output results/distilbert_results.jsonl

# GLiNER
python runners/run_gliner.py --input data/training_all_elements.jsonl \
  --output results/gliner_results.jsonl

# PHI BERT
python runners/run_phibert.py --input data/training_all_elements.jsonl \
  --output results/phibert_results.jsonl

# Llama 3.2
python runners/run_llama.py --input data/training_all_elements.jsonl \
  --output results/llama_results.jsonl

# Qwen 2.5
python runners/run_qwen.py --input data/training_all_elements.jsonl \
  --output results/qwen_results.jsonl
```

Compare results:
```bash
python analyze_results.py --results-dir results/
```

---

**Generated by:** `generate_training_data.py`
**Format Version:** 2.0 (Simplified 3-key format)
