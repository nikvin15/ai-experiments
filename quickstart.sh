#!/bin/bash
# Quick Start Script for PII Verifier Experiments

set -e

echo "================================"
echo "PII Verifier Experiments Setup"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data results models

# Generate sample test data
echo ""
echo "Generating sample test data..."
python3 generate_test_data.py --output data/test_emails.jsonl --count 50 --type email
python3 generate_test_data.py --output data/test_persons.jsonl --count 50 --type person
python3 generate_test_data.py --output data/test_financial.jsonl --count 50 --type financial
python3 generate_test_data.py --output data/test_medical.jsonl --count 50 --type medical

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download models:"
echo "   python3 -c \"from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('ai4privacy/distilbert_finetuned_ai4privacy_v2').save_pretrained('./models/distilbert_ai4privacy'); AutoTokenizer.from_pretrained('ai4privacy/distilbert_finetuned_ai4privacy_v2').save_pretrained('./models/distilbert_ai4privacy')\""
echo ""
echo "2. Run experiments:"
echo "   python3 runners/run_distilbert.py --input data/test_emails.jsonl --output results/distilbert_results.jsonl --model-path models/distilbert_ai4privacy"
echo ""
echo "3. Analyze results:"
echo "   python3 analyze_results.py --results-dir results/"
echo ""
