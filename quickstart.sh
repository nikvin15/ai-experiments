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

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv env
    echo "Virtual environment created!"
    echo ""
    echo "Please activate the virtual environment:"
    echo "  Linux/Mac: source env/bin/activate"
    echo "  Windows:   env\\Scripts\\activate"
    echo ""
    echo "Then run this script again."
    exit 0
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "⚠️  Virtual environment not activated!"
    echo ""
    echo "Please activate it first:"
    echo "  Linux/Mac: source env/bin/activate"
    echo "  Windows:   env\\Scripts\\activate"
    echo ""
    echo "Then run this script again."
    exit 1
fi

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
echo "1. Run experiments (models auto-download on first run):"
echo "   python3 runners/run_distilbert.py --input data/test_emails.jsonl --output results/distilbert_results.jsonl --model-path models/distilbert_ai4privacy"
echo ""
echo "2. Analyze results:"
echo "   python3 analyze_results.py --results-dir results/"
echo ""
echo "Note: Models will be automatically downloaded on first run."
echo "First run takes a few minutes, subsequent runs are instant."
echo ""
