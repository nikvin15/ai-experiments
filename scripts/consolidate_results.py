"""
CSV Consolidation Script for PII Verification Results

This script takes multiple JSONL result files (with reasoning) from different models
and consolidates them into a single CSV for easy comparison.

Output CSV format:
- Input Text | PII | Model1_Verified | Model1_Reason | Model2_Verified | Model2_Reason | ...

Each row represents one PII from one input text, showing how different models evaluated it.

Usage:
    python scripts/consolidate_results.py \
        --inputs results/phi3_mini_4bit_reasoning.jsonl results/gemma_2b_4bit_reasoning.jsonl \
        --output analysis/comparison.csv

    # Or use pattern matching
    python scripts/consolidate_results.py \
        --inputs results/*_reasoning.jsonl \
        --output analysis/comparison.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class ResultConsolidator:
    """Consolidate PII verification results from multiple models into CSV."""

    def __init__(self, input_files: List[str], output_file: str):
        """
        Initialize consolidator.

        Args:
            input_files: List of JSONL result files (with reasoning mode)
            output_file: Output CSV file path
        """
        self.input_files = [Path(f) for f in input_files]
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Data structure: {record_id: {pii_name: {model_name: (verified, reason)}}}
        self.consolidated_data = defaultdict(lambda: defaultdict(dict))

        # Track all unique PIIs per record
        self.record_piis = defaultdict(set)

        # Track model names
        self.model_names = []

    def load_results(self):
        """Load all result files and organize by record and PII."""
        for input_file in self.input_files:
            if not input_file.exists():
                print(f"Warning: File not found: {input_file}")
                continue

            print(f"Loading: {input_file}")

            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        self._process_record(record, input_file.stem)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON in {input_file}: {e}")
                        continue

    def _process_record(self, record: Dict, model_name: str):
        """
        Process a single record and extract PII verification results.

        Args:
            record: Record from JSONL file
            model_name: Name of the model (extracted from filename)
        """
        record_id = record.get("recordId", "unknown")
        input_text = record.get("input", "")
        verified_piis = record.get("verified_piis", [])

        # Track model name
        if model_name not in self.model_names:
            self.model_names.append(model_name)

        # Process verified PIIs
        if isinstance(verified_piis, list):
            for pii_result in verified_piis:
                if isinstance(pii_result, dict):
                    pii_name = pii_result.get("pii", "")
                    verified = pii_result.get("verified", False)
                    reason = pii_result.get("reason", "No reason provided")

                    # Store result
                    self.consolidated_data[record_id][pii_name][model_name] = (verified, reason)
                    self.record_piis[record_id].add(pii_name)

                    # Store input text (will be overwritten but should be same for all models)
                    if "input_text" not in self.consolidated_data[record_id]:
                        self.consolidated_data[record_id]["input_text"] = input_text

    def write_csv(self):
        """Write consolidated results to CSV file."""
        # Create CSV header
        header = ["Input Text", "PII"]

        # Add columns for each model (Verified, Reason)
        for model_name in self.model_names:
            header.append(f"{model_name}_Verified")
            header.append(f"{model_name}_Reason")

        print(f"\nWriting CSV to: {self.output_file}")
        print(f"Models: {', '.join(self.model_names)}")

        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            # Write rows (one per PII per input)
            total_rows = 0
            for record_id in sorted(self.consolidated_data.keys()):
                input_text = self.consolidated_data[record_id].get("input_text", "")

                # Get all PIIs for this record
                piis = sorted(self.record_piis[record_id])

                for pii_name in piis:
                    row = [
                        self._truncate_text(input_text, 200),  # Truncate long texts
                        pii_name
                    ]

                    # Add results from each model
                    for model_name in self.model_names:
                        if model_name in self.consolidated_data[record_id][pii_name]:
                            verified, reason = self.consolidated_data[record_id][pii_name][model_name]
                            row.append("YES" if verified else "NO")
                            row.append(reason)
                        else:
                            # Model didn't process this PII (missing in results)
                            row.append("N/A")
                            row.append("Not processed by this model")

                    writer.writerow(row)
                    total_rows += 1

            print(f"Total rows written: {total_rows}")

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("CONSOLIDATION SUMMARY")
        print("=" * 80)

        print(f"\nTotal records: {len(self.consolidated_data)}")
        print(f"Total models: {len(self.model_names)}")
        print(f"Models: {', '.join(self.model_names)}")

        total_piis = sum(len(piis) for piis in self.record_piis.values())
        print(f"Total PIIs across all records: {total_piis}")

        print("\nPer-record breakdown:")
        for record_id in sorted(self.record_piis.keys()):
            pii_count = len(self.record_piis[record_id])
            print(f"  {record_id}: {pii_count} PIIs")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate PII verification results from multiple models into CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Consolidate specific files
  python scripts/consolidate_results.py \\
      --inputs results/phi3_mini_4bit_reasoning.jsonl results/gemma_2b_4bit_reasoning.jsonl \\
      --output analysis/comparison.csv

  # Use pattern matching (shell expands wildcards)
  python scripts/consolidate_results.py \\
      --inputs results/*_reasoning.jsonl \\
      --output analysis/comparison.csv

  # Consolidate all 4-bit reasoning results
  python scripts/consolidate_results.py \\
      --inputs results/*_4bit_reasoning.jsonl \\
      --output analysis/4bit_comparison.csv
        """
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files with reasoning results (can use wildcards)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file path"
    )

    args = parser.parse_args()

    # Check if any input files exist
    existing_files = [f for f in args.inputs if Path(f).exists()]
    if not existing_files:
        print(f"Error: No input files found")
        print(f"Searched for: {args.inputs}")
        sys.exit(1)

    print(f"Found {len(existing_files)} input files")

    # Consolidate results
    consolidator = ResultConsolidator(existing_files, args.output)
    consolidator.load_results()
    consolidator.write_csv()
    consolidator.print_summary()

    print(f"\n✓ CSV file saved to: {args.output}")


if __name__ == "__main__":
    main()
