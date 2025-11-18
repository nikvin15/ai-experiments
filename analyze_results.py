"""
Analyze and Compare Model Results

Compare performance across different models and generate report.

Usage:
    python analyze_results.py --results-dir results/
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_results(file_path):
    """Load results from JSONL file."""
    results = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    return results


def analyze_model_performance(results):
    """Analyze performance metrics for a single model."""
    latencies = []
    confidences = []
    verified_count = 0
    total = len(results)

    for result in results:
        latencies.append(result['result']['latencyMs'])
        confidences.append(result['result']['confidence'])

        if result['result']['verified']:
            verified_count += 1

    return {
        'total': total,
        'verified': verified_count,
        'rejected': total - verified_count,
        'verification_rate': verified_count / total if total > 0 else 0,
        'latency': {
            'mean': float(np.mean(latencies)) if latencies else 0,
            'median': float(np.median(latencies)) if latencies else 0,
            'p50': float(np.percentile(latencies, 50)) if latencies else 0,
            'p95': float(np.percentile(latencies, 95)) if latencies else 0,
            'p99': float(np.percentile(latencies, 99)) if latencies else 0,
            'min': float(np.min(latencies)) if latencies else 0,
            'max': float(np.max(latencies)) if latencies else 0
        },
        'confidence': {
            'mean': float(np.mean(confidences)) if confidences else 0,
            'median': float(np.median(confidences)) if confidences else 0,
            'min': float(np.min(confidences)) if confidences else 0,
            'max': float(np.max(confidences)) if confidences else 0
        }
    }


def compare_models(results_dir):
    """Compare all models in results directory."""
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Load all result files
    model_stats = {}

    for result_file in results_dir.glob("*.jsonl"):
        print(f"Analyzing: {result_file.name}")

        results = load_results(result_file)

        if not results:
            print(f"  No results found, skipping")
            continue

        # Get model name from first result
        model_name = results[0]['model']['name']
        model_type = results[0]['model']['type']
        model_params = results[0]['model']['parameters']

        # Analyze
        stats = analyze_model_performance(results)
        stats['model_name'] = model_name
        stats['model_type'] = model_type
        stats['model_params'] = model_params

        model_stats[model_name] = stats

    # Print comparison table
    print("\n" + "="*80)
    print("Model Performance Comparison")
    print("="*80)

    # Header
    print(f"{'Model':<25} {'Type':<6} {'Params':<8} {'P95 Latency':<12} {'Verified':<10} {'Avg Conf':<10}")
    print("-"*80)

    # Sort by latency
    for model_name in sorted(model_stats.keys(), key=lambda x: model_stats[x]['latency']['p95']):
        stats = model_stats[model_name]

        print(
            f"{model_name:<25} "
            f"{stats['model_type']:<6} "
            f"{stats['model_params']:<8} "
            f"{stats['latency']['p95']:>10.2f}ms "
            f"{stats['verification_rate']:>9.1%} "
            f"{stats['confidence']['mean']:>9.3f}"
        )

    print("="*80)

    # Detailed stats
    print("\nDetailed Statistics:")
    print("="*80)

    for model_name, stats in model_stats.items():
        print(f"\n{model_name}:")
        print(f"  Total: {stats['total']}")
        print(f"  Verified: {stats['verified']} ({stats['verification_rate']*100:.1f}%)")
        print(f"  Rejected: {stats['rejected']}")
        print(f"  Latency:")
        print(f"    Mean: {stats['latency']['mean']:.2f}ms")
        print(f"    P50:  {stats['latency']['p50']:.2f}ms")
        print(f"    P95:  {stats['latency']['p95']:.2f}ms")
        print(f"    P99:  {stats['latency']['p99']:.2f}ms")
        print(f"  Confidence:")
        print(f"    Mean:   {stats['confidence']['mean']:.3f}")
        print(f"    Median: {stats['confidence']['median']:.3f}")

    # Best performers
    print("\n" + "="*80)
    print("Best Performers:")
    print("="*80)

    if model_stats:
        # Fastest
        fastest = min(model_stats.items(), key=lambda x: x[1]['latency']['p95'])
        print(f"Fastest (P95): {fastest[0]} - {fastest[1]['latency']['p95']:.2f}ms")

        # Highest confidence
        highest_conf = max(model_stats.items(), key=lambda x: x[1]['confidence']['mean'])
        print(f"Highest Confidence: {highest_conf[0]} - {highest_conf[1]['confidence']['mean']:.3f}")

        # Most selective
        most_selective = min(model_stats.items(), key=lambda x: x[1]['verification_rate'])
        print(f"Most Selective: {most_selective[0]} - {most_selective[1]['verification_rate']*100:.1f}% verified")


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare model results")
    parser.add_argument("--results-dir", default="results", help="Results directory")

    args = parser.parse_args()

    compare_models(args.results_dir)


if __name__ == "__main__":
    main()
