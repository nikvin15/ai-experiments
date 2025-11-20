#!/bin/bash

################################################################################
# PII Verification - Comprehensive Model Benchmark Script
#
# This script runs ALL models with ALL parameter variations and generates
# a consolidated CSV for easy comparison.
#
# Usage:
#   bash scripts/run_all_models.sh
#
# Options:
#   --input FILE       Input JSONL file (default: data/test_sample.jsonl)
#   --models MODEL     Comma-separated list of models to run (default: all)
#                      Options: phi3mini,phi3small,gemma2b,gemma9b,llama3b,llama8b,qwen3b,qwen1.5b
#   --variations VAR   Comma-separated variations (default: 4bit,reasoning)
#                      Options: 4bit,reasoning,vllm,fp16
#   --skip-csv         Skip CSV consolidation at the end
#   --clean            Clean existing results before running
#
# Examples:
#   # Run all models with 4-bit + reasoning (recommended)
#   bash scripts/run_all_models.sh
#
#   # Run only Phi-3 mini and Gemma 2B with all variations
#   bash scripts/run_all_models.sh --models phi3mini,gemma2b --variations 4bit,reasoning,vllm
#
#   # Run all model sizes (small to large)
#   bash scripts/run_all_models.sh --models phi3mini,phi3small,gemma2b,gemma9b,llama3b,llama8b --variations reasoning
#
#   # Run only Llama 8B with reasoning
#   bash scripts/run_all_models.sh --models llama8b --variations reasoning
#
#   # Clean and run all
#   bash scripts/run_all_models.sh --clean
################################################################################

set -e  # Exit on error

# Default values
INPUT_FILE="data/test_sample.jsonl"
MODELS="phi3mini,phi3small,gemma2b,gemma9b,llama3b,llama8b,qwen3b,qwen1.5b"
VARIATIONS="4bit,reasoning"
SKIP_CSV=false
CLEAN=false
RESULTS_DIR="results"
ANALYSIS_DIR="analysis"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --variations)
            VARIATIONS="$2"
            shift 2
            ;;
        --skip-csv)
            SKIP_CSV=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$ANALYSIS_DIR"

# Clean results if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning existing results..."
    rm -f "$RESULTS_DIR"/*_reasoning.jsonl
    echo "✓ Cleaned results directory"
fi

# Convert comma-separated to arrays
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
IFS=',' read -ra VAR_ARRAY <<< "$VARIATIONS"

echo "=============================================================================="
echo "PII VERIFICATION - COMPREHENSIVE MODEL BENCHMARK"
echo "=============================================================================="
echo "Input file: $INPUT_FILE"
echo "Models to run: ${MODEL_ARRAY[*]}"
echo "Variations: ${VAR_ARRAY[*]}"
echo "Results directory: $RESULTS_DIR"
echo "=============================================================================="
echo ""

# Track success/failure
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
declare -a FAILED_COMMANDS

# Helper function to run a model
run_model() {
    local runner=$1
    local model_path=$2
    local output_file=$3
    local extra_args=$4

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$TOTAL_RUNS] Running: $output_file"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cmd="python3 runners/$runner --input $INPUT_FILE --output $RESULTS_DIR/$output_file --model-path $model_path $extra_args"
    echo "Command: $cmd"
    echo ""

    if eval $cmd; then
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        echo "✓ Success: $output_file"
    else
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "✗ Failed: $output_file"
        FAILED_COMMANDS+=("$cmd")
    fi
}

# Check if variation should run
should_run_variation() {
    local var=$1
    for v in "${VAR_ARRAY[@]}"; do
        if [ "$v" = "$var" ]; then
            return 0
        fi
    done
    return 1
}

# Check if model should run
should_run_model() {
    local model=$1
    for m in "${MODEL_ARRAY[@]}"; do
        if [ "$m" = "$model" ]; then
            return 0
        fi
    done
    return 1
}

#------------------------------------------------------------------------------
# 1. Phi-3-mini (3.8B) - Best Reasoning
#------------------------------------------------------------------------------

if should_run_model "phi3mini"; then
    echo ""
    echo "▶ Running Phi-3-mini (3.8B) - Best Reasoning"
    echo ""

    if should_run_variation "4bit"; then
        run_model "run_phi3.py" "microsoft/Phi-3-mini-4k-instruct" "phi3_mini_4bit.jsonl" ""
    fi

    if should_run_variation "reasoning"; then
        run_model "run_phi3.py" "microsoft/Phi-3-mini-4k-instruct" "phi3_mini_4bit_reasoning.jsonl" "--with-reasoning"
    fi

    if should_run_variation "vllm"; then
        run_model "run_phi3.py" "microsoft/Phi-3-mini-4k-instruct" "phi3_mini_vllm.jsonl" "--use-vllm"
        run_model "run_phi3.py" "microsoft/Phi-3-mini-4k-instruct" "phi3_mini_vllm_reasoning.jsonl" "--use-vllm --with-reasoning"
    fi

    if should_run_variation "fp16"; then
        run_model "run_phi3.py" "microsoft/Phi-3-mini-4k-instruct" "phi3_mini_fp16.jsonl" "--disable-quantization"
    fi
fi

#------------------------------------------------------------------------------
# 1.5. Phi-3-small (7B) - Enhanced Reasoning (⚠️ Requires 6GB+ VRAM)
#------------------------------------------------------------------------------

if should_run_model "phi3small"; then
    echo ""
    echo "▶ Running Phi-3-small (7B) - Enhanced Reasoning (⚠️ Requires 6GB+ VRAM)"
    echo ""

    if should_run_variation "4bit"; then
        run_model "run_phi3.py" "microsoft/Phi-3-small-8k-instruct" "phi3_small_4bit.jsonl" ""
    fi

    if should_run_variation "reasoning"; then
        run_model "run_phi3.py" "microsoft/Phi-3-small-8k-instruct" "phi3_small_4bit_reasoning.jsonl" "--with-reasoning"
    fi

    if should_run_variation "vllm"; then
        run_model "run_phi3.py" "microsoft/Phi-3-small-8k-instruct" "phi3_small_vllm.jsonl" "--use-vllm"
        run_model "run_phi3.py" "microsoft/Phi-3-small-8k-instruct" "phi3_small_vllm_reasoning.jsonl" "--use-vllm --with-reasoning"
    fi

    if should_run_variation "fp16"; then
        echo "⚠️ Warning: Phi-3-small FP16 requires 14GB VRAM - skipping by default"
        # Uncomment to run: run_model "run_phi3.py" "microsoft/Phi-3-small-8k-instruct" "phi3_small_fp16.jsonl" "--disable-quantization"
    fi
fi

#------------------------------------------------------------------------------
# 2. Gemma-2-2B - Fastest Viable
#------------------------------------------------------------------------------

if should_run_model "gemma2b"; then
    echo ""
    echo "▶ Running Gemma-2-2B - Fastest Viable"
    echo ""

    if should_run_variation "4bit"; then
        run_model "run_gemma.py" "google/gemma-2-2b-it" "gemma_2b_4bit.jsonl" ""
    fi

    if should_run_variation "reasoning"; then
        run_model "run_gemma.py" "google/gemma-2-2b-it" "gemma_2b_4bit_reasoning.jsonl" "--with-reasoning"
    fi

    if should_run_variation "vllm"; then
        run_model "run_gemma.py" "google/gemma-2-2b-it" "gemma_2b_vllm.jsonl" "--use-vllm"
        run_model "run_gemma.py" "google/gemma-2-2b-it" "gemma_2b_vllm_reasoning.jsonl" "--use-vllm --with-reasoning"
    fi

    if should_run_variation "fp16"; then
        run_model "run_gemma.py" "google/gemma-2-2b-it" "gemma_2b_fp16.jsonl" "--disable-quantization"
    fi
fi

#------------------------------------------------------------------------------
# 2.5. Gemma-2-9B - Enhanced Performance (⚠️ Requires 6GB+ VRAM)
#------------------------------------------------------------------------------

if should_run_model "gemma9b"; then
    echo ""
    echo "▶ Running Gemma-2-9B - Enhanced Performance (⚠️ Requires 6GB+ VRAM)"
    echo ""

    if should_run_variation "4bit"; then
        run_model "run_gemma.py" "google/gemma-2-9b-it" "gemma_2_9b_4bit.jsonl" ""
    fi

    if should_run_variation "reasoning"; then
        run_model "run_gemma.py" "google/gemma-2-9b-it" "gemma_2_9b_4bit_reasoning.jsonl" "--with-reasoning"
    fi

    if should_run_variation "vllm"; then
        run_model "run_gemma.py" "google/gemma-2-9b-it" "gemma_2_9b_vllm.jsonl" "--use-vllm"
        run_model "run_gemma.py" "google/gemma-2-9b-it" "gemma_2_9b_vllm_reasoning.jsonl" "--use-vllm --with-reasoning"
    fi

    if should_run_variation "fp16"; then
        echo "⚠️ Warning: Gemma-2-9B FP16 requires 18GB VRAM - skipping by default"
        # Uncomment to run: run_model "run_gemma.py" "google/gemma-2-9b-it" "gemma_2_9b_fp16.jsonl" "--disable-quantization"
    fi
fi

#------------------------------------------------------------------------------
# 3. Llama 3.2 3B - Most Accurate (≤5GB)
#------------------------------------------------------------------------------

if should_run_model "llama3b"; then
    echo ""
    echo "▶ Running Llama 3.2 3B - Most Accurate (≤5GB)"
    echo ""

    if should_run_variation "4bit"; then
        run_model "run_llama.py" "meta-llama/Llama-3.2-3B-Instruct" "llama_3.2_3b_4bit.jsonl" ""
    fi

    if should_run_variation "reasoning"; then
        run_model "run_llama.py" "meta-llama/Llama-3.2-3B-Instruct" "llama_3.2_3b_4bit_reasoning.jsonl" "--with-reasoning"
    fi

    if should_run_variation "vllm"; then
        run_model "run_llama.py" "meta-llama/Llama-3.2-3B-Instruct" "llama_3.2_3b_vllm.jsonl" "--use-vllm"
        run_model "run_llama.py" "meta-llama/Llama-3.2-3B-Instruct" "llama_3.2_3b_vllm_reasoning.jsonl" "--use-vllm --with-reasoning"
    fi

    if should_run_variation "fp16"; then
        run_model "run_llama.py" "meta-llama/Llama-3.2-3B-Instruct" "llama_3.2_3b_fp16.jsonl" "--disable-quantization"
    fi
fi

#------------------------------------------------------------------------------
# 3.1. Llama 3.1 8B - Highest Accuracy (8GB+ VRAM)
#------------------------------------------------------------------------------

if should_run_model "llama8b"; then
    echo ""
    echo "▶ Running Llama 3.1 8B - Highest Accuracy (⚠️ Requires 8GB+ VRAM)"
    echo ""

    if should_run_variation "4bit"; then
        run_model "run_llama.py" "meta-llama/Llama-3.1-8B-Instruct" "llama_3.1_8b_4bit.jsonl" ""
    fi

    if should_run_variation "reasoning"; then
        run_model "run_llama.py" "meta-llama/Llama-3.1-8B-Instruct" "llama_3.1_8b_4bit_reasoning.jsonl" "--with-reasoning"
    fi

    if should_run_variation "vllm"; then
        run_model "run_llama.py" "meta-llama/Llama-3.1-8B-Instruct" "llama_3.1_8b_vllm.jsonl" "--use-vllm"
        run_model "run_llama.py" "meta-llama/Llama-3.1-8B-Instruct" "llama_3.1_8b_vllm_reasoning.jsonl" "--use-vllm --with-reasoning"
    fi

    if should_run_variation "fp16"; then
        echo "⚠️ Warning: Llama 8B FP16 requires 16GB VRAM - skipping by default"
        # Uncomment to run: run_model "run_llama.py" "meta-llama/Llama-3.1-8B-Instruct" "llama_3.1_8b_fp16.jsonl" "--disable-quantization"
    fi
fi

#------------------------------------------------------------------------------
# 4. Qwen 2.5 3B - Balanced
#------------------------------------------------------------------------------

if should_run_model "qwen3b"; then
    echo ""
    echo "▶ Running Qwen 2.5 3B - Balanced"
    echo ""

    if should_run_variation "4bit"; then
        run_model "run_qwen.py" "Qwen/Qwen2.5-3B-Instruct" "qwen_2.5_3b_4bit.jsonl" ""
    fi

    if should_run_variation "reasoning"; then
        run_model "run_qwen.py" "Qwen/Qwen2.5-3B-Instruct" "qwen_2.5_3b_4bit_reasoning.jsonl" "--with-reasoning"
    fi

    if should_run_variation "vllm"; then
        run_model "run_qwen.py" "Qwen/Qwen2.5-3B-Instruct" "qwen_2.5_3b_vllm.jsonl" "--use-vllm"
        run_model "run_qwen.py" "Qwen/Qwen2.5-3B-Instruct" "qwen_2.5_3b_vllm_reasoning.jsonl" "--use-vllm --with-reasoning"
    fi

    if should_run_variation "fp16"; then
        run_model "run_qwen.py" "Qwen/Qwen2.5-3B-Instruct" "qwen_2.5_3b_fp16.jsonl" "--disable-quantization"
    fi
fi

#------------------------------------------------------------------------------
# 5. Qwen 2.5 1.5B - Budget Option
#------------------------------------------------------------------------------

if should_run_model "qwen1.5b"; then
    echo ""
    echo "▶ Running Qwen 2.5 1.5B - Budget Option"
    echo ""

    if should_run_variation "4bit"; then
        run_model "run_qwen.py" "Qwen/Qwen2.5-1.5B-Instruct" "qwen_2.5_1.5b_4bit.jsonl" ""
    fi

    if should_run_variation "reasoning"; then
        run_model "run_qwen.py" "Qwen/Qwen2.5-1.5B-Instruct" "qwen_2.5_1.5b_4bit_reasoning.jsonl" "--with-reasoning"
    fi

    if should_run_variation "vllm"; then
        run_model "run_qwen.py" "Qwen/Qwen2.5-1.5B-Instruct" "qwen_2.5_1.5b_vllm.jsonl" "--use-vllm"
        run_model "run_qwen.py" "Qwen/Qwen2.5-1.5B-Instruct" "qwen_2.5_1.5b_vllm_reasoning.jsonl" "--use-vllm --with-reasoning"
    fi

    if should_run_variation "fp16"; then
        run_model "run_qwen.py" "Qwen/Qwen2.5-1.5B-Instruct" "qwen_2.5_1.5b_fp16.jsonl" "--disable-quantization"
    fi
fi

#------------------------------------------------------------------------------
# Summary and CSV Consolidation
#------------------------------------------------------------------------------

echo ""
echo "=============================================================================="
echo "BENCHMARK SUMMARY"
echo "=============================================================================="
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $SUCCESSFUL_RUNS"
echo "Failed: $FAILED_RUNS"
echo "=============================================================================="

if [ $FAILED_RUNS -gt 0 ]; then
    echo ""
    echo "Failed commands:"
    for cmd in "${FAILED_COMMANDS[@]}"; do
        echo "  $cmd"
    done
    echo ""
fi

# CSV Consolidation
if [ "$SKIP_CSV" = false ] && [ $SUCCESSFUL_RUNS -gt 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "CONSOLIDATING RESULTS TO CSV"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Generate timestamp for output file
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_CSV="$ANALYSIS_DIR/all_models_comparison_${TIMESTAMP}.csv"

    # Consolidate all reasoning results
    echo "Consolidating results from: $RESULTS_DIR/*_reasoning.jsonl"
    echo "Output CSV: $OUTPUT_CSV"
    echo ""

    if python3 scripts/consolidate_results.py \
        --inputs "$RESULTS_DIR"/*_reasoning.jsonl \
        --output "$OUTPUT_CSV"; then
        echo ""
        echo "✓ CSV consolidation successful!"
        echo "✓ Results saved to: $OUTPUT_CSV"
        echo ""
        echo "Open the CSV in Excel/Google Sheets to review model comparisons."
    else
        echo ""
        echo "✗ CSV consolidation failed"
        echo "Run manually: python3 scripts/consolidate_results.py --inputs $RESULTS_DIR/*_reasoning.jsonl --output $OUTPUT_CSV"
    fi
fi

echo ""
echo "=============================================================================="
echo "BENCHMARK COMPLETE"
echo "=============================================================================="
echo ""

if [ $FAILED_RUNS -gt 0 ]; then
    exit 1
else
    exit 0
fi
