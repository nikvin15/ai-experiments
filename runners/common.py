"""
Common Utilities for Model Experiments

Shared functions for loading data, saving results, and measuring performance.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Generator, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JSONLReader:
    """Read JSONL files line by line."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

    def read(self) -> Generator[Dict, None, None]:
        """
        Read JSONL file and yield records.

        Yields:
            Dictionary with recordId, input, entityType, entityValue, metadata
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)

                    # Validate required fields
                    if 'recordId' not in record:
                        logger.warning(f"Line {line_num}: Missing recordId, skipping")
                        continue

                    if 'input' not in record:
                        logger.warning(f"Line {line_num}: Missing input, skipping")
                        continue

                    yield record

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: Invalid JSON - {e}")
                    continue

    def count(self) -> int:
        """Count total records in file."""
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


class JSONLWriter:
    """Write results to JSONL file."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self.file = open(self.file_path, 'a', encoding='utf-8')
        logger.info(f"Writing results to: {self.file_path}")

    def write(self, record: Dict):
        """
        Write single record to JSONL file.

        Args:
            record: Dictionary with result data
        """
        try:
            json_line = json.dumps(record, ensure_ascii=False)
            self.file.write(json_line + '\n')
            self.file.flush()  # Ensure written immediately
        except Exception as e:
            logger.error(f"Error writing record: {e}")

    def close(self):
        """Close file handle."""
        if self.file and not self.file.closed:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PerformanceTracker:
    """Track model performance metrics."""

    def __init__(self):
        self.latencies = []
        self.confidences = []
        self.verified_count = 0
        self.rejected_count = 0
        self.error_count = 0
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def end(self):
        """End timing."""
        self.end_time = time.time()

    def record(self, latency_ms: float, confidence: float, verified: bool):
        """
        Record single inference result.

        Args:
            latency_ms: Inference latency in milliseconds
            confidence: Model confidence (0-1)
            verified: Whether entity was verified
        """
        self.latencies.append(latency_ms)
        self.confidences.append(confidence)

        if verified:
            self.verified_count += 1
        else:
            self.rejected_count += 1

    def record_error(self):
        """Record inference error."""
        self.error_count += 1

    def get_stats(self) -> Dict:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        import numpy as np

        total = self.verified_count + self.rejected_count
        total_time = (self.end_time - self.start_time) if self.end_time else 0

        return {
            "total_processed": total,
            "verified": self.verified_count,
            "rejected": self.rejected_count,
            "verification_rate": self.verified_count / total if total > 0 else 0,
            "error_count": self.error_count,
            "latency": {
                "mean_ms": float(np.mean(self.latencies)) if self.latencies else 0,
                "median_ms": float(np.median(self.latencies)) if self.latencies else 0,
                "p50_ms": float(np.percentile(self.latencies, 50)) if self.latencies else 0,
                "p95_ms": float(np.percentile(self.latencies, 95)) if self.latencies else 0,
                "p99_ms": float(np.percentile(self.latencies, 99)) if self.latencies else 0,
                "min_ms": float(np.min(self.latencies)) if self.latencies else 0,
                "max_ms": float(np.max(self.latencies)) if self.latencies else 0
            },
            "confidence": {
                "mean": float(np.mean(self.confidences)) if self.confidences else 0,
                "median": float(np.median(self.confidences)) if self.confidences else 0,
                "min": float(np.min(self.confidences)) if self.confidences else 0,
                "max": float(np.max(self.confidences)) if self.confidences else 0
            },
            "throughput": {
                "total_time_sec": total_time,
                "items_per_sec": total / total_time if total_time > 0 else 0
            }
        }

    def print_summary(self, model_name: str):
        """Print performance summary."""
        stats = self.get_stats()

        logger.info(f"\n{'='*60}")
        logger.info(f"Performance Summary - {model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Processed: {stats['total_processed']}")
        logger.info(f"Verified: {stats['verified']} ({stats['verification_rate']*100:.1f}%)")
        logger.info(f"Rejected: {stats['rejected']}")
        logger.info(f"Errors: {stats['error_count']}")
        logger.info(f"\nLatency:")
        logger.info(f"  Mean: {stats['latency']['mean_ms']:.2f}ms")
        logger.info(f"  P50:  {stats['latency']['p50_ms']:.2f}ms")
        logger.info(f"  P95:  {stats['latency']['p95_ms']:.2f}ms")
        logger.info(f"  P99:  {stats['latency']['p99_ms']:.2f}ms")
        logger.info(f"\nConfidence:")
        logger.info(f"  Mean: {stats['confidence']['mean']:.3f}")
        logger.info(f"  Median: {stats['confidence']['median']:.3f}")
        logger.info(f"\nThroughput:")
        logger.info(f"  Total Time: {stats['throughput']['total_time_sec']:.2f}s")
        logger.info(f"  Items/sec: {stats['throughput']['items_per_sec']:.2f}")
        logger.info(f"{'='*60}\n")


def batch_generator(items: List, batch_size: int) -> Generator[List, None, None]:
    """
    Generate batches from list of items.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Yields:
        List of items (batch)
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def create_result_record(
    input_record: Dict,
    verified: bool,
    confidence: float,
    reason: str,
    latency_ms: float,
    model_name: str,
    model_type: str,
    model_params: str
) -> Dict:
    """
    Create standardized result record.

    Args:
        input_record: Original input record
        verified: Whether entity was verified
        confidence: Model confidence (0-1)
        reason: Explanation of verification
        latency_ms: Inference latency
        model_name: Name of model
        model_type: Type (cpu/gpu)
        model_params: Parameter count (e.g., "66M")

    Returns:
        Standardized result dictionary
    """
    return {
        "recordId": input_record.get("recordId"),
        "input": input_record.get("input"),
        "entityType": input_record.get("entityType"),
        "entityValue": input_record.get("entityValue"),
        "metadata": input_record.get("metadata", {}),
        "result": {
            "verified": verified,
            "confidence": round(confidence, 4),
            "reason": reason,
            "latencyMs": round(latency_ms, 2)
        },
        "model": {
            "name": model_name,
            "type": model_type,
            "parameters": model_params
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def load_pii_prompt_template() -> str:
    """
    Load prompt template for LLM-based verification (DEPRECATED - single entity).

    Returns:
        Prompt template string
    """
    return """You are a PII (Personally Identifiable Information) verification assistant. Your task is to verify if a detected entity is truly PII or a false positive.

Context: {context}
Detected Entity Type: {entity_type}
Detected Entity Value: {entity_value}

Analyze the context and determine if the detected entity is actually PII:
- Consider the surrounding text
- Check if it's used in a context that indicates personal information
- Identify false positives (e.g., "John Smith School" is not a person)

Respond in JSON format:
{{
  "verified": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}

Response:"""


def load_batch_verification_prompt_template(with_reasoning: bool = False) -> str:
    """
    Load prompt template for batch PII verification.

    Args:
        with_reasoning: If True, include reasoning in output

    Returns:
        Prompt template string for batch verification
    """
    if with_reasoning:
        output_format = """{{
  "results": [
    {{
      "pii": "Email Address",
      "verified": true,
      "reason": "Valid email used for user contact in legitimate context"
    }},
    {{
      "pii": "Phone Number",
      "verified": false,
      "reason": "1-800 number is a toll-free business line, not personal information"
    }}
  ]
}}"""
    else:
        output_format = """{{
  "verified_piis": ["Email Address", "Social Security Number"]
}}"""

    return f"""You are a PII verification assistant. CPU-based models have detected potential PIIs in the input below. Your task is to verify which ones are TRUE personal information and filter out FALSE POSITIVES.

Input: {{input_text}}

Detected PIIs: {{detected_piis}}

For each detected PII, determine if it's truly personal information:

**VERIFY as TRUE PII if:**
- It's actual personal data in context (real SSN, email, phone, name, etc.)
- It's used to identify or contact a specific individual
- It appears in a context related to a person (employee, patient, customer, user)

**REJECT as FALSE POSITIVE only if:**
- It's a company/organization name that looks like a person name (e.g., "John Smith School")
- It's a toll-free business number (1-800, 1-888, etc.)
- It's a generic system email (e.g., noreply@, support@, admin@)
- It's a placeholder or example (e.g., "555-0123" in obvious example context)
- It's a product name, ticket ID, or system identifier

**Default to TRUE:** When in doubt, verify as true PII. It's better to be cautious.

Respond ONLY in JSON format:
{output_format}

Response:"""


def parse_llm_response(response: str) -> Dict:
    """
    Parse LLM response and extract verification result (DEPRECATED - single entity).

    Args:
        response: Raw LLM response

    Returns:
        Dictionary with verified, confidence, reason
    """
    try:
        # Try to find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = response[start:end]
            result = json.loads(json_str)

            return {
                "verified": result.get("verified", True),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", "LLM verification")
            }
        else:
            # No JSON found, use heuristic
            response_lower = response.lower()
            verified = "true" in response_lower or "yes" in response_lower

            return {
                "verified": verified,
                "confidence": 0.7 if verified else 0.6,
                "reason": "Parsed from LLM text response"
            }

    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return {
            "verified": True,  # Conservative default
            "confidence": 0.5,
            "reason": f"Parse error: {str(e)}"
        }


def parse_batch_verification_response(response: str, detected_piis: List[str], with_reasoning: bool = False):
    """
    Parse LLM response for batch PII verification.

    Args:
        response: Raw LLM response
        detected_piis: Original list of detected PIIs
        with_reasoning: Whether reasoning mode was used

    Returns:
        - If with_reasoning=False: List of verified PII names
        - If with_reasoning=True: List of dicts with pii, verified, reason
    """
    try:
        # Try to find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = response[start:end]
            result = json.loads(json_str)

            if with_reasoning:
                # Expecting: {"results": [{"pii": "...", "verified": true, ...}]}
                if "results" in result and isinstance(result["results"], list):
                    return result["results"]
                else:
                    # Fallback: assume all detected PIIs are verified
                    logger.warning("Unexpected reasoning format, using fallback")
                    return [
                        {
                            "pii": pii,
                            "verified": True,
                            "reason": "Parse error - assuming verified"
                        }
                        for pii in detected_piis
                    ]
            else:
                # Expecting: {"verified_piis": ["Email Address", "SSN"]}
                if "verified_piis" in result and isinstance(result["verified_piis"], list):
                    return result["verified_piis"]
                else:
                    # Fallback: return all detected PIIs
                    logger.warning("Unexpected simple format, using fallback")
                    return detected_piis

        else:
            # No JSON found, use heuristic fallback
            logger.warning("No JSON found in response, using fallback")
            if with_reasoning:
                # Return all as verified
                return [
                    {
                        "pii": pii,
                        "verified": True,
                        "reason": "No JSON found - assuming verified"
                    }
                    for pii in detected_piis
                ]
            else:
                # Return all detected PIIs
                return detected_piis

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in batch verification: {e}")
        # Return fallback
        if with_reasoning:
            return [
                {
                    "pii": pii,
                    "verified": True,
                    "reason": f"JSON parse error: {str(e)}"
                }
                for pii in detected_piis
            ]
        else:
            return detected_piis

    except Exception as e:
        logger.error(f"Error parsing batch verification response: {e}")
        # Return fallback
        if with_reasoning:
            return [
                {
                    "pii": pii,
                    "verified": True,
                    "reason": f"Parse error: {str(e)}"
                }
                for pii in detected_piis
            ]
        else:
            return detected_piis


def check_gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device(force_cpu: bool = False) -> str:
    """
    Get device for model (cuda or cpu).

    Args:
        force_cpu: Force CPU even if GPU available

    Returns:
        Device string ("cuda" or "cpu")
    """
    if force_cpu:
        return "cpu"

    if check_gpu_available():
        import torch
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    else:
        logger.info("Using CPU (no GPU available)")
        return "cpu"
