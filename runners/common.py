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

    def __init__(self, file_path: str, overwrite: bool = True):
        """
        Initialize JSONL writer.

        Args:
            file_path: Path to output JSONL file
            overwrite: If True, overwrite existing file. If False, append to existing file.
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in write mode (overwrite) or append mode
        mode = 'w' if overwrite else 'a'
        self.file = open(self.file_path, mode, encoding='utf-8')

        if overwrite:
            logger.info(f"Writing results to: {self.file_path} (overwrite mode)")
        else:
            logger.info(f"Appending results to: {self.file_path}")

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


def load_data_element_descriptions() -> Dict[str, str]:
    """
    Load data element descriptions from default_data_elements.json.

    Returns:
        Dictionary mapping element names to descriptions
    """
    try:
        current_dir = Path(__file__).parent.parent
        json_path = current_dir / "default_data_elements.json"

        if not json_path.exists():
            logger.warning(f"default_data_elements.json not found at {json_path}")
            return {}

        with open(json_path, 'r') as f:
            elements = json.load(f)

        # Create mapping: element name -> description
        descriptions = {}
        for element in elements:
            name = element.get("name", "")
            description = element.get("description", "")
            if name and description:
                descriptions[name] = description

        logger.info(f"Loaded {len(descriptions)} data element descriptions")
        return descriptions

    except Exception as e:
        logger.error(f"Error loading data element descriptions: {e}")
        return {}


def format_element_descriptions(detected_piis: List[str], all_descriptions: Dict[str, str]) -> str:
    """
    Format element descriptions for detected PIIs.

    Args:
        detected_piis: List of detected PII names
        all_descriptions: All available descriptions

    Returns:
        Formatted string with descriptions for detected PIIs
    """
    if not detected_piis:
        return "No PIIs detected."

    formatted = []
    for pii in detected_piis:
        description = all_descriptions.get(pii, "No description available.")
        formatted.append(f"- **{pii}**: {description}")

    return "\n".join(formatted)


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
      "reason": "Email username contains personal name pattern indicating individual ownership"
    }},
    {{
      "pii": "Phone Number",
      "verified": false,
      "reason": "Toll-free prefix indicates organization-level contact, not personal phone"
    }}
  ]
}}"""
        response_instruction = """CRITICAL JSON OUTPUT REQUIREMENTS:
1. Output ONLY a valid JSON object - nothing else
2. Do NOT wrap JSON in markdown code blocks (no ```)
3. Do NOT add any explanatory text before or after the JSON
4. Start your response with { and end with }
5. Ensure all strings are properly escaped (use \\" for quotes inside strings)
6. Match all braces and brackets: {{ }} [ ]
7. Use double quotes for all keys and string values
8. No trailing commas after last items
9. Keep reason strings under 100 characters
10. Boolean values must be lowercase: true, false (not True, False)

Your ENTIRE response must be this JSON object:"""
    else:
        output_format = """Email Address, Social Security Number

(List only the verified PII types, comma-separated. If NONE are verified, respond: NONE)"""
        response_instruction = """CRITICAL OUTPUT RULES:
1. Output ONLY the comma-separated list of verified PII types - nothing else
2. Do NOT add any explanatory text, commentary, or conversation
3. Do NOT say "I apologize" or "Let me analyze" or any other text
4. If you detect PIIs: output ONLY their names separated by commas
5. If no PIIs are verified: output ONLY the word "NONE"
6. Do NOT add any text before or after your answer

Respond with comma-separated list:"""

    return f"""You are a specialized PII Verification AI. A preliminary system has already flagged potential Personally Identifiable Information (PII) in some text. Your task is to analyze each flagged element and determine if it is genuine personal PII or a false positive.

Your primary goal is to protect privacy, so it is safer to be overcautious than to miss personal data.

Here is the original text containing potential PII:
<input_text>
{{input_text}}
</input_text>

Here are the specific strings that were flagged as potential PII:
<detected_piis>
{{detected_piis}}
</detected_piis>

Here are descriptions of what each PII element type means:
<element_descriptions>
{{element_descriptions}}
</element_descriptions>

For each detected element, you must apply these three tests. An element is Personal PII if it relates to a specific, identifiable person:

**OWNERSHIP Test**: Does this belong to a PERSON or an ORGANIZATION/SYSTEM?
- Person: Data contains individual names, personal usernames, or is inherently tied to one person (e.g., a user's personal score, individual employee ID)
- Organization/System: Data contains functional roles (support, admin), shared contacts, or system identifiers (ticket-id, ref_no)

**SPECIFICITY Test**: Does this single out ONE SPECIFIC PERSON?
- Specific: It's a unique personal identifier (personal email, individual employee ID in personal context) or private contact method (home address, personal mobile number)
- Not Specific: It's a shared identifier (team alias, department phone) or belongs to a business entity (info@company.com, business address)

**CONTEXT Test**: How is the data USED in the text?
- Personal Use: Appears in user profiles, customer records, employee files, or used to identify an individual
- Organizational/System Use: Used as company name, generic business contact, system configuration value, or non-personal technical ID

For each element in the detected_piis list, you must:

1. **Apply all three tests** (Ownership, Specificity, Context) to evaluate the element
2. **Provide reasoning** that explains your analysis based on the tests
3. **Make a decision**: VERIFIED (it is personal PII) or REJECTED (it is a false positive)

Important: If any test suggests the element could be personal PII, or if you're uncertain, choose VERIFIED to err on the side of privacy protection.

Analyze each element thoroughly and remember that protecting privacy is the priority.

{response_instruction}
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
        if with_reasoning:
            # Reasoning mode: Expect JSON format
            response_clean = response.strip()

            # Remove markdown code blocks if present
            if response_clean.startswith('```'):
                # Remove opening ```json or ```
                lines = response_clean.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove closing ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                response_clean = '\n'.join(lines).strip()

            # Handle case where LLM returns ONLY markdown markers with no content
            if not response_clean or response_clean == '```':
                logger.error("LLM returned empty or invalid response (only markdown markers)")
                logger.error("This usually means the model is confused by the prompt format")
                logger.error("Falling back to rejecting all PIIs for safety")
                return [
                    {
                        "pii": pii,
                        "verified": False,
                        "reason": "LLM failed to provide valid JSON response"
                    }
                    for pii in detected_piis
                ]

            # Find JSON boundaries
            start = response_clean.find('{')
            end = response_clean.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = response_clean[start:end]

                # Fix common JSON issues before parsing
                # Replace Python-style booleans with JSON booleans
                json_str = json_str.replace(': True', ': true').replace(': False', ': false')

                result = json.loads(json_str)

                # Expecting: {"results": [{"pii": "...", "verified": true, "reason": "..."}]}
                if "results" in result and isinstance(result["results"], list):
                    return result["results"]
                else:
                    # Fallback: reject all for safety (better than false positives)
                    logger.warning("Unexpected reasoning format (missing 'results' key)")
                    logger.warning(f"Got keys: {list(result.keys())}")
                    return [
                        {
                            "pii": pii,
                            "verified": False,
                            "reason": "Parse error - unexpected JSON structure"
                        }
                        for pii in detected_piis
                    ]
            else:
                # No JSON found - reject all for safety
                logger.warning("No JSON found in reasoning response")
                logger.warning(f"Response was: {response_clean[:200]}")
                return [
                    {
                        "pii": pii,
                        "verified": False,
                        "reason": "No valid JSON in response"
                    }
                    for pii in detected_piis
                ]

        else:
            # Simple mode: Expect comma-separated format
            response_clean = response.strip()

            # Remove common garbage prefixes that models sometimes add
            garbage_prefixes = [
                "none your response",
                "i apologize",
                "let me analyze",
                "here is",
                "the verified",
                "answer:",
                "response:",
            ]
            response_lower = response_clean.lower()
            for prefix in garbage_prefixes:
                if response_lower.startswith(prefix):
                    # Try to find where the actual comma-separated list starts
                    # Look for first occurrence of a known PII type or "NONE"
                    for detected_pii in detected_piis:
                        idx = response_clean.find(detected_pii)
                        if idx > 0:
                            response_clean = response_clean[idx:]
                            logger.warning(f"Removed garbage prefix, extracted: {response_clean[:100]}")
                            break
                    break

            # Check for explicit NONE response (check if NONE appears in first 20 chars)
            if "NONE" in response_clean[:20].upper():
                logger.info("LLM returned NONE - no PIIs verified")
                return []

            # Try comma-separated parsing first (new format)
            # Split by comma and clean each item
            parts = [part.strip() for part in response_clean.split(',')]
            parts = [part for part in parts if part and part.upper() != "NONE"]

            # Filter out garbage - only keep parts that could be PII names
            # (contain only letters, spaces, and basic punctuation)
            valid_parts = []
            for part in parts:
                # Remove any text after newlines
                if '\n' in part:
                    part = part.split('\n')[0].strip()

                # Check if this looks like a PII name (not a sentence)
                if len(part) < 100 and not any(bad in part.lower() for bad in ['apologize', 'analyze', 'response', 'superf00k']):
                    valid_parts.append(part)

            parts = valid_parts

            if parts:
                # Fuzzy match parsed PIIs against detected PIIs
                verified = []
                for parsed_pii in parts:
                    parsed_lower = parsed_pii.lower()
                    best_match = None

                    # Try to find best match in detected PIIs
                    for detected_pii in detected_piis:
                        detected_lower = detected_pii.lower()

                        # Exact match
                        if parsed_lower == detected_lower:
                            best_match = detected_pii
                            break

                        # Substring match (either direction)
                        if parsed_lower in detected_lower or detected_lower in parsed_lower:
                            best_match = detected_pii
                            break

                        # Partial word match (e.g., "Email" matches "Email Address")
                        parsed_words = set(parsed_lower.split())
                        detected_words = set(detected_lower.split())
                        if parsed_words & detected_words:  # Intersection
                            best_match = detected_pii

                    if best_match and best_match not in verified:
                        verified.append(best_match)
                    elif not best_match:
                        logger.warning(f"Could not match parsed PII '{parsed_pii}' to detected PIIs")

                logger.info(f"Comma-separated parsing: {len(verified)} PIIs verified")
                return verified

            # Fallback: Try JSON format (backward compatibility)
            start = response.find('{')
            end = response.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)

                if "verified_piis" in result and isinstance(result["verified_piis"], list):
                    logger.info("Fell back to JSON parsing successfully")
                    return result["verified_piis"]

            # Final fallback: return all detected PIIs (conservative)
            logger.warning("Could not parse response, using conservative fallback (verify all)")
            return detected_piis

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in batch verification: {e}")
        logger.error(f"Problematic response: {response[:500]}")  # Log first 500 chars
        # Conservative fallback
        if with_reasoning:
            return [
                {
                    "pii": pii,
                    "verified": True,
                    "reason": "JSON parsing failed - conservatively marking as verified (privacy-first approach)"
                }
                for pii in detected_piis
            ]
        else:
            return detected_piis

    except Exception as e:
        logger.error(f"Error parsing batch verification response: {e}")
        # Conservative fallback
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
