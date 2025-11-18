"""
GLiNER Financial Entity Verifier Runner

Runs GLiNER zero-shot NER on financial entity test data.
Model: urchade/gliner_base (400M parameters)

Usage:
    python run_gliner.py --input data/test_financial.jsonl --output results/gliner_results.jsonl
"""

import argparse
import time
import torch
from pathlib import Path
from common import (
    JSONLReader, JSONLWriter, PerformanceTracker,
    create_result_record, get_device, logger
)


class GlinerVerifier:
    """GLiNER-based financial entity verifier."""

    MODEL_NAME = "urchade/gliner_base"

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize GLiNER verifier.

        Args:
            model_path: Path to GLiNER model
            device: Device to run on (cpu/cuda)
        """
        self.model_path = Path(model_path)
        self.device = device

        logger.info(f"Loading GLiNER model from {model_path}")
        logger.info(f"Device: {device}")

        try:
            from gliner import GLiNER

            # Auto-download model if not present
            self._ensure_model_downloaded(GLiNER)

            self.model = GLiNER.from_pretrained(str(self.model_path))
            self.model.to(device)
            logger.info("GLiNER model loaded successfully")
        except ImportError:
            logger.error("gliner package not installed. Install with: pip install gliner")
            raise

    def _ensure_model_downloaded(self, GLiNER):
        """Download model if not present."""
        if not self.model_path.exists() or not list(self.model_path.glob("*.safetensors")):
            logger.info(f"Model not found at {self.model_path}")
            logger.info(f"Downloading {self.MODEL_NAME} from HuggingFace...")
            logger.info("This may take a few minutes on first run...")

            try:
                # Download model
                model = GLiNER.from_pretrained(self.MODEL_NAME)

                # Save to specified path
                self.model_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(self.model_path))

                logger.info(f"Model downloaded and saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                logger.error("Please check your internet connection and HuggingFace access")
                raise

        # Financial entity types GLiNER supports
        self.entity_types = [
            "cryptocurrency address", "bitcoin address", "ethereum address",
            "IBAN", "SWIFT code", "BIC code", "routing number",
            "bank account", "credit card", "debit card"
        ]

    def verify(self, text: str, entity_type: str, entity_value: str) -> tuple:
        """
        Verify if detected financial entity is valid.

        Args:
            text: Context text
            entity_type: Entity type
            entity_value: Entity value

        Returns:
            Tuple of (verified, confidence, reason)
        """
        # Map entity types to GLiNER labels
        type_mapping = {
            "CRYPTO": "cryptocurrency address",
            "BITCOIN": "bitcoin address",
            "ETHEREUM": "ethereum address",
            "IBAN": "IBAN",
            "SWIFT_CODE": "SWIFT code",
            "SWIFT": "SWIFT code",
            "BIC": "BIC code",
            "ROUTING_NUMBER": "routing number",
            "BANK_ACCOUNT": "bank account",
            "CREDIT_CARD": "credit card"
        }

        gliner_label = type_mapping.get(entity_type.upper(), entity_type.lower())

        # Run GLiNER zero-shot NER
        entities = self.model.predict_entities(
            text,
            [gliner_label],
            threshold=0.5
        )

        # Check if entity value was detected
        verified = False
        confidence = 0.5
        reason = "Not detected by GLiNER"

        for entity in entities:
            entity_text = entity["text"]
            entity_score = entity["score"]

            # Check if detected entity matches our value
            if entity_value.lower() in entity_text.lower() or entity_text.lower() in entity_value.lower():
                verified = True
                confidence = float(entity_score)
                reason = f"Detected by GLiNER with confidence {confidence:.2f}"
                break

        # If not detected, check with lower threshold
        if not verified:
            entities_low = self.model.predict_entities(
                text,
                [gliner_label],
                threshold=0.3
            )

            for entity in entities_low:
                entity_text = entity["text"]
                entity_score = entity["score"]

                if entity_value.lower() in entity_text.lower() or entity_text.lower() in entity_value.lower():
                    verified = True
                    confidence = float(entity_score)
                    reason = f"Detected with low confidence {confidence:.2f}"
                    break

        return verified, confidence, reason

    def verify_batch(self, texts: list, entity_types: list, entity_values: list) -> list:
        """
        Verify batch of entities.

        Args:
            texts: List of context texts
            entity_types: List of entity types
            entity_values: List of entity values

        Returns:
            List of (verified, confidence, reason) tuples
        """
        results = []

        for text, entity_type, entity_value in zip(texts, entity_types, entity_values):
            result = self.verify(text, entity_type, entity_value)
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description="Run GLiNER Financial Entity Verifier")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model-path", required=True, help="Path to GLiNER model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (GLiNER processes sequentially)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device (auto-detect if not set)")

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = get_device()

    # Initialize components
    reader = JSONLReader(args.input)
    tracker = PerformanceTracker()
    verifier = GlinerVerifier(
        model_path=args.model_path,
        device=args.device
    )

    model_name = "gliner_base"

    logger.info(f"Processing {reader.count()} records")
    logger.info(f"Batch size: {args.batch_size}")

    # Process records
    tracker.start()

    with JSONLWriter(args.output) as writer:
        batch = []

        for record in reader.read():
            batch.append(record)

            if len(batch) >= args.batch_size:
                # Process batch
                texts = [r["input"] for r in batch]
                entity_types = [r.get("entityType", "UNKNOWN") for r in batch]
                entity_values = [r.get("entityValue", "") for r in batch]

                # Time batch inference
                batch_start = time.time()
                results = verifier.verify_batch(texts, entity_types, entity_values)
                batch_time = (time.time() - batch_start) * 1000
                per_item_latency = batch_time / len(batch)

                # Write results
                for record, (verified, confidence, reason) in zip(batch, results):
                    result_record = create_result_record(
                        input_record=record,
                        verified=verified,
                        confidence=confidence,
                        reason=reason,
                        latency_ms=per_item_latency,
                        model_name=model_name,
                        model_type="cpu",
                        model_params="400M"
                    )
                    writer.write(result_record)
                    tracker.record(per_item_latency, confidence, verified)

                batch = []

        # Process remaining items
        if batch:
            texts = [r["input"] for r in batch]
            entity_types = [r.get("entityType", "UNKNOWN") for r in batch]
            entity_values = [r.get("entityValue", "") for r in batch]

            batch_start = time.time()
            results = verifier.verify_batch(texts, entity_types, entity_values)
            batch_time = (time.time() - batch_start) * 1000
            per_item_latency = batch_time / len(batch)

            for record, (verified, confidence, reason) in zip(batch, results):
                result_record = create_result_record(
                    input_record=record,
                    verified=verified,
                    confidence=confidence,
                    reason=reason,
                    latency_ms=per_item_latency,
                    model_name=model_name,
                    model_type="cpu",
                    model_params="400M"
                )
                writer.write(result_record)
                tracker.record(per_item_latency, confidence, verified)

    tracker.end()
    tracker.print_summary(model_name)

    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
