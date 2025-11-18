"""
PHI BERT Medical Entity Verifier Runner

Runs PHI-BERT on medical/PHI entity test data.
Model: obi/deid_bert_i2b2 (110M parameters)

Usage:
    python run_phibert.py --input data/test_medical.jsonl --output results/phibert_results.jsonl
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from common import (
    JSONLReader, JSONLWriter, PerformanceTracker,
    create_result_record, get_device, logger
)


class PhiBertVerifier:
    """PHI-BERT based medical entity verifier."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize PHI-BERT verifier.

        Args:
            model_path: Path to PHI-BERT model
            device: Device to run on (cpu/cuda)
        """
        self.model_path = Path(model_path)
        self.device = device

        logger.info(f"Loading PHI-BERT model from {model_path}")
        logger.info(f"Device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Get label mapping
        self.id2label = self.model.config.id2label

        logger.info("PHI-BERT model loaded successfully")
        logger.info(f"Supported labels: {list(self.id2label.values())}")

    def verify(self, text: str, entity_type: str, entity_value: str) -> tuple:
        """
        Verify if detected medical entity is valid PHI.

        Args:
            text: Context text
            entity_type: Entity type
            entity_value: Entity value

        Returns:
            Tuple of (verified, confidence, reason)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0]
            probabilities = torch.softmax(outputs.logits, dim=2)[0]

        # Find entity_value in text
        entity_start = text.lower().find(entity_value.lower())

        if entity_start == -1:
            return False, 0.5, "Entity value not found in context"

        entity_end = entity_start + len(entity_value)

        # Find corresponding tokens
        detected_labels = []
        detected_confidences = []

        for i, (pred_id, probs) in enumerate(zip(predictions, probabilities)):
            token_start, token_end = offset_mapping[i]

            # Check if token overlaps with entity
            if token_start < entity_end and token_end > entity_start:
                label = self.id2label[pred_id.item()]
                confidence = probs[pred_id].item()

                if label != "O":  # Not "Outside" label
                    detected_labels.append(label)
                    detected_confidences.append(confidence)

        # Determine if entity is verified
        if detected_labels:
            # Entity was detected by PHI-BERT
            avg_confidence = np.mean(detected_confidences)
            verified = True
            most_common_label = max(set(detected_labels), key=detected_labels.count)
            reason = f"Detected as {most_common_label} with confidence {avg_confidence:.2f}"
        else:
            # Entity not detected as PHI
            verified = False
            avg_confidence = 0.6
            reason = "Not detected as PHI by model"

        return verified, avg_confidence, reason

    def verify_batch(self, texts: list, entity_types: list, entity_values: list) -> list:
        """
        Verify batch of entities (processes sequentially for now).

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
    parser = argparse.ArgumentParser(description="Run PHI-BERT Medical Entity Verifier")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model-path", required=True, help="Path to PHI-BERT model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device (auto-detect if not set)")

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = get_device()

    # Initialize components
    reader = JSONLReader(args.input)
    tracker = PerformanceTracker()
    verifier = PhiBertVerifier(
        model_path=args.model_path,
        device=args.device
    )

    model_name = "phi_bert_i2b2"

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
                        model_params="110M"
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
                    model_params="110M"
                )
                writer.write(result_record)
                tracker.record(per_item_latency, confidence, verified)

    tracker.end()
    tracker.print_summary(model_name)

    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
