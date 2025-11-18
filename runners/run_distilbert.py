"""
DistilBERT PII Verifier Runner

Runs DistilBERT fine-tuned for PII detection on test data.
Model: ai4privacy/distilbert_finetuned_ai4privacy_v2 (66M parameters)

Usage:
    python run_distilbert.py --input data/test.jsonl --output results/distilbert_results.jsonl
"""

import argparse
import time
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from common import (
    JSONLReader, JSONLWriter, PerformanceTracker,
    create_result_record, batch_generator, get_device, logger
)


class DistilBertVerifier:
    """DistilBERT-based PII verifier."""

    def __init__(self, model_path: str, device: str = "cpu", use_onnx: bool = False):
        """
        Initialize DistilBERT verifier.

        Args:
            model_path: Path to model directory
            device: Device to run on (cpu/cuda)
            use_onnx: Whether to use ONNX optimized model
        """
        self.model_path = Path(model_path)
        self.device = device
        self.use_onnx = use_onnx

        logger.info(f"Loading DistilBERT model from {model_path}")
        logger.info(f"Device: {device}, ONNX: {use_onnx}")

        if use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

        logger.info("DistilBERT model loaded successfully")

    def _load_pytorch_model(self):
        """Load PyTorch model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_onnx_model(self):
        """Load ONNX optimized model."""
        try:
            import onnxruntime as ort

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Look for ONNX model file
            onnx_path = self.model_path / "model.onnx"
            if not onnx_path.exists():
                logger.warning("ONNX model not found, falling back to PyTorch")
                self._load_pytorch_model()
                self.use_onnx = False
                return

            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            logger.info("ONNX model loaded")

        except ImportError:
            logger.warning("onnxruntime not installed, falling back to PyTorch")
            self._load_pytorch_model()
            self.use_onnx = False

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
        start_time = time.time()

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt" if not self.use_onnx else "np"
        )

        # Run inference
        if self.use_onnx:
            outputs = self.ort_session.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
            )
            logits = outputs[0]
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()

        # Parse results
        import numpy as np
        results = []

        for i, (logit, entity_type, entity_value) in enumerate(zip(logits, entity_types, entity_values)):
            # Apply softmax to get probabilities
            probs = np.exp(logit) / np.sum(np.exp(logit))

            # Class 1 = PII, Class 0 = Not PII
            confidence = float(probs[1])
            verified = confidence >= 0.5

            # Generate reason based on confidence
            if verified:
                if confidence > 0.9:
                    reason = f"High confidence PII detection ({entity_type})"
                elif confidence > 0.7:
                    reason = f"Likely PII ({entity_type})"
                else:
                    reason = f"Possible PII ({entity_type})"
            else:
                if confidence < 0.3:
                    reason = f"Low confidence, likely false positive ({entity_type})"
                else:
                    reason = f"Uncertain, may be false positive ({entity_type})"

            results.append((verified, confidence, reason))

        return results

    def verify(self, text: str, entity_type: str, entity_value: str) -> tuple:
        """
        Verify single entity.

        Args:
            text: Context text
            entity_type: Entity type
            entity_value: Entity value

        Returns:
            Tuple of (verified, confidence, reason)
        """
        results = self.verify_batch([text], [entity_type], [entity_value])
        return results[0]


def main():
    parser = argparse.ArgumentParser(description="Run DistilBERT PII Verifier")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model-path", required=True, help="Path to DistilBERT model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--use-onnx", action="store_true", help="Use ONNX optimized model")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device (auto-detect if not set)")

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = get_device()

    # Initialize components
    reader = JSONLReader(args.input)
    tracker = PerformanceTracker()
    verifier = DistilBertVerifier(
        model_path=args.model_path,
        device=args.device,
        use_onnx=args.use_onnx
    )

    model_name = "distilbert_ai4privacy_v2"
    if args.use_onnx:
        model_name += "_onnx"

    logger.info(f"Processing {reader.count()} records")
    logger.info(f"Batch size: {args.batch_size}")

    # Process in batches
    tracker.start()

    with JSONLWriter(args.output) as writer:
        batch = []
        batch_records = []

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
                        model_params="66M"
                    )
                    writer.write(result_record)
                    tracker.record(per_item_latency, confidence, verified)

                # Clear batch
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
                    model_params="66M"
                )
                writer.write(result_record)
                tracker.record(per_item_latency, confidence, verified)

    tracker.end()
    tracker.print_summary(model_name)

    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
