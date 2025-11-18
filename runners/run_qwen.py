"""
Qwen 2.5 PII Verifier Runner

Runs Qwen 2.5 (0.5B/1.5B/3B) for PII verification using prompting.
Models: Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct

Usage:
    python run_qwen.py --input data/test.jsonl --output results/qwen_results.jsonl --model-path models/qwen_2.5_3b
"""

import argparse
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common import (
    JSONLReader, JSONLWriter, PerformanceTracker,
    create_result_record, get_device, logger,
    load_pii_prompt_template, parse_llm_response
)


class QwenVerifier:
    """Qwen 2.5 based PII verifier using prompting."""

    MODEL_05B = "Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_15B = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_4bit: bool = False,
        use_vllm: bool = False
    ):
        """
        Initialize Qwen verifier.

        Args:
            model_path: Path to Qwen model
            device: Device to run on (cuda recommended)
            use_4bit: Whether to use 4-bit quantization
            use_vllm: Whether to use vLLM for inference
        """
        self.model_path = Path(model_path)
        self.device = device
        self.use_4bit = use_4bit
        self.use_vllm = use_vllm

        logger.info(f"Loading Qwen 2.5 model from {model_path}")
        logger.info(f"Device: {device}, 4-bit: {use_4bit}, vLLM: {use_vllm}")

        # Auto-download model if not present
        self._ensure_model_downloaded()

        if use_vllm:
            self._load_vllm_model()
        else:
            self._load_transformers_model()

        self.prompt_template = load_pii_prompt_template()

        logger.info("Qwen 2.5 model loaded successfully")

    def _ensure_model_downloaded(self):
        """Download model if not present."""
        if not self.model_path.exists() or not (self.model_path / "config.json").exists():
            logger.info(f"Model not found at {self.model_path}")

            # Determine which model to download based on path name
            path_str = str(self.model_path).lower()
            if "0.5b" in path_str or "05b" in path_str:
                model_name = self.MODEL_05B
            elif "1.5b" in path_str or "15b" in path_str:
                model_name = self.MODEL_15B
            else:
                model_name = self.MODEL_3B  # Default to 3B

            logger.info(f"Downloading {model_name} from HuggingFace...")
            logger.info("This may take several minutes on first run...")

            try:
                # Download model and tokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )

                # Save to specified path
                self.model_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(self.model_path)
                tokenizer.save_pretrained(self.model_path)

                logger.info(f"Model downloaded and saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                logger.error("Please check your internet connection and HuggingFace access")
                raise

    def _load_transformers_model(self):
        """Load model using Transformers library."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization if requested
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True  # Qwen may require this
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        self.model.eval()

    def _load_vllm_model(self):
        """Load model using vLLM for optimized inference."""
        try:
            from vllm import LLM, SamplingParams

            self.vllm_model = LLM(
                model=str(self.model_path),
                tensor_parallel_size=1,
                dtype="float16",
                trust_remote_code=True
            )

            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=200,
                stop=["}", "\n\n"]
            )

            logger.info("vLLM model loaded")

        except ImportError:
            logger.error("vLLM not installed. Install with: pip install vllm")
            raise

    def verify(self, text: str, entity_type: str, entity_value: str) -> tuple:
        """
        Verify single entity using Qwen.

        Args:
            text: Context text
            entity_type: Entity type
            entity_value: Entity value

        Returns:
            Tuple of (verified, confidence, reason)
        """
        # Create prompt
        prompt = self.prompt_template.format(
            context=text,
            entity_type=entity_type,
            entity_value=entity_value
        )

        # Format for Qwen chat template
        messages = [
            {"role": "system", "content": "You are a helpful PII verification assistant."},
            {"role": "user", "content": prompt}
        ]

        # Generate response
        if self.use_vllm:
            # vLLM handles chat format internally
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            outputs = self.vllm_model.generate([formatted_prompt], self.sampling_params)
            response = outputs[0].outputs[0].text

        else:
            # Transformers
            text_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

        # Parse response
        result = parse_llm_response(response)
        return result["verified"], result["confidence"], result["reason"]

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
        if self.use_vllm:
            # vLLM can process batches efficiently
            prompts = []
            for text, entity_type, entity_value in zip(texts, entity_types, entity_values):
                prompt_content = self.prompt_template.format(
                    context=text,
                    entity_type=entity_type,
                    entity_value=entity_value
                )

                messages = [
                    {"role": "system", "content": "You are a helpful PII verification assistant."},
                    {"role": "user", "content": prompt_content}
                ]

                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                prompts.append(formatted_prompt)

            outputs = self.vllm_model.generate(prompts, self.sampling_params)

            results = []
            for output in outputs:
                response = output.outputs[0].text
                result = parse_llm_response(response)
                results.append((result["verified"], result["confidence"], result["reason"]))

            return results

        else:
            # Process sequentially with Transformers
            results = []
            for text, entity_type, entity_value in zip(texts, entity_types, entity_values):
                result = self.verify(text, entity_type, entity_value)
                results.append(result)

            return results


def main():
    parser = argparse.ArgumentParser(description="Run Qwen 2.5 PII Verifier")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model-path", required=True, help="Path to Qwen model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (smaller for LLMs)")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for faster inference")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device")

    args = parser.parse_args()

    # Initialize components
    reader = JSONLReader(args.input)
    tracker = PerformanceTracker()
    verifier = QwenVerifier(
        model_path=args.model_path,
        device=args.device,
        use_4bit=args.use_4bit,
        use_vllm=args.use_vllm
    )

    # Determine model name
    if "0.5B" in str(args.model_path) or "0.5b" in str(args.model_path):
        model_name = "qwen_2.5_0.5b"
        model_params = "0.5B"
    elif "1.5B" in str(args.model_path) or "1.5b" in str(args.model_path):
        model_name = "qwen_2.5_1.5b"
        model_params = "1.5B"
    else:
        model_name = "qwen_2.5_3b"
        model_params = "3B"

    if args.use_4bit:
        model_name += "_4bit"
    if args.use_vllm:
        model_name += "_vllm"

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
                        model_type="gpu",
                        model_params=model_params
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
                    model_type="gpu",
                    model_params=model_params
                )
                writer.write(result_record)
                tracker.record(per_item_latency, confidence, verified)

    tracker.end()
    tracker.print_summary(model_name)

    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
