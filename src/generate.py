"""Batched greedy generation backends."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Generator:
    """Load and use a mock, vLLM, or Transformers generation backend."""

    def __init__(
        self,
        model_name: str,
        backend: str = "auto",
        dtype: str = "bfloat16",
        quantization_4bit: bool = False,
        max_new_tokens: int = 32,
        mock_llm: bool = False,
        mock_prediction: str = "mock answer",
    ) -> None:
        self.model_name = model_name
        self.backend = "mock" if mock_llm else backend
        self.max_new_tokens = max_new_tokens
        self.mock_prediction = mock_prediction
        self.model: Any = None
        self.tokenizer: Any = None
        self.torch: Any = None
        if self.backend == "mock":
            return
        if self.backend not in ("auto", "vllm", "transformers"):
            raise ValueError(f"Unsupported generation backend: {self.backend}")
        if self.backend in ("auto", "vllm"):
            try:
                from vllm import LLM, SamplingParams
            except ImportError:
                if self.backend == "vllm":
                    raise RuntimeError("backend=vllm requested but vLLM is not installed.")
                logger.info("vLLM unavailable; falling back to Transformers.")
            else:
                import torch

                self.torch = torch
                self.model = LLM(
                    model=model_name,
                    dtype=dtype,
                    quantization="bitsandbytes" if quantization_4bit else None,
                )
                self.tokenizer = self.model.get_tokenizer()
                self.SamplingParams = SamplingParams
                self.backend = "vllm"
                return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs: dict[str, Any] = {
            "torch_dtype": getattr(torch, dtype),
            "device_map": "auto",
        }
        if quantization_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.backend = "transformers"

    def format_chat(self, prompt: str) -> str:
        """Apply the model's user-turn chat template when available."""
        if self.backend == "mock":
            return prompt
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def token_count(self, prompt: str) -> int:
        """Count tokens in a formatted prompt."""
        if self.backend == "mock":
            return len(prompt.split())
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def generated_token_count(self, text: str) -> int:
        """Count generated tokens for throughput logging."""
        return self.token_count(text)

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one greedy prediction for each formatted prompt."""
        if not prompts:
            return []
        if self.backend == "mock":
            return [self.mock_prediction for _ in prompts]
        if self.backend == "vllm":
            params = self.SamplingParams(
                temperature=0.0, max_tokens=self.max_new_tokens
            )
            outputs = self.model.generate(prompts, params)
            return [output.outputs[0].text.strip() for output in outputs]

        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return [
            self.tokenizer.decode(row[input_length:], skip_special_tokens=True).strip()
            for row in generated
        ]

    def gpu_memory_bytes(self) -> int | None:
        """Return current allocated CUDA memory when torch exposes it."""
        if self.backend == "mock" or self.torch is None or not self.torch.cuda.is_available():
            return None
        return int(self.torch.cuda.memory_allocated())
