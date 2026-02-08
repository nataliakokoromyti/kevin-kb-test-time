"""Model loading and generation utilities for Kevin-32B."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenMetrics:
    ttft_s: float
    total_s: float
    tokens_generated: int
    decode_tok_s: float


class KevinHF:
    def __init__(self, model_id: str, torch_dtype: str = "auto") -> None:
        if torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 768, temperature: float = 0.0) -> tuple[str, GenMetrics]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = int(out.shape[1] - prompt_len)
        total = t1 - t0

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        metrics = GenMetrics(
            ttft_s=-1.0,
            total_s=total,
            tokens_generated=max(new_tokens, 0),
            decode_tok_s=(new_tokens / total) if total > 0 and new_tokens > 0 else 0.0,
        )
        return text, metrics
