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
    def __init__(
        self,
        model_id: str,
        torch_dtype: str = "auto",
        load_in_4bit: bool = True,
    ) -> None:
        if torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                raise RuntimeError(
                    "load_in_4bit=True requires bitsandbytes + compatible CUDA environment. "
                    "Install bitsandbytes or disable 4-bit mode with --no-load-in-4bit."
                ) from e
        else:
            load_kwargs["dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    def generate(self, prompt: str, max_new_tokens: int = 16384, temperature: float = 0.9) -> tuple[str, GenMetrics]:
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

        # Decode only generated continuation, not the prompt.
        generated = out[0][prompt_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        metrics = GenMetrics(
            ttft_s=-1.0,
            total_s=total,
            tokens_generated=max(new_tokens, 0),
            decode_tok_s=(new_tokens / total) if total > 0 and new_tokens > 0 else 0.0,
        )
        return text, metrics

    def generate_many(self, prompts: list[str], max_new_tokens: int = 16384, temperature: float = 0.9) -> list[tuple[str, GenMetrics]]:
        results = []
        for p in prompts:
            results.append(self.generate(p, max_new_tokens=max_new_tokens, temperature=temperature))
        return results


class KevinVLLM:
    def __init__(
        self,
        model_id: str,
        torch_dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 32768,
    ) -> None:
        try:
            from vllm import LLM
        except Exception as e:
            raise RuntimeError(
                "vLLM backend requires `vllm` installed in this environment."
            ) from e

        dtype = "auto"
        if torch_dtype == "float16":
            dtype = "float16"
        elif torch_dtype == "bfloat16":
            dtype = "bfloat16"

        self._llm = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    def generate(self, prompt: str, max_new_tokens: int = 16384, temperature: float = 0.9) -> tuple[str, GenMetrics]:
        return self.generate_many([prompt], max_new_tokens=max_new_tokens, temperature=temperature)[0]

    def generate_many(self, prompts: list[str], max_new_tokens: int = 16384, temperature: float = 0.9) -> list[tuple[str, GenMetrics]]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            n=1,
        )
        t0 = time.perf_counter()
        outputs = self._llm.generate(prompts, params)
        t1 = time.perf_counter()

        total = max(t1 - t0, 1e-9)
        per_sample_time = total / max(len(outputs), 1)

        results: list[tuple[str, GenMetrics]] = []
        for out in outputs:
            cand = out.outputs[0]
            tok_n = len(cand.token_ids)
            metrics = GenMetrics(
                ttft_s=-1.0,
                total_s=per_sample_time,
                tokens_generated=tok_n,
                decode_tok_s=(tok_n / per_sample_time) if per_sample_time > 0 and tok_n > 0 else 0.0,
            )
            results.append((cand.text, metrics))
        return results


def create_model(
    *,
    model_backend: str,
    model_id: str,
    torch_dtype: str,
    load_in_4bit: bool,
    vllm_tensor_parallel_size: int,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: int,
):
    if model_backend == "vllm":
        return KevinVLLM(
            model_id=model_id,
            torch_dtype=torch_dtype,
            tensor_parallel_size=vllm_tensor_parallel_size,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
        )
    if model_backend == "hf":
        return KevinHF(
            model_id=model_id,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
        )
    raise ValueError(f"Unknown model backend: {model_backend}")
