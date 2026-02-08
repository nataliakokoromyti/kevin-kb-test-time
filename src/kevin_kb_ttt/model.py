"""Remote model client for Modal-hosted Kevin inference."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence


@dataclass
class GenMetrics:
    ttft_s: float
    total_s: float
    tokens_generated: int
    decode_tok_s: float


class KevinModalOpenAI:
    """Client for an OpenAI-compatible Modal endpoint.

    Expected endpoint: {base_url}/chat/completions
    """

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        max_parallel_requests: int = 8,
    ) -> None:
        self.model_id = model_id
        self.base_url = (base_url or os.getenv("MODAL_LLM_BASE_URL", "")).strip().rstrip("/")
        if not self.base_url:
            raise RuntimeError(
                "Modal inference requires MODAL_LLM_BASE_URL (or --modal-llm-base-url) "
                "pointing to an OpenAI-compatible endpoint."
            )
        self.api_key = (api_key or os.getenv("MODAL_LLM_API_KEY", "")).strip() or None
        self.timeout_s = timeout_s
        self.max_parallel_requests = max(1, int(max_parallel_requests))

    def _post_chat(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"Modal LLM HTTP {e.code}: {e.reason}. Body: {err_body[:800]}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to call Modal LLM endpoint: {e}") from e

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from Modal LLM endpoint: {raw[:800]}") from e

    def generate(self, prompt: str, max_new_tokens: int = 16384, temperature: float = 0.9) -> tuple[str, GenMetrics]:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stream": False,
        }

        t0 = time.perf_counter()
        out = self._post_chat(payload)
        t1 = time.perf_counter()
        total = max(t1 - t0, 1e-9)

        try:
            text = out["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected Modal LLM response shape: {out}") from e

        usage = out.get("usage", {})
        tokens_generated = int(usage.get("completion_tokens") or 0)
        if tokens_generated <= 0:
            tokens_generated = max(1, len(text.split()))

        metrics = GenMetrics(
            ttft_s=-1.0,
            total_s=total,
            tokens_generated=tokens_generated,
            decode_tok_s=(tokens_generated / total) if total > 0 else 0.0,
        )
        return text, metrics

    def generate_many(
        self,
        prompts: list[str],
        max_new_tokens: int = 16384,
        temperature: float = 0.9,
        temperatures: Sequence[float] | None = None,
    ) -> list[tuple[str, GenMetrics]]:
        if not prompts:
            return []
        if temperatures is not None and len(temperatures) != len(prompts):
            raise ValueError("temperatures length must match prompts length")
        if temperatures is None:
            temperatures = [temperature] * len(prompts)
        workers = min(self.max_parallel_requests, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self.generate, prompts[i], max_new_tokens, float(temperatures[i]))
                for i in range(len(prompts))
            ]
            return [f.result() for f in futures]


def create_model(
    *,
    model_backend: str,
    model_id: str,
    modal_llm_base_url: str | None,
    modal_llm_api_key: str | None,
    modal_llm_timeout_s: float,
    modal_llm_max_parallel_requests: int,
):
    if model_backend != "modal_openai":
        raise ValueError(
            f"Unknown model backend: {model_backend}. "
            "This repo is configured for Modal-only inference; use --model-backend modal_openai."
        )
    return KevinModalOpenAI(
        model_id=model_id,
        base_url=modal_llm_base_url,
        api_key=modal_llm_api_key,
        timeout_s=modal_llm_timeout_s,
        max_parallel_requests=modal_llm_max_parallel_requests,
    )
