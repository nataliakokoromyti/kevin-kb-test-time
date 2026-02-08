"""KernelBench-backed prompting, parsing, and exact script-harness evaluation helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code


THINKING_PATTERN = re.compile(
    r"<(?:think|thinking|THOUGHT)>(.*?)</(?:think|thinking|THOUGHT)>",
    re.DOTALL | re.IGNORECASE,
)
KERNEL_BLOCK_PATTERN = re.compile(
    r"<KERNEL>\s*```(?:cuda|python|cpp)?\s*\n?(.*?)```\s*</KERNEL>",
    re.DOTALL | re.IGNORECASE,
)
KERNEL_BLOCK_SIMPLE_PATTERN = re.compile(
    r"<KERNEL>(.*?)</KERNEL>",
    re.DOTALL | re.IGNORECASE,
)
SUMMARY_BLOCK_PATTERN = re.compile(
    r"<SUMMARY>(.*?)</SUMMARY>",
    re.DOTALL | re.IGNORECASE,
)

MAX_ERROR_LEN = 800
MAX_ERROR_LINES = 10


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _kb_repo_root() -> Path:
    kb = _repo_root() / "third_party" / "KernelBench"
    if not kb.exists():
        raise RuntimeError("KernelBench submodule not found at third_party/KernelBench. Run: git submodule update --init --recursive")
    return kb


def _parse_bool(text: str, key: str) -> bool | None:
    m = re.search(rf"\b{re.escape(key)}=(True|False)\b", text)
    if not m:
        return None
    return m.group(1) == "True"


def _parse_float_line(text: str, prefix: str) -> float | None:
    for line in text.splitlines():
        if prefix in line:
            m = re.search(r"(-?\d+(?:\.\d+)?)", line)
            if m:
                return float(m.group(1))
    return None


def _parse_speedup_line(text: str, prefix: str) -> float | None:
    for line in text.splitlines():
        if prefix in line:
            m = re.search(r"(-?\d+(?:\.\d+)?)x", line)
            if m:
                return float(m.group(1))
    return None


def _extract_key_error_from_text(output: str) -> str | None:
    patterns = [
        r"(\w+Error: .+?)(?:\n\n|\n(?=[A-Z])|$)",
        r"(\w+Exception: .+?)(?:\n\n|\n(?=[A-Z])|$)",
        r"(CUDA error: .+?)(?:\n|$)",
        r"(RuntimeError: .+?)(?:\n|$)",
        r"(TypeError: .+?)(?:\n|$)",
        r"(ValueError: .+?)(?:\n|$)",
        r"(AssertionError: .+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            text = re.sub(r"\n\s*\n", "\n", match.group(1).strip())
            return text[:MAX_ERROR_LEN]

    lines = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Traceback"):
            continue
        if line.startswith("File \"") and ", line " in line:
            continue
        if line.startswith("During handling of"):
            continue
        lines.append(line)
        if len(lines) >= MAX_ERROR_LINES:
            break
    if lines:
        return "\n".join(lines)[:MAX_ERROR_LEN]
    return None


@dataclass
class ParsedResponse:
    thought: str
    thought_summary: str
    kernel: str
    raw: str
    format_ok: bool


def parse_structured_response(text: str) -> ParsedResponse:
    raw = text
    thought = ""
    thought_summary = ""
    kernel = ""

    think_match = THINKING_PATTERN.search(text)
    if think_match:
        thought = think_match.group(1).strip()
        text = THINKING_PATTERN.sub("", text).strip()

    summary_match = SUMMARY_BLOCK_PATTERN.search(text)
    if summary_match:
        thought_summary = summary_match.group(1).strip()
        text = SUMMARY_BLOCK_PATTERN.sub("", text).strip()

    kernel_match = KERNEL_BLOCK_PATTERN.search(text)
    if kernel_match:
        kernel = kernel_match.group(1).strip()
    else:
        kernel_match = KERNEL_BLOCK_SIMPLE_PATTERN.search(text)
        if kernel_match:
            inner = kernel_match.group(1).strip()
            kernel = extract_first_code(inner, ["python", "cpp"]) or inner

    if not kernel:
        kernel = extract_first_code(text, ["python", "cpp"]) or ""

    format_ok = bool(kernel) and ("class ModelNew" in kernel or "def forward" in kernel)
    return ParsedResponse(
        thought=thought,
        thought_summary=thought_summary,
        kernel=kernel,
        raw=raw,
        format_ok=format_ok,
    )


def extract_kernel_code(raw_text: str, require_structured: bool = False) -> str:
    parsed = parse_structured_response(raw_text)
    if require_structured and not parsed.format_ok:
        raise RuntimeError("Model output did not match <KERNEL> structured format or valid kernel code.")
    if parsed.kernel:
        return parsed.kernel
    raise RuntimeError("Model output did not contain a code block.")


def check_for_cheating(kernel_code: str) -> bool:
    pytorch_patterns = [
        "torch.nn.functional",
        "torch.nn.",
        "F.conv",
        "F.linear",
        "F.relu",
        "F.gelu",
        "F.softmax",
        "F.batch_norm",
        "F.layer_norm",
        "F.dropout",
        "nn.functional.",
    ]
    if any(p in kernel_code for p in pytorch_patterns):
        return True

    if "try:" in kernel_code or "except:" in kernel_code or "except " in kernel_code:
        return True

    if re.search(r"\bpass\b", kernel_code):
        return True

    has_custom = any(
        p in kernel_code
        for p in ["@triton.jit", "@triton.autotune", "load_inline", "cpp_extension", "cute::", "from cutlass", "@T.prim_func", "tvm.build"]
    )
    if not has_custom:
        for op in ["torch.mm", "torch.bmm", "torch.matmul", "torch.conv", "torch.einsum"]:
            if op in kernel_code:
                return True

    return False


def extract_key_error(error_message: str | None) -> str:
    if not error_message:
        return ""
    text = _extract_key_error_from_text(error_message)
    return text or ""


def categorize_error(eval_result: dict[str, Any]) -> str:
    if not eval_result.get("format_ok", False):
        return "format_error"
    if not eval_result.get("compiled", False):
        return "compilation_error"
    if not eval_result.get("correctness", False):
        err = eval_result.get("error_message", "") or ""
        if "Error" in err or "Exception" in err:
            return "runtime_error"
        return "correctness_error"
    speed = eval_result.get("speedup_vs_ref", None)
    if speed is not None and speed >= 0 and speed < 1.0:
        return "performance_warning"
    return "success"


def get_error_guidance(error_category: str, backend: str) -> str:
    guidance = {
        "format_error": "Ensure your output is valid `ModelNew` code in <KERNEL>...</KERNEL>.",
        "compilation_error": f"Fix {backend.upper()} syntax/API issues and imports.",
        "runtime_error": "Fix runtime issues: shape mismatches, bad indexing, invalid launch config.",
        "correctness_error": "Kernel runs but output is wrong. Revisit algorithm and boundary handling.",
        "performance_warning": "Kernel is correct but slow. Optimize memory access and parallelization.",
    }
    return guidance.get(error_category, "Fix issues from the latest evaluator feedback.")


def get_problem(level: int, problem_id: int, dataset_source: str, dataset_name: str, base_path: str | None) -> tuple[str, str]:
    ds_kwargs = {
        "level": level,
        "source": dataset_source,
        "dataset_name": dataset_name,
    }
    if dataset_source == "local" and base_path:
        ds_kwargs["base_path"] = base_path
    ds = construct_kernelbench_dataset(**ds_kwargs)
    p = ds.get_problem_by_id(problem_id)
    return p.name, p.code


def build_prompt(ref_arch_src: str, backend: str = "cuda", prompt_option: str = "one_shot", precision: str = "fp32") -> str:
    return get_prompt_for_backend(ref_arch_src, backend=backend, option=prompt_option, precision=precision)


def static_check(kernel_code: str, backend: str = "cuda", precision: str = "fp32") -> tuple[bool, str | None, list[str]]:
    ok, err, warnings = validate_kernel_static(kernel_code, backend=backend, precision=precision)
    return ok, err, warnings


_EVAL_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _cache_key(level: int, problem_id: int, backend: str, kernel_code: str, modal_gpu: str) -> str:
    h = hashlib.sha1(kernel_code.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{level}:{problem_id}:{backend}:{modal_gpu}:{h}"


def _cache_prune(maxsize: int = 512) -> None:
    while len(_EVAL_CACHE) > maxsize:
        _EVAL_CACHE.popitem(last=False)


def _run_kb_modal_harness(
    *,
    level: int,
    problem_id: int,
    kernel_code: str,
    dataset_source: str,
    dataset_name: str,
    backend: str,
    precision: str,
    num_correct_trials: int,
    num_perf_trials: int,
    modal_gpu: str,
    timeout_s: float,
) -> dict[str, Any]:
    kb_root = _kb_repo_root()
    script = kb_root / "scripts" / "run_and_check.py"
    if not script.exists():
        raise RuntimeError("KernelBench harness script not found: third_party/KernelBench/scripts/run_and_check.py")

    with tempfile.NamedTemporaryFile("w", suffix="_kernel.py", delete=False, encoding="utf-8") as tmp:
        tmp.write(kernel_code)
        kernel_path = tmp.name

    cmd = [
        "python",
        str(script),
        "ref_origin=kernelbench",
        f"dataset_src={dataset_source}",
        f"dataset_name={dataset_name}",
        f"level={level}",
        f"problem_id={problem_id}",
        f"kernel_src_path={kernel_path}",
        "eval_mode=modal",
        f"gpu={modal_gpu}",
        f"num_correct_trials={num_correct_trials}",
        f"num_perf_trials={num_perf_trials}",
        f"backend={backend}",
        f"precision={precision}",
        "check_kernel=true",
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(kb_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    finally:
        try:
            os.remove(kernel_path)
        except OSError:
            pass

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    compiled = _parse_bool(output, "compiled")
    correctness = _parse_bool(output, "correctness")
    runtime_us = _parse_float_line(output, "Custom Kernel exec time")
    ref_runtime_us = _parse_float_line(output, "PyTorch Reference Eager exec time")
    speedup = _parse_speedup_line(output, "Speedup over eager")

    if runtime_us is None:
        runtime_us = -1.0
    if ref_runtime_us is None:
        ref_runtime_us = -1.0
    if speedup is None and runtime_us > 0 and ref_runtime_us > 0:
        speedup = ref_runtime_us / runtime_us
    if speedup is None:
        speedup = -1.0

    err = None
    if proc.returncode != 0:
        err = _extract_key_error_from_text(output) or f"KB harness exited with code {proc.returncode}"

    return {
        "format_ok": True,
        "compiled": bool(compiled) if compiled is not None else False,
        "correctness": bool(correctness) if correctness is not None else False,
        "tests_passed": 0,
        "tests_total": num_correct_trials,
        "speedup_vs_ref": float(speedup),
        "runtime_us": float(runtime_us),
        "ref_runtime_us": float(ref_runtime_us),
        "cheated": False,
        "error_message": err,
        "code_length": len(kernel_code),
        "metadata": {
            "harness": "KernelBench/scripts/run_and_check.py",
            "modal_gpu": modal_gpu,
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
            "timings": {"total_eval_s": time.perf_counter() - t0},
        },
    }


async def evaluate_kernel_async(
    *,
    level: int,
    problem_id: int,
    ref_arch_src: str,
    kernel_code: str,
    dataset_source: str,
    dataset_name: str,
    device: str,
    num_correct_trials: int,
    num_perf_trials: int,
    backend: str,
    precision: str,
    timeout_s: float = 120.0,
    cache_results: bool = True,
    use_modal: bool = True,
    modal_gpu: str = "L40S",
    modal_timeout_s: float = 120.0,
) -> dict[str, Any]:
    del ref_arch_src, device

    if not use_modal:
        return {
            "format_ok": True,
            "compiled": False,
            "correctness": False,
            "tests_passed": 0,
            "tests_total": num_correct_trials,
            "speedup_vs_ref": -1.0,
            "runtime_us": -1.0,
            "ref_runtime_us": -1.0,
            "cheated": False,
            "error_message": "Local eval disabled. Use Modal eval only.",
            "code_length": len(kernel_code),
            "metadata": {"timings": {"total_eval_s": 0.0}},
        }

    t_start = time.perf_counter()
    if check_for_cheating(kernel_code):
        return {
            "format_ok": True,
            "compiled": False,
            "correctness": False,
            "tests_passed": 0,
            "tests_total": num_correct_trials,
            "speedup_vs_ref": -1.0,
            "runtime_us": -1.0,
            "ref_runtime_us": -1.0,
            "cheated": True,
            "error_message": "Kernel detected as cheating pattern",
            "code_length": len(kernel_code),
            "metadata": {"timings": {"total_eval_s": time.perf_counter() - t_start}},
        }

    key = _cache_key(level, problem_id, backend, kernel_code, modal_gpu)
    if cache_results and key in _EVAL_CACHE:
        cached = dict(_EVAL_CACHE[key])
        meta = dict(cached.get("metadata", {}))
        meta["cache_hit"] = True
        meta.setdefault("timings", {})["total_eval_s"] = time.perf_counter() - t_start
        cached["metadata"] = meta
        _EVAL_CACHE.move_to_end(key)
        return cached

    fut = asyncio.to_thread(
        _run_kb_modal_harness,
        level=level,
        problem_id=problem_id,
        kernel_code=kernel_code,
        dataset_source=dataset_source,
        dataset_name=dataset_name,
        backend=backend,
        precision=precision,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        modal_gpu=modal_gpu,
        timeout_s=modal_timeout_s,
    )
    try:
        payload = await asyncio.wait_for(fut, timeout=timeout_s)
    except asyncio.TimeoutError:
        payload = {
            "format_ok": True,
            "compiled": False,
            "correctness": False,
            "tests_passed": 0,
            "tests_total": num_correct_trials,
            "speedup_vs_ref": -1.0,
            "runtime_us": -1.0,
            "ref_runtime_us": -1.0,
            "cheated": False,
            "error_message": f"Evaluation timed out after {timeout_s}s",
            "code_length": len(kernel_code),
            "metadata": {"timings": {"total_eval_s": time.perf_counter() - t_start}},
        }

    if cache_results:
        _EVAL_CACHE[key] = dict(payload)
        _cache_prune()

    return payload
