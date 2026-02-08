"""KernelBench-backed evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from kernelbench import eval as kb_eval
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code


@dataclass
class EvalSummary:
    compiled: bool
    correctness: bool
    runtime_us: float
    ref_runtime_us: float
    speedup_vs_ref: float
    metadata: dict


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


def extract_kernel_code(raw_text: str) -> str:
    code = extract_first_code(raw_text, ["python", "cpp"])
    if not code:
        raise RuntimeError("Model output did not contain a code block.")
    return code


def static_check(kernel_code: str, backend: str = "cuda", precision: str = "fp32") -> tuple[bool, str | None, list[str]]:
    ok, err, warnings = validate_kernel_static(kernel_code, backend=backend, precision=precision)
    return ok, err, warnings


def evaluate_kernel(
    ref_arch_src: str,
    kernel_code: str,
    device: str,
    num_correct_trials: int,
    num_perf_trials: int,
    backend: str,
    precision: str,
) -> EvalSummary:
    result = kb_eval.eval_kernel_against_ref(
        original_model_src=ref_arch_src,
        custom_model_src=kernel_code,
        measure_performance=True,
        timing_method="cuda_event",
        verbose=False,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        device=torch.device(device),
        backend=backend,
        precision=kb_eval.get_torch_dtype_from_string(precision),
    )

    speedup = -1.0
    if result.runtime > 0 and result.ref_runtime > 0:
        speedup = float(result.ref_runtime / result.runtime)

    return EvalSummary(
        compiled=bool(result.compiled),
        correctness=bool(result.correctness),
        runtime_us=float(result.runtime),
        ref_runtime_us=float(result.ref_runtime),
        speedup_vs_ref=speedup,
        metadata=dict(result.metadata or {}),
    )


def to_dict(dataclass_obj):
    return asdict(dataclass_obj)
