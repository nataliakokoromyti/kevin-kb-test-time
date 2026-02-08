from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from dataclasses import asdict
from datetime import datetime

from .kb_pipeline import (
    build_prompt,
    categorize_error,
    evaluate_kernel_async,
    extract_key_error,
    get_error_guidance,
    get_problem,
    parse_structured_response,
)
from .model import create_model

MAX_KERNEL_HISTORY_LEN = 2000
MAX_SUMMARY_CHARS = 800
KEVIN_CORRECTNESS_BONUS = 0.3


STRUCTURED_SYSTEM_WITH_THINK = """You are an expert GPU kernel developer. Optimize the given PyTorch reference with a custom {backend} kernel.

Return:
1. A complete Python code block containing `ModelNew`.
2. A short plain-text summary (2-4 sentences) of key changes and next fixes.

You may think step-by-step naturally before the final answer.
Do not use custom wrapper tags like <KERNEL> or <SUMMARY>.
"""

STRUCTURED_SYSTEM_NO_THINK = """You are an expert GPU kernel developer. Optimize the given PyTorch reference with a custom {backend} kernel.

Return:
1. A complete Python code block containing `ModelNew`.
2. A short plain-text summary (2-4 sentences) of key changes and next fixes.

Do not use custom wrapper tags like <KERNEL> or <SUMMARY>.
"""

REFINEMENT_TEMPLATE = """
## Prior Attempts
{history_block}

## Instructions
Use the full trajectory above. Keep what worked and avoid repeating failed patterns.
Focus on fixing the latest failure while preserving prior gains.
{latest_guidance}
"""

ATTEMPT_BLOCK_TEMPLATE = """
### Attempt {turn}
- Status: {error_category}
- Compiled: {compiled}
- Tests Passed: {tests_passed}/{tests_total}
{speedup_line}

Summary:
{summary}

Kernel:
```python
{kernel}
```

{error_section}
"""


def _truncate_kernel(kernel_code: str, max_len: int = MAX_KERNEL_HISTORY_LEN) -> str:
    if len(kernel_code) <= max_len:
        return kernel_code
    return kernel_code[:max_len] + "\n# ... (truncated)"


def _truncate_summary(summary: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rstrip() + "..."


def _fallback_summary(eval_result: dict, error_category: str) -> str:
    if error_category == "success":
        s = eval_result.get("speedup_vs_ref")
        if isinstance(s, (int, float)) and s >= 0:
            return f"Previous attempt compiled and passed checks with speedup {s:.2f}x."
        return "Previous attempt compiled and passed checks."
    if not eval_result.get("compiled"):
        return "Previous attempt failed to compile; focus on syntax/API fixes."
    if not eval_result.get("correctness"):
        return (
            f"Previous attempt compiled but passed {eval_result.get('tests_passed', 0)}/"
            f"{eval_result.get('tests_total', 0)} checks; fix correctness."
        )
    return "Previous attempt needs refinement based on feedback."


def _error_category_display(error_category: str) -> str:
    labels = {
        "format_error": "FORMAT ERROR",
        "compilation_error": "COMPILATION ERROR",
        "runtime_error": "RUNTIME ERROR",
        "correctness_error": "CORRECTNESS ERROR",
        "performance_warning": "CORRECT BUT SLOW",
        "success": "SUCCESS",
    }
    return labels.get(error_category, error_category.upper())


def _kevin_score(eval_result: dict) -> float:
    """Kevin-style step score: S = 0.3 * correct + speedup * correct."""
    correct = bool(eval_result.get("correctness", False))
    if not correct:
        return 0.0
    speedup = float(eval_result.get("speedup_vs_ref", 0.0))
    return KEVIN_CORRECTNESS_BONUS + speedup


def _beam_temperatures(
    count: int,
    base: float,
    jitter: float,
    min_temp: float,
    max_temp: float,
) -> list[float]:
    """Create a deterministic spread of temperatures for beam diversity."""
    if count <= 1 or jitter <= 0:
        return [max(min(base, max_temp), min_temp)] * max(count, 1)
    temps: list[float] = []
    for i in range(count):
        frac = -1.0 + 2.0 * i / (count - 1)
        t = base + frac * jitter
        t = max(min(t, max_temp), min_temp)
        temps.append(t)
    return temps


def _build_initial_prompt(base_prompt: str, backend: str, include_think: bool) -> str:
    system = STRUCTURED_SYSTEM_WITH_THINK if include_think else STRUCTURED_SYSTEM_NO_THINK
    return system.format(backend=backend.upper()) + "\n\n" + base_prompt


def _build_refinement_prompt(
    base_prompt: str,
    backend: str,
    include_think: bool,
    history: list[dict],
) -> str:
    if not history:
        return _build_initial_prompt(base_prompt, backend, include_think)

    blocks: list[str] = []
    latest_error_category = "success"
    for idx, entry in enumerate(history, start=1):
        eval_result = entry.get("eval", {})
        error_category = categorize_error(eval_result)
        latest_error_category = error_category
        speedup = eval_result.get("speedup_vs_ref")
        speedup_line = ""
        if isinstance(speedup, (int, float)) and speedup >= 0:
            speedup_line = f"- Speedup: {speedup:.2f}x"

        err_msg = eval_result.get("error_message")
        err_text = extract_key_error(err_msg)
        error_section = ""
        if err_text and error_category != "success":
            error_section = f"Error Details:\n```\n{err_text}\n```"

        summary = entry.get("summary") or _fallback_summary(eval_result, error_category)
        summary = _truncate_summary(summary)

        blocks.append(
            ATTEMPT_BLOCK_TEMPLATE.format(
                turn=idx,
                error_category=_error_category_display(error_category),
                compiled="Yes" if eval_result.get("compiled") else "No",
                tests_passed=eval_result.get("tests_passed", 0),
                tests_total=eval_result.get("tests_total", 0),
                speedup_line=speedup_line,
                summary=summary,
                kernel=_truncate_kernel(entry.get("kernel", "")),
                error_section=error_section,
            )
        )

    refinement = REFINEMENT_TEMPLATE.format(
        history_block="\n".join(blocks),
        latest_guidance=get_error_guidance(latest_error_category, backend),
    )

    system = STRUCTURED_SYSTEM_WITH_THINK if include_think else STRUCTURED_SYSTEM_NO_THINK
    return system.format(backend=backend.upper()) + "\n\n" + base_prompt + "\n\n" + refinement


def _make_failed_eval(format_ok: bool, error_message: str, code_length: int) -> dict:
    return {
        "format_ok": format_ok,
        "compiled": False,
        "correctness": False,
        "tests_passed": 0,
        "tests_total": 0,
        "speedup_vs_ref": -1.0,
        "runtime_us": -1.0,
        "ref_runtime_us": -1.0,
        "cheated": False,
        "error_message": error_message,
        "code_length": code_length,
        "metadata": {},
    }


def _evaluate_generated_response(args: argparse.Namespace, ref_arch_src: str, raw_text: str, genm: object) -> dict:
    parsed = parse_structured_response(raw_text)
    kernel = parsed.kernel or ""

    if not parsed.format_ok:
        return {
            "generation": asdict(genm),
            "raw": raw_text,
            "thought": parsed.thought,
            "summary": parsed.thought_summary,
            "kernel": kernel,
            "static_check_ok": False,
            "static_error": "format_error",
            "static_warnings": [],
            "eval": _make_failed_eval(False, "Could not extract valid kernel code block", len(kernel)),
        }

    eval_result = asyncio.run(
        evaluate_kernel_async(
            level=args.level,
            problem_id=args.problem_id,
            ref_arch_src=ref_arch_src,
            kernel_code=kernel,
            dataset_source=args.dataset_source,
            dataset_name=args.dataset_name,
            device=args.device,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            backend=args.backend,
            precision=args.precision,
            timeout_s=args.eval_timeout_s,
            cache_results=args.cache_results,
            use_modal=True,
            modal_gpu=args.modal_gpu,
            modal_timeout_s=args.modal_timeout_s,
        )
    )

    return {
        "generation": asdict(genm),
        "raw": raw_text,
        "thought": parsed.thought,
        "summary": parsed.thought_summary,
        "kernel": kernel,
        "static_check_ok": None,
        "static_warnings": [],
        "eval": eval_result,
    }


def _run_beam_trajectory(
    args: argparse.Namespace,
    llm,
    ref_arch_src: str,
    base_prompt: str,
    history: list[dict],
    steps: int,
    round_idx: int,
    trajectory_id: int,
    generation_temperature: float,
    attempts: list[dict],
) -> dict:
    local_history = copy.deepcopy(history)
    best_kernel = None
    best_speedup = 0.0
    best_score = 0.0

    for h in local_history:
        ev = h.get("eval", {})
        if ev.get("correctness"):
            s = float(ev.get("speedup_vs_ref", 0.0))
            score = _kevin_score(ev)
            if s > best_speedup:
                best_speedup = s
                best_kernel = h.get("kernel")
            if score > best_score:
                best_score = score

    for step_idx in range(steps):
        if local_history:
            prompt = _build_refinement_prompt(
                base_prompt=base_prompt,
                backend=args.backend,
                include_think=args.include_think,
                history=local_history,
            )
        else:
            prompt = _build_initial_prompt(base_prompt, args.backend, args.include_think)

        raw, genm = llm.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=generation_temperature)
        attempt = _evaluate_generated_response(args, ref_arch_src, raw, genm)
        attempt.update(
            {
                "round": round_idx,
                "trajectory_id": trajectory_id,
                "trajectory_step": step_idx + 1,
                "generation_temperature": generation_temperature,
            }
        )
        attempts.append(attempt)

        carry_summary = attempt.get("summary")
        if not carry_summary:
            carry_summary = _fallback_summary(attempt["eval"], categorize_error(attempt["eval"]))

        local_history.append(
            {
                "kernel": attempt.get("kernel", ""),
                "summary": _truncate_summary(carry_summary),
                "eval": attempt.get("eval", {}),
            }
        )

        ev = attempt.get("eval", {})
        if ev.get("correctness"):
            s = float(ev.get("speedup_vs_ref", 0.0))
            score = _kevin_score(ev)
            if s > best_speedup:
                best_speedup = s
                best_kernel = attempt.get("kernel")
            if score > best_score:
                best_score = score

    return {
        "best_kernel": best_kernel,
        "best_speedup": best_speedup,
        "best_score": best_score,
        "history": local_history,
    }


def run(args: argparse.Namespace) -> dict:
    problem_name, ref_arch_src = get_problem(
        level=args.level,
        problem_id=args.problem_id,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        base_path=args.kb_base_path,
    )

    base_prompt = build_prompt(
        ref_arch_src,
        backend=args.backend,
        prompt_option=args.prompt_option,
        precision=args.precision,
    )

    llm = create_model(
        model_backend=args.model_backend,
        model_id=args.model_id,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
    )
    attempts: list[dict] = []

    if args.technique == "greedy":
        prompt = _build_initial_prompt(base_prompt, args.backend, args.include_think)
        raw, genm = llm.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        attempts.append(_evaluate_generated_response(args, ref_arch_src, raw, genm))

    elif args.technique == "best_of_n":
        prompt = _build_initial_prompt(base_prompt, args.backend, args.include_think)
        prompt_batch = [prompt for _ in range(args.n_samples)]
        generations = llm.generate_many(prompt_batch, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        for raw, genm in generations:
            attempts.append(_evaluate_generated_response(args, ref_arch_src, raw, genm))

    elif args.technique == "serial_refine":
        history: list[dict] = []
        for turn_idx in range(args.turns):
            if history:
                prompt = _build_refinement_prompt(
                    base_prompt=base_prompt,
                    backend=args.backend,
                    include_think=args.include_think,
                    history=history,
                )
            else:
                prompt = _build_initial_prompt(base_prompt, args.backend, args.include_think)

            raw, genm = llm.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            attempt = _evaluate_generated_response(args, ref_arch_src, raw, genm)
            attempts.append(attempt)

            carry_summary = attempt.get("summary")
            if not carry_summary:
                carry_summary = _fallback_summary(attempt["eval"], categorize_error(attempt["eval"]))

            history.append(
                {
                    "kernel": attempt.get("kernel", ""),
                    "summary": _truncate_summary(carry_summary),
                    "eval": attempt.get("eval", {}),
                }
            )

            if args.early_stop_on_correct and attempt.get("eval", {}).get("correctness"):
                speed = float(attempt["eval"].get("speedup_vs_ref", 0.0))
                if args.speedup_threshold is None or speed >= args.speedup_threshold:
                    break

    elif args.technique == "beam_search":
        if args.beam_width > args.num_beams:
            raise ValueError("beam_width must be <= num_beams")
        if args.num_beams % args.beam_width != 0:
            raise ValueError("num_beams must be divisible by beam_width")

        trajectories = []
        round1_temps = _beam_temperatures(
            count=args.num_beams,
            base=args.beam_temperature,
            jitter=args.beam_temp_jitter,
            min_temp=args.beam_min_temperature,
            max_temp=args.beam_max_temperature,
        )
        for beam_idx in range(args.num_beams):
            trajectories.append(
                _run_beam_trajectory(
                    args=args,
                    llm=llm,
                    ref_arch_src=ref_arch_src,
                    base_prompt=base_prompt,
                    history=[],
                    steps=args.steps_per_round,
                    round_idx=1,
                    trajectory_id=beam_idx,
                    generation_temperature=round1_temps[beam_idx],
                    attempts=attempts,
                )
            )

        for round_idx in range(2, args.num_rounds + 1):
            trajectories.sort(key=lambda t: t["best_score"], reverse=True)
            survivors = trajectories[: args.beam_width]
            expanded = []
            clones_per_survivor = args.num_beams // args.beam_width
            trajectory_id = 0
            for survivor_rank, survivor in enumerate(survivors):
                clone_temps = _beam_temperatures(
                    count=clones_per_survivor,
                    base=args.beam_temperature,
                    jitter=args.beam_temp_jitter,
                    min_temp=args.beam_min_temperature,
                    max_temp=args.beam_max_temperature,
                )
                # Slight rank-conditioned offset: top survivors get slightly more exploitative temps.
                rank_offset = min(0.05 * survivor_rank, 0.15)
                for clone_idx in range(clones_per_survivor):
                    t = max(
                        min(clone_temps[clone_idx] - rank_offset, args.beam_max_temperature),
                        args.beam_min_temperature,
                    )
                    expanded.append(
                        _run_beam_trajectory(
                            args=args,
                            llm=llm,
                            ref_arch_src=ref_arch_src,
                            base_prompt=base_prompt,
                            history=survivor["history"],
                            steps=args.steps_per_round,
                            round_idx=round_idx,
                            trajectory_id=trajectory_id,
                            generation_temperature=t,
                            attempts=attempts,
                        )
                    )
                    trajectory_id += 1
            trajectories = expanded
    else:
        raise ValueError(f"Unknown technique: {args.technique}")

    best = None
    for a in attempts:
        ev = a.get("eval", {})
        if not ev.get("correctness"):
            continue
        if best is None or ev.get("speedup_vs_ref", -1.0) > best["eval"].get("speedup_vs_ref", -1.0):
            best = a

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_id": args.model_id,
        "technique": args.technique,
        "problem": {"level": args.level, "problem_id": args.problem_id, "name": problem_name},
        "settings": {
            "dataset_source": args.dataset_source,
            "dataset_name": args.dataset_name,
            "kb_base_path": args.kb_base_path,
            "device": args.device,
            "num_correct_trials": args.num_correct_trials,
            "num_perf_trials": args.num_perf_trials,
            "precision": args.precision,
            "backend": args.backend,
            "prompt_option": args.prompt_option,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "turns": args.turns,
            "num_beams": args.num_beams,
            "beam_width": args.beam_width,
            "steps_per_round": args.steps_per_round,
            "num_rounds": args.num_rounds,
            "beam_temperature": args.beam_temperature,
            "beam_temp_jitter": args.beam_temp_jitter,
            "beam_min_temperature": args.beam_min_temperature,
            "beam_max_temperature": args.beam_max_temperature,
            "include_think": args.include_think,
            "eval_timeout_s": args.eval_timeout_s,
            "cache_results": args.cache_results,
            "use_modal": True,
            "modal_gpu": args.modal_gpu,
            "modal_timeout_s": args.modal_timeout_s,
            "model_backend": args.model_backend,
            "load_in_4bit": args.load_in_4bit,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_max_model_len": args.vllm_max_model_len,
            "early_stop_on_correct": args.early_stop_on_correct,
            "speedup_threshold": args.speedup_threshold,
        },
        "attempts": attempts,
        "best": best,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="cognition-ai/Kevin-32B")
    p.add_argument("--model-backend", default="hf", choices=["hf", "vllm"])
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--vllm-max-model-len", type=int, default=32768)
    p.add_argument("--technique", default="greedy", choices=["greedy", "best_of_n", "serial_refine", "beam_search"])
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--num-beams", type=int, default=16)
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--steps-per-round", type=int, default=4)
    p.add_argument("--num-rounds", type=int, default=2)
    p.add_argument("--beam-temperature", type=float, default=0.9)
    p.add_argument("--beam-temp-jitter", type=float, default=0.2)
    p.add_argument("--beam-min-temperature", type=float, default=0.6)
    p.add_argument("--beam-max-temperature", type=float, default=1.2)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--problem-id", type=int, required=True)
    p.add_argument("--dataset-source", default="huggingface", choices=["local", "huggingface"])
    p.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    p.add_argument("--kb-base-path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-correct-trials", type=int, default=5)
    p.add_argument("--num-perf-trials", type=int, default=100)
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--backend", default="cuda")
    p.add_argument("--prompt-option", default="one_shot", choices=["zero_shot", "one_shot", "few_shot"])
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--include-think", action="store_true", default=False)
    p.add_argument("--eval-timeout-s", type=float, default=120.0)
    p.add_argument("--cache-results", action="store_true", default=True)
    p.add_argument("--no-cache-results", dest="cache_results", action="store_false")
    p.add_argument("--modal-gpu", default="L40S")
    p.add_argument("--modal-timeout-s", type=float, default=120.0)
    p.add_argument("--early-stop-on-correct", action="store_true", default=False)
    p.add_argument("--speedup-threshold", type=float, default=None)
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(
        args.results_dir,
        f"{args.technique}_l{args.level}_p{args.problem_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote result to: {out}")


if __name__ == "__main__":
    main()
