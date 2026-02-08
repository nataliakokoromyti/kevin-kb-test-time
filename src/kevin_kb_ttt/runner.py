from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
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

MAX_KERNEL_HISTORY_LEN = 8000
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
    # Preserve both setup/signature and tail-end kernel details.
    head_len = int(max_len * 0.6)
    tail_len = max_len - head_len
    head = kernel_code[:head_len]
    tail = kernel_code[-tail_len:]
    return head + "\n# ... (middle truncated) ...\n" + tail


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


def _kernel_fingerprint(kernel_code: str) -> str:
    # Normalize whitespace to detect semantically identical repeats with minor formatting changes.
    normalized = " ".join((kernel_code or "").split())
    return hashlib.sha1(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()


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


def _make_duplicate_attempt(genm: object, raw_text: str, parsed, kernel: str) -> dict:
    return {
        "generation": asdict(genm),
        "raw": raw_text,
        "thought": parsed.thought,
        "summary": parsed.thought_summary,
        "kernel": kernel,
        "static_check_ok": None,
        "static_warnings": [],
        "duplicate_kernel": True,
        "eval": {
            "format_ok": True,
            "compiled": False,
            "correctness": False,
            "tests_passed": 0,
            "tests_total": 0,
            "speedup_vs_ref": -1.0,
            "runtime_us": -1.0,
            "ref_runtime_us": -1.0,
            "cheated": False,
            "error_message": "Duplicate kernel detected: identical to a previous attempt. Try a different approach.",
            "code_length": len(kernel),
            "metadata": {"duplicate_kernel": True},
        },
    }


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


def _prepare_generated_attempt(
    args: argparse.Namespace,
    raw_text: str,
    genm: object,
    seen_kernel_fingerprints: set[str] | None = None,
) -> tuple[dict, str | None]:
    parsed = parse_structured_response(raw_text)
    kernel = parsed.kernel or ""
    kernel_fp = _kernel_fingerprint(kernel)

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
            "duplicate_kernel": False,
            "kernel_fingerprint": kernel_fp,
            "eval": _make_failed_eval(False, "Could not extract valid kernel code block", len(kernel)),
        }, None

    if seen_kernel_fingerprints is not None and kernel_fp in seen_kernel_fingerprints:
        attempt = _make_duplicate_attempt(genm, raw_text, parsed, kernel)
        attempt["kernel_fingerprint"] = kernel_fp
        return attempt, None

    attempt = {
        "generation": asdict(genm),
        "raw": raw_text,
        "thought": parsed.thought,
        "summary": parsed.thought_summary,
        "kernel": kernel,
        "static_check_ok": None,
        "static_warnings": [],
        "duplicate_kernel": False,
        "kernel_fingerprint": kernel_fp,
    }
    return attempt, kernel


async def _evaluate_kernels_batch_async(args: argparse.Namespace, ref_arch_src: str, kernels: list[str]) -> list[dict]:
    coros = [
        evaluate_kernel_async(
            level=args.level,
            problem_id=args.problem_id,
            ref_arch_src=ref_arch_src,
            kernel_code=kernel_code,
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
        for kernel_code in kernels
    ]
    return await asyncio.gather(*coros)


async def _evaluate_generated_response_async(
    args: argparse.Namespace,
    ref_arch_src: str,
    raw_text: str,
    genm: object,
    seen_kernel_fingerprints: set[str] | None = None,
) -> dict:
    attempt, kernel_for_eval = _prepare_generated_attempt(args, raw_text, genm, seen_kernel_fingerprints)
    if kernel_for_eval is None:
        return attempt
    attempt["eval"] = (await _evaluate_kernels_batch_async(args, ref_arch_src, [kernel_for_eval]))[0]
    return attempt


def _apply_attempt_to_history(history: list[dict], attempt: dict) -> None:
    carry_summary = attempt.get("summary")
    if not carry_summary:
        carry_summary = _fallback_summary(attempt["eval"], categorize_error(attempt["eval"]))

    history.append(
        {
            "attempt_id": attempt.get("attempt_id"),
            "kernel": attempt.get("kernel", ""),
            "summary": _truncate_summary(carry_summary),
            "eval": attempt.get("eval", {}),
            "kernel_fingerprint": attempt.get("kernel_fingerprint"),
        }
    )


def _parent_attempt_id_from_history(history: list[dict]) -> int | None:
    if not history:
        return None
    parent_id = history[-1].get("attempt_id")
    if isinstance(parent_id, int):
        return parent_id
    return None


def _compute_best_attempt(attempts: list[dict]) -> dict | None:
    best = None
    for a in attempts:
        ev = a.get("eval", {})
        if not ev.get("correctness"):
            continue
        if best is None or ev.get("speedup_vs_ref", -1.0) > best["eval"].get("speedup_vs_ref", -1.0):
            best = a
    return best


def _build_search_graph(attempts: list[dict]) -> tuple[list[dict], list[dict]]:
    graph_nodes = []
    graph_edges = []
    for a in attempts:
        attempt_id = a.get("attempt_id")
        parent_attempt_id = a.get("parent_attempt_id")
        ev = a.get("eval", {})
        if isinstance(attempt_id, int):
            graph_nodes.append(
                {
                    "attempt_id": attempt_id,
                    "parent_attempt_id": parent_attempt_id,
                    "trajectory_id": a.get("trajectory_id"),
                    "parent_trajectory_id": a.get("parent_trajectory_id"),
                    "round": a.get("round"),
                    "trajectory_step": a.get("trajectory_step"),
                    "correctness": bool(ev.get("correctness", False)),
                    "speedup_vs_ref": float(ev.get("speedup_vs_ref", -1.0)),
                    "score": _kevin_score(ev),
                    "duplicate_kernel": bool(a.get("duplicate_kernel", False)),
                }
            )
        if isinstance(attempt_id, int) and isinstance(parent_attempt_id, int):
            graph_edges.append({"from_attempt_id": parent_attempt_id, "to_attempt_id": attempt_id})
    return graph_nodes, graph_edges


def _make_result_payload(
    *,
    args: argparse.Namespace,
    problem_name: str,
    attempts: list[dict],
    settings: dict,
    progress: dict | None = None,
) -> dict:
    best = _compute_best_attempt(attempts)
    graph_nodes, graph_edges = _build_search_graph(attempts)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_id": args.model_id,
        "technique": args.technique,
        "problem": {"level": args.level, "problem_id": args.problem_id, "name": problem_name},
        "settings": settings,
        "attempts": attempts,
        "search_graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
        "best": best,
    }
    if progress is not None:
        payload["progress"] = progress
    return payload


def _write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _write_intermediate_result(args: argparse.Namespace, run_id: str, label: str, payload: dict) -> str:
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints", run_id)
    checkpoint_path = os.path.join(checkpoint_dir, f"{label}.json")
    _write_json_atomic(checkpoint_path, payload)
    return checkpoint_path


async def run_async(args: argparse.Namespace) -> dict:
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
        modal_llm_base_url=args.modal_llm_base_url,
        modal_llm_api_key=args.modal_llm_api_key,
        modal_llm_timeout_s=args.modal_llm_timeout_s,
        modal_llm_max_parallel_requests=args.modal_llm_max_parallel_requests,
    )
    attempts: list[dict] = []
    next_attempt_id = 1
    run_id = getattr(args, "run_id", None) or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    settings = {
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
        "modal_llm_base_url": args.modal_llm_base_url,
        "modal_llm_timeout_s": args.modal_llm_timeout_s,
        "modal_llm_max_parallel_requests": args.modal_llm_max_parallel_requests,
        "early_stop_on_correct": args.early_stop_on_correct,
        "speedup_threshold": args.speedup_threshold,
        "save_intermediate": args.save_intermediate,
        "run_id": run_id,
    }

    if args.technique == "greedy":
        prompt = _build_initial_prompt(base_prompt, args.backend, args.include_think)
        raw, genm = llm.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        attempt = await _evaluate_generated_response_async(args, ref_arch_src, raw, genm, set())
        attempt["attempt_id"] = next_attempt_id
        next_attempt_id += 1
        attempt["parent_attempt_id"] = None
        attempts.append(attempt)

    elif args.technique == "best_of_n":
        prompt = _build_initial_prompt(base_prompt, args.backend, args.include_think)
        prompt_batch = [prompt for _ in range(args.n_samples)]
        generations = llm.generate_many(prompt_batch, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        seen_fingerprints: set[str] = set()
        pending_attempts: list[dict] = []
        eval_indices: list[int] = []
        eval_kernels: list[str] = []

        for raw, genm in generations:
            attempt, kernel_for_eval = _prepare_generated_attempt(args, raw, genm, seen_fingerprints)
            attempt["attempt_id"] = next_attempt_id
            next_attempt_id += 1
            attempt["parent_attempt_id"] = None

            pending_attempts.append(attempt)
            if kernel_for_eval is not None:
                eval_indices.append(len(pending_attempts) - 1)
                eval_kernels.append(kernel_for_eval)

            fp = attempt.get("kernel_fingerprint")
            if fp:
                seen_fingerprints.add(fp)

        if eval_kernels:
            eval_results = await _evaluate_kernels_batch_async(args, ref_arch_src, eval_kernels)
            for pending_idx, eval_result in zip(eval_indices, eval_results):
                pending_attempts[pending_idx]["eval"] = eval_result

        attempts.extend(pending_attempts)

    elif args.technique == "serial_refine":
        history: list[dict] = []
        seen_fingerprints: set[str] = set()
        for turn_idx in range(args.turns):
            parent_attempt_id = _parent_attempt_id_from_history(history)
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
            attempt = await _evaluate_generated_response_async(args, ref_arch_src, raw, genm, seen_fingerprints)
            attempt["attempt_id"] = next_attempt_id
            next_attempt_id += 1
            attempt["parent_attempt_id"] = parent_attempt_id
            attempts.append(attempt)
            fp = attempt.get("kernel_fingerprint")
            if fp:
                seen_fingerprints.add(fp)

            _apply_attempt_to_history(history, attempt)

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
                {
                    "trajectory_id": beam_idx,
                    "history": [],
                    "best_kernel": None,
                    "best_speedup": 0.0,
                    "best_score": 0.0,
                    "seen_fingerprints": set(),
                    "temperature": round1_temps[beam_idx],
                    "parent_trajectory_id": None,
                }
            )

        for round_idx in range(1, args.num_rounds + 1):
            for step_idx in range(args.steps_per_round):
                pending_attempts: list[dict] = []
                eval_indices: list[int] = []
                eval_kernels: list[str] = []
                prompt_batch: list[str] = []
                temp_batch: list[float] = []
                parent_attempt_ids: list[int | None] = []

                for traj in trajectories:
                    parent_attempt_id = _parent_attempt_id_from_history(traj["history"])
                    parent_attempt_ids.append(parent_attempt_id)
                    if traj["history"]:
                        prompt = _build_refinement_prompt(
                            base_prompt=base_prompt,
                            backend=args.backend,
                            include_think=args.include_think,
                            history=traj["history"],
                        )
                    else:
                        prompt = _build_initial_prompt(base_prompt, args.backend, args.include_think)
                    prompt_batch.append(prompt)
                    temp_batch.append(float(traj["temperature"]))

                generations = llm.generate_many(
                    prompt_batch,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.beam_temperature,
                    temperatures=temp_batch,
                )

                for idx, traj in enumerate(trajectories):
                    raw, genm = generations[idx]
                    parent_attempt_id = parent_attempt_ids[idx]
                    attempt, kernel_for_eval = _prepare_generated_attempt(
                        args,
                        raw,
                        genm,
                        traj["seen_fingerprints"],
                    )
                    attempt.update(
                        {
                            "attempt_id": next_attempt_id,
                            "parent_attempt_id": parent_attempt_id,
                            "round": round_idx,
                            "trajectory_id": traj["trajectory_id"],
                            "parent_trajectory_id": traj.get("parent_trajectory_id"),
                            "trajectory_step": step_idx + 1,
                            "generation_temperature": traj["temperature"],
                        }
                    )
                    next_attempt_id += 1
                    pending_attempts.append({"trajectory": traj, "attempt": attempt})
                    if kernel_for_eval is not None:
                        eval_indices.append(len(pending_attempts) - 1)
                        eval_kernels.append(kernel_for_eval)

                if eval_kernels:
                    eval_results = await _evaluate_kernels_batch_async(args, ref_arch_src, eval_kernels)
                    for pending_idx, eval_result in zip(eval_indices, eval_results):
                        pending_attempts[pending_idx]["attempt"]["eval"] = eval_result

                for item in pending_attempts:
                    traj = item["trajectory"]
                    attempt = item["attempt"]
                    attempts.append(attempt)

                    fp = attempt.get("kernel_fingerprint")
                    if fp:
                        traj["seen_fingerprints"].add(fp)
                    _apply_attempt_to_history(traj["history"], attempt)

                    ev = attempt.get("eval", {})
                    if ev.get("correctness"):
                        speed = float(ev.get("speedup_vs_ref", 0.0))
                        score = _kevin_score(ev)
                        if speed > traj["best_speedup"]:
                            traj["best_speedup"] = speed
                            traj["best_kernel"] = attempt.get("kernel")
                        if score > traj["best_score"]:
                            traj["best_score"] = score

                step_best = _compute_best_attempt(attempts)
                step_best_speedup = -1.0
                if step_best is not None:
                    step_best_speedup = float(step_best.get("eval", {}).get("speedup_vs_ref", -1.0))
                print(
                    f"[beam] round={round_idx}/{args.num_rounds} step={step_idx + 1}/{args.steps_per_round} "
                    f"attempts={len(attempts)} best_speedup={step_best_speedup:.2f}x"
                )
                if args.save_intermediate:
                    progress = {
                        "status": "in_progress",
                        "round": round_idx,
                        "step": step_idx + 1,
                        "attempts_total": len(attempts),
                        "best_speedup_vs_ref": step_best_speedup,
                        "trajectories": [
                            {
                                "trajectory_id": t["trajectory_id"],
                                "parent_trajectory_id": t.get("parent_trajectory_id"),
                                "history_len": len(t["history"]),
                                "best_speedup": t["best_speedup"],
                                "best_score": t["best_score"],
                            }
                            for t in trajectories
                        ],
                    }
                    checkpoint_payload = _make_result_payload(
                        args=args,
                        problem_name=problem_name,
                        attempts=attempts,
                        settings=settings,
                        progress=progress,
                    )
                    _write_intermediate_result(
                        args,
                        run_id,
                        f"round_{round_idx:02d}_step_{step_idx + 1:02d}",
                        checkpoint_payload,
                    )

            if round_idx >= args.num_rounds:
                continue

            trajectories.sort(key=lambda t: t["best_score"], reverse=True)
            survivors = trajectories[: args.beam_width]
            clones_per_survivor = args.num_beams // args.beam_width
            expanded = []
            new_trajectory_id = 0
            for survivor_rank, survivor in enumerate(survivors):
                clone_temps = _beam_temperatures(
                    count=clones_per_survivor,
                    base=args.beam_temperature,
                    jitter=args.beam_temp_jitter,
                    min_temp=args.beam_min_temperature,
                    max_temp=args.beam_max_temperature,
                )
                rank_offset = min(0.05 * survivor_rank, 0.15)
                for clone_idx in range(clones_per_survivor):
                    # Make lower-ranked survivors more exploratory (hotter) than top survivors.
                    t = max(
                        min(clone_temps[clone_idx] + rank_offset, args.beam_max_temperature),
                        args.beam_min_temperature,
                    )
                    expanded.append(
                        {
                            "trajectory_id": new_trajectory_id,
                            "history": copy.deepcopy(survivor["history"]),
                            "best_kernel": survivor["best_kernel"],
                            "best_speedup": survivor["best_speedup"],
                            "best_score": survivor["best_score"],
                            "seen_fingerprints": set(survivor["seen_fingerprints"]),
                            "temperature": t,
                            "parent_trajectory_id": survivor["trajectory_id"],
                        }
                    )
                    new_trajectory_id += 1
            trajectories = expanded
            if args.save_intermediate:
                round_best = _compute_best_attempt(attempts)
                round_best_speedup = -1.0
                if round_best is not None:
                    round_best_speedup = float(round_best.get("eval", {}).get("speedup_vs_ref", -1.0))
                progress = {
                    "status": "in_progress",
                    "round": round_idx,
                    "event": "after_prune_expand",
                    "attempts_total": len(attempts),
                    "best_speedup_vs_ref": round_best_speedup,
                    "survivor_trajectory_ids": [s["trajectory_id"] for s in survivors],
                    "next_trajectory_ids": [t["trajectory_id"] for t in trajectories],
                }
                checkpoint_payload = _make_result_payload(
                    args=args,
                    problem_name=problem_name,
                    attempts=attempts,
                    settings=settings,
                    progress=progress,
                )
                _write_intermediate_result(args, run_id, f"round_{round_idx:02d}_post_prune", checkpoint_payload)
    else:
        raise ValueError(f"Unknown technique: {args.technique}")
    return _make_result_payload(
        args=args,
        problem_name=problem_name,
        attempts=attempts,
        settings=settings,
    )


def run(args: argparse.Namespace) -> dict:
    return asyncio.run(run_async(args))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="cognition-ai/Kevin-32B")
    p.add_argument("--model-backend", default="modal_openai", choices=["modal_openai"])
    p.add_argument("--modal-llm-base-url", default=os.getenv("MODAL_LLM_BASE_URL"))
    p.add_argument("--modal-llm-api-key", default=os.getenv("MODAL_LLM_API_KEY"))
    p.add_argument("--modal-llm-timeout-s", type=float, default=600.0)
    p.add_argument("--modal-llm-max-parallel-requests", type=int, default=8)
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
    p.add_argument("--save-intermediate", action="store_true", default=True)
    p.add_argument("--no-save-intermediate", dest="save_intermediate", action="store_false")
    p.add_argument("--run-id", default=None)
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
