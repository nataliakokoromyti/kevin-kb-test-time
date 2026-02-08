from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime

from .kb_pipeline import (
    build_prompt,
    evaluate_kernel,
    extract_kernel_code,
    get_problem,
    static_check,
    to_dict,
)
from .model import KevinHF


def _eval_feedback(summary) -> str:
    if not summary.compiled:
        return f"compile_failed metadata={summary.metadata}"
    if not summary.correctness:
        return f"correctness_failed metadata={summary.metadata}"
    return f"correct speedup_vs_ref={summary.speedup_vs_ref:.4f}"


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

    llm = KevinHF(args.model_id, torch_dtype=args.torch_dtype)

    attempts = []

    if args.technique == "greedy":
        raw, genm = llm.generate(base_prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        kernel = extract_kernel_code(raw)
        ok, err, warns = static_check(kernel, backend=args.backend, precision=args.precision)
        if not ok:
            attempts.append({"static_check_ok": False, "static_error": err, "static_warnings": warns})
        else:
            summary = evaluate_kernel(
                ref_arch_src,
                kernel,
                device=args.device,
                num_correct_trials=args.num_correct_trials,
                num_perf_trials=args.num_perf_trials,
                backend=args.backend,
                precision=args.precision,
            )
            attempts.append({
                "static_check_ok": True,
                "static_warnings": warns,
                "generation": asdict(genm),
                "eval": to_dict(summary),
                "kernel": kernel,
            })

    elif args.technique == "best_of_n":
        for _ in range(args.n_samples):
            raw, genm = llm.generate(base_prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            kernel = extract_kernel_code(raw)
            ok, err, warns = static_check(kernel, backend=args.backend, precision=args.precision)
            if not ok:
                attempts.append({"static_check_ok": False, "static_error": err, "static_warnings": warns})
                continue
            summary = evaluate_kernel(
                ref_arch_src,
                kernel,
                device=args.device,
                num_correct_trials=args.num_correct_trials,
                num_perf_trials=args.num_perf_trials,
                backend=args.backend,
                precision=args.precision,
            )
            attempts.append({
                "static_check_ok": True,
                "static_warnings": warns,
                "generation": asdict(genm),
                "eval": to_dict(summary),
                "kernel": kernel,
            })

    elif args.technique == "serial_refine":
        prompt = base_prompt
        last_kernel = ""
        for _ in range(args.turns):
            raw, genm = llm.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            kernel = extract_kernel_code(raw)
            ok, err, warns = static_check(kernel, backend=args.backend, precision=args.precision)
            if not ok:
                attempts.append({"static_check_ok": False, "static_error": err, "static_warnings": warns})
                prompt = prompt + "\n\nPrevious candidate failed static checks. Return a safe kernel-only implementation."
                continue

            summary = evaluate_kernel(
                ref_arch_src,
                kernel,
                device=args.device,
                num_correct_trials=args.num_correct_trials,
                num_perf_trials=args.num_perf_trials,
                backend=args.backend,
                precision=args.precision,
            )
            attempts.append({
                "static_check_ok": True,
                "static_warnings": warns,
                "generation": asdict(genm),
                "eval": to_dict(summary),
                "kernel": kernel,
            })
            feedback = _eval_feedback(summary)
            last_kernel = kernel
            prompt = (
                base_prompt
                + "\n\nYou produced this candidate:\n"
                + last_kernel
                + "\n\nEvaluator feedback:\n"
                + feedback
                + "\n\nReturn an improved full kernel implementation only."
            )
    else:
        raise ValueError(f"Unknown technique: {args.technique}")

    best = None
    for a in attempts:
        ev = a.get("eval")
        if not ev:
            continue
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
        },
        "attempts": attempts,
        "best": best,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="cognition-ai/Kevin-32B")
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--technique", default="greedy", choices=["greedy", "best_of_n", "serial_refine"])
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--turns", type=int, default=3)
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
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
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
