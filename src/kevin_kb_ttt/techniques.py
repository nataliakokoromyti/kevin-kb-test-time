"""Simple test-time techniques for kernel generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TechniqueConfig:
    technique: str
    n_samples: int
    turns: int


def refinement_prompt(base_prompt: str, last_kernel: str, feedback: str) -> str:
    return (
        base_prompt
        + "\n\nYou produced this kernel previously:\n"
        + last_kernel
        + "\n\nFeedback from evaluator:\n"
        + feedback
        + "\n\nReturn an improved full kernel implementation only."
    )
