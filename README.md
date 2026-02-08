# Kevin KB Test-Time

Test-time experimentation harness for `cognition-ai/Kevin-32B` using **KernelBench as the evaluation and prompting package**.

This repo avoids re-implementing KernelBench internals and uses KB directly for:
- prompt construction (`kernelbench.prompt_constructor_toml`)
- static reward-hacking checks (`kernelbench.kernel_static_checker`)
- correctness + timing eval (`kernelbench.eval`)
- dataset loading from Hugging Face (`ScalingIntelligence/KernelBench`)
- structured multi-turn parsing (`<think>`, `<KERNEL>`, `<SUMMARY>`)
- async eval with timeout + duplicate-kernel cache
- Modal-only evaluation path (KB script-style)

## 1) Setup

```powershell
cd C:\Users\natal\kevin-kb-test-time
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

This repo includes KernelBench as a submodule for exact harness execution.
Initialize submodules:

```powershell
git submodule update --init --recursive
```

## 2) Smoke run (single problem)

```powershell
kevin-kb-run --level 1 --problem-id 1 --technique greedy --max-new-tokens 16384
# fallback if script PATH is stale:
python -m kevin_kb_ttt.runner --level 1 --problem-id 1 --technique greedy --max-new-tokens 16384
```

## 3) Compare test-time techniques

```powershell
kevin-kb-run --level 1 --problem-id 1 --technique best_of_n --n-samples 4
kevin-kb-run --level 1 --problem-id 1 --technique serial_refine --turns 3 --modal-gpu L40S --modal-timeout-s 120 --cache-results
kevin-kb-run --level 1 --problem-id 1 --technique beam_search --num-beams 16 --beam-width 4 --steps-per-round 4 --num-rounds 2 --beam-temperature 0.9 --modal-gpu L40S --modal-timeout-s 120 --cache-results
```

Results are written to `results/*.json`.

## Structured Output Contract

For multi-turn refinement and beam search, the runner instructs the model to emit:

- `<think>...</think>` (optional; enabled with `--include-think`)
- `<KERNEL>```python ... ```</KERNEL>` (required)
- `<SUMMARY>...</SUMMARY>` (used as compressed carry-over between turns)

The next-turn prompt carries summary + categorized evaluator feedback (format/compile/runtime/correctness/perf), not raw CoT.

## Eval Controls

- Modal is the only supported eval mode.
- Eval is executed through exact KB harness script: `third_party/KernelBench/scripts/run_and_check.py`
- `--modal-gpu`: target Modal GPU (for example `L40S`, `H100`, `A100`)
- `--modal-timeout-s`: timeout for each Modal evaluation
- `--cache-results` / `--no-cache-results`: LRU cache for duplicate kernels
- `--max-new-tokens`: use high values (recommended `16384`; minimum `8192`), otherwise Kevin generations get truncated

## Notes

- Default model id: `cognition-ai/Kevin-32B`
- Set up Modal auth before running: `modal token new`
- If you choose `--dataset-source local`, pass `--kb-base-path` explicitly.
