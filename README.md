# Kevin KB Test-Time

Test-time experimentation harness for `cognition-ai/Kevin-32B` using **KernelBench as the evaluation and prompting package**.

This repo avoids re-implementing KernelBench internals and uses KB directly for:
- prompt construction (`kernelbench.prompt_constructor_toml`)
- static reward-hacking checks (`kernelbench.kernel_static_checker`)
- correctness + timing eval (`kernelbench.eval`)
- dataset loading from Hugging Face (`ScalingIntelligence/KernelBench`)
- Kevin/QwQ-native multi-turn parsing (`<think>` optional, Python code block + plain summary)
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

## 2b) Modal-Only Inference

Inference is remote-only. The runner calls an OpenAI-compatible endpoint hosted on Modal.

Set:
```powershell
$env:MODAL_LLM_BASE_URL="https://<your-modal-endpoint>/v1"
$env:MODAL_LLM_API_KEY="<optional-api-key>"
```

or pass flags directly:
```powershell
kevin-kb-run --model-backend modal_openai --modal-llm-base-url https://<your-modal-endpoint>/v1 ...
```

## 3) Compare test-time techniques

```powershell
kevin-kb-run --level 1 --problem-id 1 --technique best_of_n --n-samples 4
kevin-kb-run --level 1 --problem-id 1 --technique serial_refine --turns 3 --modal-gpu L40S --modal-timeout-s 120 --cache-results
kevin-kb-run --level 1 --problem-id 1 --technique beam_search --num-beams 16 --beam-width 4 --steps-per-round 4 --num-rounds 2 --beam-temperature 0.9 --modal-gpu L40S --modal-timeout-s 120 --cache-results
```

Results are written to `results/*.json`.

## Output Contract

For multi-turn refinement and beam search, the runner expects:

- Optional natural `<think>...</think>` reasoning
- A Python code block with full `ModelNew`
- A short plain-text summary after the code block

The next-turn prompt carries summary + categorized evaluator feedback (format/compile/runtime/correctness/perf), not raw CoT. No custom wrapper tags are required.

## Eval Controls

- Modal is the only supported eval mode.
- Modal is the only supported inference mode.
- Eval is executed through exact KB harness script: `third_party/KernelBench/scripts/run_and_check.py`
- `--modal-gpu`: target Modal GPU (for example `L40S`, `H100`, `A100`)
- `--modal-timeout-s`: timeout for each Modal evaluation
- `--cache-results` / `--no-cache-results`: LRU cache for duplicate kernels
- `--max-new-tokens`: use high values (recommended `16384`; minimum `8192`), otherwise Kevin generations get truncated
- `--save-intermediate` / `--no-save-intermediate`: save beam-search checkpoints to `results/checkpoints/<run_id>/`
- `--run-id`: optional stable id for checkpoint folder naming
- `--trim-history-to-fit` / `--no-trim-history-to-fit`: drop oldest refinement attempts if prompt exceeds budget
- `--max-prompt-tokens`: approximate prompt token budget (default `24000`)
- `--prompt-warning-tokens`: print warning when prompt exceeds threshold (default `24000`)

## Notes

- Default model id: `cognition-ai/Kevin-32B`
- Default inference backend: `modal_openai` (no local HF/vLLM loading)
- Set up Modal auth before running: `modal token new`
- If you choose `--dataset-source local`, pass `--kb-base-path` explicitly.
