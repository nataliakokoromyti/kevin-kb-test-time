# Kevin KB Test-Time

Test-time experimentation harness for `cognition-ai/Kevin-32B` using **KernelBench as the evaluation and prompting package**.

This repo avoids re-implementing KernelBench internals and uses KB directly for:
- prompt construction (`kernelbench.prompt_constructor_toml`)
- static reward-hacking checks (`kernelbench.kernel_static_checker`)
- correctness + timing eval (`kernelbench.eval`)
- dataset loading from Hugging Face (`ScalingIntelligence/KernelBench`)

## 1) Setup

```powershell
cd C:\Users\natal\kevin-kb-test-time
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

KernelBench is consumed as an installed package; no local KernelBench checkout is required.

## 2) Smoke run (single problem)

```powershell
kevin-kb-run --level 1 --problem-id 1 --technique greedy --max-new-tokens 512
# fallback if script PATH is stale:
python -m kevin_kb_ttt.runner --level 1 --problem-id 1 --technique greedy --max-new-tokens 512
```

## 3) Compare test-time techniques

```powershell
kevin-kb-run --level 1 --problem-id 1 --technique best_of_n --n-samples 4
kevin-kb-run --level 1 --problem-id 1 --technique serial_refine --turns 3
```

Results are written to `results/*.json`.

## Notes

- Default model id: `cognition-ai/Kevin-32B`
- You need a CUDA-capable environment to run local timing eval.
- If you choose `--dataset-source local`, pass `--kb-base-path` explicitly.
