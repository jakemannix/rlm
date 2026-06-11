# Sleep-time consolidation scripts

CLI entry points for the nightly gate → judge → LoRA → eval cycle in
`rlm/sleep/`. Design rationale: `docs/learning_signal.md`; runbook:
`docs/sleep_consolidation.md`.

## Offline smoke tests (no GPU, no API key, no network)

```bash
uv run python scripts/sleep/run_nightly.py --synthetic --mock-judge --dry-run --days 2
uv run python scripts/sleep/run_sweep.py --synthetic --mock-judge --dry-run
```

`--synthetic` uses the deterministic offline trace generator,
`--mock-judge` cans the judge, `--dry-run` swaps LoRA training + NLL evals
for no-op fakes — exercising the full pipeline plumbing and artifacts.

## Real runs (GPU + judge API key)

```bash
# Install the extra: uv pip install -e ".[sleep]"
OPENAI_API_KEY=... uv run python scripts/sleep/run_nightly.py \
    --dataset THUDM/AgentInstruct --config os \
    --policy-model Qwen/Qwen2.5-1.5B-Instruct \
    --judge-model gpt-4o --days 2

OPENAI_API_KEY=... uv run python scripts/sleep/run_sweep.py \
    --dataset THUDM/AgentInstruct --config os \
    --grid '{"adapter.lr": [1e-4, 2e-4, 5e-4], "adapter.rank": [8, 16, 32]}'
```

Artifacts land under `runs/sleep/`: per-day `gate_decisions.json`,
`judge_outputs.json`, `sft_data.jsonl`, `adapter/`, `day_result.json`;
sweeps add `sweep_results.csv`.

For Colab, use `notebooks/sleep_consolidation_sweep.ipynb` (regenerate via
`python notebooks/build_sleep_notebook.py`).
