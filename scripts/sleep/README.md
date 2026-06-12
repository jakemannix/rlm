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

## Real runs (no API key needed — the policy model judges itself)

```bash
# Install the extra: uv sync --extra sleep
# (CPU torch in uv-managed envs; pip/Colab installs get CUDA wheels)

# CPU-feasible at 0.5B; use cuda for 1.5B+. AgentInstruct domains are
# splits: os, db, alfworld, webshop, kg, mind2web (122-538 episodes each).
uv run python scripts/sleep/run_nightly.py \
    --dataset THUDM/AgentInstruct --config os --episodes-per-day 16 \
    --policy-model Qwen/Qwen2.5-0.5B-Instruct --judge-model self \
    --device cpu --use-surprise --days 1

# Multi-night cumulative re-distill baseline:
#   add --cumulative
# External-judge ablation (needs OPENAI_API_KEY):
#   --judge-model gpt-4o

uv run python scripts/sleep/run_sweep.py \
    --dataset THUDM/AgentInstruct --config os --judge-model self --use-surprise \
    --grid '{"adapter.lr": [1e-4, 2e-4, 5e-4], "adapter.rank": [8, 16, 32]}'
```

Judge reflections are cached (`judge_cache.json` in the run dir), so sweep
points that share gate/judge settings pay for the judge once, and re-runs
are free.

Artifacts land under `runs/sleep/`: per-day `gate_decisions.json`,
`judge_outputs.json`, `sft_data.jsonl`, `adapter/`, `day_result.json`;
sweeps add `sweep_results.csv`.

For Colab, use `notebooks/sleep_consolidation_sweep.ipynb` (regenerate via
`python notebooks/build_sleep_notebook.py`).

## Claude Code traces → gold-labeled memories

Natural sessions carry the error→correction signal AgentInstruct lacks
(measured: 41% of episodes vs 9% on the best AgentInstruct domain). The
transcripts are personal data: archive and label locally, never commit
them, and prefer `--device mps`/`cpu` over remote GPUs.

```bash
# 1. Preserve transcripts before Claude Code's cleanup prunes them
#    (idempotent; re-run any time; archive lives outside the repo)
uv run python scripts/sleep/extract_cc_sessions.py

# 2. Offline smoke of the labeling stack
uv run python scripts/sleep/label_gold.py --mock-judge --limit-episodes 5

# 3. Real labeling: reflect -> verify -> 5-dim rubric -> adversarial
#    refuters -> gold/silver/reject (self-judge, no API key)
uv run python scripts/sleep/label_gold.py --device mps \
    --policy-model Qwen/Qwen2.5-1.5B-Instruct --limit-episodes 30
```

Outputs under `runs/sleep/gold/` (gitignored): `gold_labels.jsonl`
(every candidate with scores and votes), `sft_gold.jsonl` (trainer-ready
gold examples), `summary.json`. The canary
(`scripts/sleep/verify_canary.py`) still applies — run it before
trusting a new judge model's labels.
