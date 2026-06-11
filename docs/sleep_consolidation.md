# Sleep-time consolidation PoC (`rlm/sleep/`)

> **Status: working proof-of-concept (2026-06).** Implements the practical
> nightly-consolidation loop motivated by `learning_signal.md` (what signal to
> learn from) and `memory_field_positioning.md` (why the niche is open). This is
> the engineering-as-data-collection track: the curation traces this pipeline
> produces are exactly what a meta-learned parametric writer would later train on.

## The loop

```
day N episodes ──► GATE (cheap) ──► JUDGE (high TTC) ──► verified SFT + replay
                   ~25% budget       the SAME model,           │
                                     more thinking budget      ▼
day N+1 + test ◄── EVAL (NLL) ◄──── adapter ◄──────────── LoRA TRAIN
```

The judge is the policy model itself by default — frozen base weights with
a larger test-time-compute budget (self-consistency sampling, longer
generations). No external teacher, no API key: the generation–verification
gap *within one model* is the supervision (`learning_signal.md` §5).

| Stage | Module | What it does |
|---|---|---|
| Traces | `rlm/sleep/traces.py` | HF agentic traces (default `THUDM/AgentInstruct`; domains are *splits*: os/db/alfworld/webshop/kg/mind2web) normalized to `Episode`; deterministic hash splits; sequential "days" |
| Gate | `rlm/sleep/gate.py` | Heuristic learnworthiness score (errors / retries / failure / length — errors scanned in tool steps *and* user observation turns) + optional NLL `SurpriseGate` (`--use-surprise`); top-`budget_fraction` above a floor |
| Judge | `rlm/sleep/judge.py` | LLM-as-a-Judge reflection: learn/skip verdict, lesson, distilled examples; self-consistency over `n_samples`; YES/NO verification pass per example; unparseable samples count as skip votes; disk cache keyed by prompt + judge settings |
| Self-judge | `rlm/sleep/local_judge.py` | **The judge is the policy model itself** (frozen base, higher TTC: sampling + long generations). External judges remain available for ablation via `--judge-model <name>` |
| Dataset | `rlm/sleep/dataset.py` | Verified examples only; general replay mix-in; **leakage guard** (fails if any example sources from held-out episodes) |
| Train | `rlm/sleep/lora.py` | Masked-SFT LoRA (peft, no trl), sized for Colab T4/L4 with 1–4B instruct models |
| Eval | `rlm/sleep/evals.py` | Next-day NLL (transfer), test NLL (generalization), retention NLL over a frozen 50-item probe (forgetting guard); `rlm/sleep/winrate.py` adds blind pairwise A/B win-rate |
| Loop | `rlm/sleep/loop.py` | `run_night(...)` — one cycle, every artifact persisted; `extra_examples` enables the cumulative re-distill baseline (`--cumulative`) |
| Sweep | `rlm/sleep/sweep.py` | Dotted-key grid (e.g. `adapter.lr`, `gate.budget_fraction`) → CSV + leaderboard; judge cache shared across points |

## Design decisions (and where they come from)

- **Budgeted gate, not a threshold-only filter** — the economic premise is
  that only a small fraction of a day deserves expensive reflection
  (`learning_signal.md` §1; RHO-style selection lifted to episode level).
- **Judge must verify before anything earns a gradient** — the
  generation–verification gap is the only supervision available on unlabeled
  experience; unverified examples are dropped (`learning_signal.md` §5).
- **Replay mixed into every nightly batch + a fixed retention probe in every
  eval** — the continual-pretraining recipe's answer to forgetting, and the
  metric that catches it if it happens anyway.
- **Hash-based splits + a leakage guard that throws** — judge-generated
  examples cite their source episode, so train/test hygiene is enforceable
  and enforced.
- **Dependency-injected train/eval in the loop** — the full pipeline runs
  offline in CI with fakes (`--dry-run`, mock judge, synthetic traces); the
  real implementations need the `sleep` extra.

## Quick start

```bash
# Offline, instant, no GPU:
uv run python scripts/sleep/run_nightly.py --synthetic --mock-judge --dry-run --days 2

# Real, fully self-contained (no API key; CPU works for 0.5B, GPU for 1.5B+):
uv sync --extra sleep   # CPU torch via [tool.uv]; Colab/pip installs get CUDA
uv run python scripts/sleep/run_nightly.py \
    --dataset THUDM/AgentInstruct --config os --episodes-per-day 16 \
    --policy-model Qwen/Qwen2.5-0.5B-Instruct --judge-model self \
    --device cpu --use-surprise --days 1

# Multi-night cumulative baseline:        add --cumulative
# External judge (ablation):              --judge-model gpt-4o  (needs OPENAI_API_KEY)
# Colab sweep: notebooks/sleep_consolidation_sweep.ipynb
```

## Reading the metrics

All three metrics are mean per-token NLL (nats); **deltas are adapted − base,
so negative = improvement**:

- `next_day_delta < 0` — tonight's lessons helped tomorrow (forward transfer).
- `test_delta < 0` — lessons generalize beyond the stream's immediate future.
- `retention_delta > 0` — the adapter is eroding general ability. Respond by
  raising `adapter.replay_ratio`, lowering `adapter.lr`/`epochs`, or tightening
  `judge.min_confidence`.

## Current limitations (deliberate PoC cuts)

- AgentInstruct traces are expert demonstrations: measured on the live
  data, the heuristic gate selects 6% (os) / 2% (db) / 9% (alfworld) and
  ~0% on webshop/kg/mind2web — on those domains `--use-surprise` is the
  only viable selector. Synthetic traces plant failures for development.
- Cross-night *merging/routing* of adapters is still open; the implemented
  multi-night baselines are independent nights (default) and cumulative
  re-distillation (`--cumulative`).
- Win-rate eval (`rlm/sleep/winrate.py`) is implemented but not wired into
  `run_night`; call it from a notebook/script on the night's adapter.
- A small self-judge can rubber-stamp verification (observed at 0.5B:
  mediocre examples pass YES/NO). The verification ablation and a
  corrupted-example canary are the Phase-5 checks for this.

See `docs/sleep_testing_plan.md` for measured CPU results and the GPU
continuation plan.
