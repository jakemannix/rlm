# Sleep-time consolidation PoC (`rlm/sleep/`)

> **Status: working proof-of-concept (2026-06).** Implements the practical
> nightly-consolidation loop motivated by `learning_signal.md` (what signal to
> learn from) and `memory_field_positioning.md` (why the niche is open). This is
> the engineering-as-data-collection track: the curation traces this pipeline
> produces are exactly what a meta-learned parametric writer would later train on.

## The loop

```
day N episodes ──► GATE (cheap) ──► JUDGE (high TTC) ──► verified SFT + replay
                   ~25% budget       reflect + distill         │
                                     + verify pass             ▼
day N+1 + test ◄── EVAL (NLL) ◄──── adapter ◄──────────── LoRA TRAIN
```

| Stage | Module | What it does |
|---|---|---|
| Traces | `rlm/sleep/traces.py` | HF agentic traces (default `THUDM/AgentInstruct`) normalized to `Episode`; deterministic hash splits; sequential "days" |
| Gate | `rlm/sleep/gate.py` | Heuristic learnworthiness score (errors / retries / failure / length) + optional NLL `SurpriseGate`; top-`budget_fraction` above a floor |
| Judge | `rlm/sleep/judge.py` | LLM-as-a-Judge reflection (any `BaseLM`): learn/skip verdict, lesson, distilled examples; self-consistency over `n_samples`; YES/NO verification pass per example |
| Dataset | `rlm/sleep/dataset.py` | Verified examples only; general replay mix-in; **leakage guard** (fails if any example sources from held-out episodes) |
| Train | `rlm/sleep/lora.py` | Masked-SFT LoRA (peft, no trl), sized for Colab T4/L4 with 1–4B instruct models |
| Eval | `rlm/sleep/evals.py` | Next-day NLL (transfer), test NLL (generalization), retention NLL (forgetting guard) |
| Loop | `rlm/sleep/loop.py` | `run_night(...)` — one cycle, every artifact persisted |
| Sweep | `rlm/sleep/sweep.py` | Dotted-key grid (e.g. `adapter.lr`, `gate.budget_fraction`) → CSV + leaderboard |

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

# Real (GPU + judge key):
uv pip install -e ".[sleep]"
OPENAI_API_KEY=... uv run python scripts/sleep/run_nightly.py --days 2

# Colab sweep:
#   notebooks/sleep_consolidation_sweep.ipynb
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

- AgentInstruct traces are expert demonstrations, so the gate's
  failure-pattern features fire rarely there; the NLL `SurpriseGate` is the
  interesting selector on clean traces (sweepable via `gate.use_surprise`).
  Synthetic traces plant failures for development and tests.
- One adapter per night, no cross-night accumulation/merging yet — the
  multi-night question (merge? route? re-distill?) is exactly the
  consolidation-stack discussion in the docs; the harness already evaluates
  any candidate policy via `eval_fn`.
- Judged win-rate eval (generate + blind A/B) is not wired in; NLL is the
  cheap, deterministic v1 metric.
