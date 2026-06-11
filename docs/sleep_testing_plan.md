# Sleep-consolidation PoC — phased testing plan & next steps

> **Audience: the next agent/human session picking up branch
> `claude/clever-hopper-b6orfo` (PR #2).** This is a handoff document: what
> exists, what has actually been verified, and a phased plan — with GPU-time
> and judge-API budgets — to take the PoC from "plumbing verified offline" to
> "real numbers on real traces". Background: `docs/sleep_consolidation.md`
> (runbook), `docs/learning_signal.md` (why this loop), `scripts/sleep/README.md`
> (CLI usage).

## Current state (2026-06-11)

**Verified** (all offline, no GPU, no network):

- 27/27 sleep tests green (`uv run pytest tests/test_sleep_*.py`), 296/296 repo-wide.
- Both CLIs run end-to-end with `--synthetic --mock-judge --dry-run`; all
  artifacts (gate decisions, judge outputs, SFT JSONL, day result, sweep CSV)
  written and well-formed.
- Deterministic hash splits, leakage guard, replay mixing, masked-SFT label
  construction — all unit-tested.

**Never executed** (this is what the phases below de-risk, in order):

1. `train_lora` / `evaluate_model` on an actual GPU (torch/peft code paths
   are import-guarded and only exercised by fakes so far).
2. Loading + normalizing real `THUDM/AgentInstruct` records (the normalizer
   is tested against synthetic records in both HF formats, not the live dataset).
3. A real judge LM producing parseable reflection JSON at scale.
4. Any actual NLL number. We do not yet know if the loop *helps*.

**Known gaps** (discovered during build, deliberate cuts — see backlog):

- `gate.use_surprise` exists in config but `run_night` never constructs a
  `SurpriseGate` — the NLL gate is currently dead config. This matters
  because AgentInstruct traces are expert demonstrations: the heuristic
  gate's error/retry/failure features rarely fire there.
- `run_sweep` re-runs gate + judge for **every** grid point. Judge API cost
  multiplies by grid size even when gate/judge settings are identical
  across points.
- Default `torch_dtype: "bfloat16"` — fine on L4/A100, **not supported on
  Colab T4** (compute capability 7.5). Use `--device cuda` + override dtype
  to `float16`, or just use L4.
- One adapter per night; no cross-night accumulation/merging.
- NLL-only eval; no judged win-rate.

---

## Phase 0 — Offline regression (CPU, 0 GPU-h, ~10 min)

Re-establish the baseline before touching anything. Run on every fresh
session and before every push:

```bash
uv run pytest -q                          # expect 296 passed, 14 skipped
uv run ruff check . && uv run ruff format --check .
uv run python scripts/sleep/run_nightly.py --synthetic --mock-judge --dry-run --days 2
uv run python scripts/sleep/run_sweep.py  --synthetic --mock-judge --dry-run
```

**Exit criteria:** all green. If not, fix before proceeding — nothing
downstream is trustworthy otherwise.

## Phase 1 — GPU plumbing shakedown (~0.5 GPU-h on L4; T4 OK with fp16)

**Goal:** prove `lora.py` and `evals.py` actually work — model loads, chat
template applies, masked labels align, training loss decreases, adapter
saves/reloads, NLLs are finite and sane. Use synthetic traces + mock judge so
the *only* new variable is the GPU code.

```bash
uv pip install -e ".[sleep]"
uv run python scripts/sleep/run_nightly.py \
    --synthetic --mock-judge --days 1 \
    --episodes-per-day 16 \
    --policy-model Qwen/Qwen2.5-0.5B-Instruct
```

(0.5B keeps the iteration loop ~5 min; bump to 1.5B once green.)

**Check specifically:**

- `training_meta.json` shows loss decreasing across steps (not NaN, not flat).
- `adapter/` directory loads via `load_adapter` and produces *different*
  NLLs than base (if identical, the adapter isn't being applied).
- Base NLL on synthetic responses is plausible (~1–4 nats/token for an
  instruct model on English text; >8 suggests a chat-template or label-mask
  bug — inspect `encode_example` output token-by-token before trusting
  anything downstream).
- Peak VRAM (`torch.cuda.max_memory_allocated`) — record it; this calibrates
  what fits where for later phases.

**Exit criteria:** one full night runs on GPU; NLL deltas are finite;
adapter round-trips. **Budget: ~0.5 GPU-h, $0 judge.**

## Phase 2 — First real night (~1–1.5 GPU-h on L4, ~$1–3 judge API)

**Goal:** the first real data point. AgentInstruct/os traces, real judge,
1.5B policy, one night.

**Prerequisite (small code task):** wire the surprise gate — in
`run_night`, when `config.gate.use_surprise` is true, construct a
`SurpriseGate` from the policy model and pass it to `select_for_reflection`.
Without it, expect the heuristic gate to select near-arbitrarily on expert
traces (low, flat scores; the `min_score` floor may select almost nothing —
which is itself worth recording as the day-0 observation).

```bash
OPENAI_API_KEY=... uv run python scripts/sleep/run_nightly.py \
    --dataset THUDM/AgentInstruct --config os \
    --policy-model Qwen/Qwen2.5-1.5B-Instruct \
    --judge-model gpt-4o --days 1
```

**Judge cost math** (so spend is predictable): 64 episodes/day ×
25% budget = ~16 reflections × `judge.n_samples` (use 3) ≈ 48 calls at
~2–4k input tokens each, plus ~1 short verification call per distilled
example (~30–50). Roughly $1–3/night with gpt-4o; scales linearly with
`n_samples` and `budget_fraction`.

**Check specifically:**

- `load_hf_traces` on the live dataset: row count, role mapping, no empty
  transcripts (this is the first contact with real AgentInstruct schema).
- `judge_outputs.json`: JSON parse failure rate (count `skip` verdicts that
  came from unparseable output vs. genuine skips). >10% parse failures →
  tighten `REFLECT_PROMPT` before burning more API budget.
- Verification pass rate. Near-100% means the verifier is rubber-stamping
  (try a deliberately corrupted example as a canary); near-0% means the
  reflect prompt and verify prompt disagree about format.
- The three deltas. **Do not expect wins yet** — this phase is about the
  measurement pipeline, not the result.

**Exit criteria:** a real `day_result.json` with finite deltas and a judge
pipeline whose failure modes are quantified. **Budget: ~1–1.5 GPU-h, ~$1–3.**

## Phase 3 — Judge caching + hyperparameter sweep (~3–6 GPU-h, ~$2–5 judge)

**Prerequisite (code task, do first):** cache judge outputs keyed by
`(episode_id, judge_model, n_samples, prompt_hash)` — a JSON file next to
the sweep dir is fine. With the current code a 12-point sweep re-spends the
judge 12×; with the cache, points sharing gate/judge settings pay once.
Sweep `gate.budget_fraction` knowing each *new* fraction adds judge calls
for the newly-included episodes only (cache hits cover the rest).

Then sweep on the **adapter surface** first (pure GPU, zero marginal judge
cost after the first point):

```bash
OPENAI_API_KEY=... uv run python scripts/sleep/run_sweep.py \
    --dataset THUDM/AgentInstruct --config os \
    --grid '{"adapter.lr": [5e-5, 1e-4, 2e-4, 5e-4], "adapter.rank": [8, 16, 32], "adapter.replay_ratio": [0.0, 0.2, 0.4]}'
```

Size guidance: one point ≈ one night minus judge ≈ 10–15 min on L4 at 1.5B.
A 12–18 point grid is an afternoon (~3–6 GPU-h). Use
`notebooks/sleep_consolidation_sweep.ipynb` for the scatter
(transfer vs. forgetting) plot.

**What we're actually asking:** is there *any* (lr, rank, replay) region
with `next_day_delta < 0` and `retention_delta ≈ 0`? Secondary: does
`replay_ratio = 0` visibly hurt retention (it should — if it doesn't, the
retention probe is too easy and needs harder items).

**Exit criteria:** sweep CSV + plot committed under `runs/` or as a docs
figure; a chosen default config justified by the leaderboard.
**Budget: ~3–6 GPU-h, ~$2–5 (with cache).**

## Phase 4 — Multi-night learning curves (~2–4 GPU-h, ~$5–15 judge)

**Goal:** the headline plot — 5–7 consecutive nights with the Phase-3 best
config, each night's adapter evaluated on its next day **and** on the fixed
test split and retention probe.

Two policies to compare (the loop already supports both via `run_night`'s
arguments; multi-night accumulation is the one real code addition):

1. **Independent nights** (current behavior): adapter_N trained from base
   on night N only. Curve answers: "is each night's material learnable?"
2. **Cumulative**: night N trains from (or merges with) adapter_{N-1}, or
   retrains from base on the union of all nights' SFT data ("re-distill" —
   cheapest correct baseline, recommended first). Curve answers: "does
   knowledge accumulate without forgetting?" — this is the actual thesis.

Plot: x = night, y = test NLL and retention NLL for both policies.
Divergence between them *is* the consolidation-stack finding, whichever
direction it goes.

**Exit criteria:** learning-curve plot + a paragraph of interpretation in
`docs/sleep_consolidation.md` (replace the "deliberate cuts" bullet).
**Budget: ~2–4 GPU-h, ~$5–15.**

## Phase 5 — Ablations & extensions (open-ended; A100 for 7–8B)

Ordered by information-per-GPU-hour:

1. **Verification ablation** (`judge.verify_examples = false`): the doc's
   central claim is the verification gate is load-bearing. One sweep column,
   ~1 GPU-h. If unverified does just as well, that's a major (negative)
   finding — report it honestly.
2. **Gate ablation**: random-selection gate vs. heuristic vs. surprise at
   equal budget. Tests whether *selection* matters or just *volume*. ~2 GPU-h.
3. **Judge TTC curve**: `n_samples ∈ {1, 3, 5}` — does more judge compute
   buy better adapters? Pure judge cost (~3× Phase 2), no extra GPU.
4. **Other domains**: `--config webshop` / `mind2web` / `db` — does the
   recipe transfer or was it tuned to os-style traces? ~1 GPU-h each.
5. **Scale**: Qwen2.5-7B-Instruct on A100-40GB (fits with rank-16 LoRA at
   `max_seq_len 1024`; expect ~3–4× Phase-2 wall-clock, so ~4–6 GPU-h for a
   night + mini-sweep). Does the effect grow, shrink, or vanish with scale?
6. **Win-rate eval**: generate responses on next-day prompts from base vs.
   adapted, blind A/B by the judge. The metric everyone will ask for; NLL
   alone won't convince anyone outside this repo.

## Budget summary

| Phase | GPU | GPU-h | Judge $ | Wall-clock |
|---|---|---|---|---|
| 0 — offline regression | none | 0 | 0 | 10 min |
| 1 — GPU shakedown | T4(fp16)/L4 | ~0.5 | 0 | ~1 h |
| 2 — first real night | L4 | 1–1.5 | $1–3 | ~2 h |
| 3 — caching + sweep | L4 | 3–6 | $2–5 | afternoon |
| 4 — learning curves | L4 | 2–4 | $5–15 | ~half day |
| 5 — ablations (1–4) | L4 | ~5 | ~$10 | as needed |
| 5 — scale (7B) | A100 | 4–6 | $3–5 | ~half day |
| **Total through Phase 4** | | **~7–12** | **~$10–25** | **~2 days** |

All GPU-h estimates assume Qwen2.5-1.5B at `max_seq_len 1024`; they are
calibrated guesses — Phase 1's measured per-night wall-clock supersedes them.

## Decision points / kill criteria

- **After Phase 2:** if judge parse-failure or verification pathologies eat
  >25% of examples, fix prompts before sweeping — sweep results on a broken
  judge are noise.
- **After Phase 3:** if *no* grid point achieves `next_day_delta < 0`
  without retention damage, stop and diagnose before Phase 4. Likeliest
  causes, in order: (a) too few examples per night (raise `budget_fraction`
  / `max_examples_per_episode`), (b) eval episodes too dissimilar to
  training distillations (inspect SFT data vs. next-day transcripts
  side-by-side), (c) the policy already knows AgentInstruct-style content
  (check whether base NLL is already very low — if so, switch domains or
  use harder traces).
- **After Phase 4:** if independent nights work but cumulative doesn't,
  that's not failure — that's the forgetting result the whole research
  program predicts; write it up.

## Engineering backlog (small, ordered; each is a self-contained PR-able task)

1. **Wire `SurpriseGate` into `run_night`** behind `gate.use_surprise`
   (Phase 2 prerequisite). Include a test with an injected fake scorer.
2. **Judge output cache** keyed by episode + judge settings (Phase 3
   prerequisite). Include a test asserting the second call makes zero LM calls.
3. **`--torch-dtype` CLI flag** (T4 needs fp16; currently config-only).
4. **Re-distill multi-night mode** in `run_nightly.py` (`--cumulative`):
   union of all prior nights' verified SFT data, retrain from base
   (Phase 4 prerequisite).
5. **Harder retention probe**: the 5-sentence `RETENTION_PROBE` is a
   smoke-detector, not an instrument. Add ~50 items sampled from a general
   instruct set (held fixed forever, committed to the repo).
6. **Win-rate eval** (`evals.py`): generation + blind pairwise judge
   (Phase 5, item 6).

## Session hygiene for the next agent

- Stay on `claude/clever-hopper-b6orfo`; push to PR #2.
- Phase 0 before and after every code change; never push red.
- Commit run artifact *summaries* (CSVs, plots, the numbers in docs) — not
  model weights or multi-MB adapter dirs. `runs/sleep/` is gitignored;
  copy anything worth keeping into `docs/` or a tracked report.
- Each backlog item lands as its own commit with its own test, in the order
  the phases need them.
