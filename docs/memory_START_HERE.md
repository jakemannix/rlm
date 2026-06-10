# Memory track — START HERE (orientation for a fresh reader)

> Entry point for anyone (esp. a fresh Fable context) picking up the `rlm/memory` work
> cold. It maps what exists, the order to read it, the current state, and the open
> question. **Everything is reproducible from the repo; no prior session context needed.**

## What this is
An experiment in giving a **frozen** LLM (Gemma-3-1b) a **persistent, reload-able memory** via a
**small adapter trained once and then frozen** ("the skill") plus a **mutable store** the skill
writes facts into and reads back — no base retraining, no re-pasting text into the prompt. The
store is saved to disk and reloaded across sessions. Goal: the kind of cheap, persistent memory an
AI agent/assistant would want.

## Status in one paragraph (2026-06-10)
**The core mechanism is proven on real text:** a frozen base + the meta-trained skill + the
persisted store recalls a *held-out real fact* across a destroyed context at **~81% of the
in-context ceiling**, generates the exact answer **~45%** of the time, with **perfect save→reload**.
**The open problem is scale/usefulness:** the current (linear) store reliably holds only **~8–16
facts** before interference; we dissected *why* (it's the write rule + key geometry + streaming, NOT
the store's size — the store could hold thousands) and have cheap fixes in flight. **A proposed
change of target** — from episodic fact recall toward *continual, lossy, semantic compression* on an
MLP substrate — is written up in `continual_memory_direction.md` and is the main thing to react to.

## Read in this order
1. **`design_consult.md`** — the original brief: persistent parametric memory on a frozen LLM.
2. **`design_consult_response.md`** — Fable's original design: the linear delta-rule store + the
   skill + the acceptance tests A1 (cross-session recall) / A2 (abstention) / A3 (capacity) / A4
   (persistence).
3. **`../runs/REPORT.md`** — the full chronological run log: what was built, A1+A4 PASS, the
   milestone-3 real-text results, and the capacity findings. (Long; skim the headers + the 🎯 lines.)
4. **`memory_capacity.md`** — capacity theory + measurements (corrected): the store holds ~`d_k·d_v`
   (thousands), not `d_k`; the binding ceilings are the *update rule*, *key rank*, and *streaming*.
   Reproduce every number with `scripts/memory/capacity_probe.py`.
5. **`memory_capacity_review.md`** — Fable's review of (4): full-vocab re-scoring, the corrected
   "decade ladder" of binding constraints, the milestone-3 metric correction, and the D0–D4 plan.
6. **`continual_memory_direction.md`** — ⭐ **the proposed next direction** (MLP substrate, continual
   compression). **This is the doc to propose on.**
7. Background as needed: `handoff_capacity_and_multihead.md` (the multi-head adjudication),
   `design_review_capacity.md` (the whitening lever), `titans_memory.md` (the original Titans/Miras
   MLP-memory notes — relevant again for the new direction).

## Code map (`rlm/memory/`)
- **Linear-store track (current, proven):** `linear_store.py` (the delta-rule store M + the optional
  match-store M₂), `skill.py` (encoders, gates, whitening, match-gate), `session.py` (live
  ingest/recall + save/reload), `episodes.py` + `real_corpus.py` (synthetic + real SQuAD episodes),
  `cache.py` (frozen-base activation cache), `trainer.py` (meta-training on cached activations).
- **MLP-memory track (Titans/Miras — the new direction's substrate):** `titans.py`, `miras.py`,
  `modeling.py`, `config.py`. This is what the project started from; `continual_memory_direction.md`
  proposes returning to it in the frozen-base + persistent + meta-trained framing.

## Scripts (`scripts/memory/`) — all reproducible
- **Probes (CPU, no model):** `capacity_probe.py` (the capacity claims — value-dim scaling, update
  rule, key rank, readout nonlinearity, lr×rank), `capacity_scaling.py`.
- **Pipeline (GPU):** `build_cache.py` → `train_skill.py` → `eval_acceptance.py` (A1–A4) /
  `eval_real.py` (held-out real recall + baselines + scale) / `diagnose_keys.py` (key geometry).
- **Drivers:** `lr_scan.sh`, `whiten_sweep.sh`, `real_text_sweep.sh`, `capacity_sweep.sh`.
- `scripts/memory/README.md` for run details.

## How to reproduce the headline claims fast
- Capacity theory (seconds, CPU): `uv run python scripts/memory/capacity_probe.py`
- The real-text result + full pipeline needs a GPU (a Colab L4) — see `../runs/REPORT.md` for the
  build_cache → train_skill → eval_real sequence and the numbers.

## The one question to answer
`continual_memory_direction.md` §0/§7: should the target shift from episodic fact recall to
continual semantic compression on an MLP substrate, and if so, *concretely how* (substrate, write
rule, anti-forgetting, smallest experiment, eval)? Propose from a fresh eye.
