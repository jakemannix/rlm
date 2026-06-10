# Memory-track run report

Durable progress/resume anchor for the Fable ladder (`docs/design_consult_response.md` §4).
Run on a Colab L4 (gemma-3-1b-it). Full per-step JSON lives on the VM under `/content/rlm/runs/`
(regenerable); this file records verdicts + key numbers so the work is resumable after the
ephemeral VM dies.

## Environment / landmines resolved
- gemma-3-1b-it config: **tie_word_embeddings=True**, **final_logit_softcapping=None**,
  hidden_size=1152, vocab=262144. (The tied-embedding readout the design relies on is valid;
  no softcap to compress injection.)
- Branch `claude/titans-memory-module-YCkCl`, starter at commit bc8c293.

## Ladder status

| Step | Gate | Status | Key numbers |
|---|---|---|---|
| 0 — oracle injection | mean Δgold≥3, top5≥0.9, rand flat | ✅ **PASS** | λ=0.5: +12.84 nats, top5 1.00, rand +0.15; λ=1.0: +13.80, top5 1.00, rand +0.12. Passing λ∈[0.5,1.0]. |
| 1 — oracle write/read | lift≥3, top5≥0.9, controls flat | ⚠️ **mechanism validated**, strict gate not met | With design-faithful flags (`--tie-kq --bare-ingest --chunk-size 1`): recall lift **0.77→9.65 nats**, top5 0.92@scale16, empty-control exactly 0.00 (A4 mechanics sound). Strict gate fails only on cross-fact **interference** (wrong-control −1→−50 as scale grows) + top5<0.9 at low scale — the per-fact-variance the branch table names, which Step-2 training (selective keys/gates) exists to fix. Injection×bind×persist all certified. |
| 1.5 — review fixes | CPU suite green | ✅ done | Adversarial review of the starter (17 agents) found 9 confirmed blocker/majors; fixed: forget/momentum **mask-gating** (filler no longer erodes facts), write-gate **stats logging**, A1-from-**in-session** (was conflated with A4), **recall-only warmup + KL ramp** (gate-collapse mitigation), **checkpoint/resume** + cache **overwrite-guard**. 45 CPU tests green (added forget-mask + warmup regression tests). |
| 2 — pipeline smoke (256 eps, 200 steps, batch 8) | runs end-to-end on real Gemma | ✅ **validated + signal** | Held-out recall **+4.57 nats**, top5 0.75, control −0.19; gates alive (no collapse). LIVE acceptance: A1 lift **+15.67 nats**, empty-control **exactly 0.00**, **A4 retention 1.00** (perfect save→reload). Only short of gates on top5 (0.75<0.8) + wrong-fact control (0.97) = specificity → full run's job. **OOM at batch 32** (BPTT graph of M[B,1152,512]/chunk + 262k-vocab logits); use batch 8 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`. |
| 2a — build cache (20k) | shards+head+meta written | 🔄 building (detached, ~30–60 min) | `cache/gemma1b_v1`, log `runs/build_cache.log` |
| 2b — train skill (4k steps) | held-out recall_lift≥3, top5≥0.8, control flat | ⏳ pending | |
| 2c — live acceptance (24 facts) | A1 + A4 PASS | ⏳ pending | |
| 3 — A2 abstention | A2 PASS without losing A1 | ⏳ pending | |
| 4 — A3 capacity | interference curve per d_k | ⏳ pending | |

## Notes
- Step 0 (2026-06-10): post-final-norm injection of the tied-embedding gold direction steers
  gemma-3-1b cleanly; random-direction control stays flat through λ≤1.0, degrades by λ≥2.
  Whole-approach de-risked at the injection splice. → proceed to Step 1.
