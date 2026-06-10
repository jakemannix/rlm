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
| 2a — build cache | shards+head+meta written | ⏳ pending | |
| 2b — train skill | held-out recall_lift≥3, top5≥0.8, control flat | ⏳ pending | |
| 2c — live acceptance | A1 + A4 PASS | ⏳ pending | |
| 3 — A2 abstention | A2 PASS without losing A1 | ⏳ pending | |
| 4 — A3 capacity | interference curve per d_k | ⏳ pending | |

## Notes
- Step 0 (2026-06-10): post-final-norm injection of the tied-embedding gold direction steers
  gemma-3-1b cleanly; random-direction control stays flat through λ≤1.0, degrades by λ≥2.
  Whole-approach de-risked at the injection splice. → proceed to Step 1.
