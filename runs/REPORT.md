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
| 1 — oracle write/read | lift≥3, top5≥0.9, controls flat | ⏳ pending | |
| 2a — build cache | shards+head+meta written | ⏳ pending | |
| 2b — train skill | held-out recall_lift≥3, top5≥0.8, control flat | ⏳ pending | |
| 2c — live acceptance | A1 + A4 PASS | ⏳ pending | |
| 3 — A2 abstention | A2 PASS without losing A1 | ⏳ pending | |
| 4 — A3 capacity | interference curve per d_k | ⏳ pending | |

## Notes
- Step 0 (2026-06-10): post-final-norm injection of the tied-embedding gold direction steers
  gemma-3-1b cleanly; random-direction control stays flat through λ≤1.0, degrades by λ≥2.
  Whole-approach de-risked at the injection splice. → proceed to Step 1.
