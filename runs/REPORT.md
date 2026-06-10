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
| 2a — build cache (20k) | shards+head+meta written | ✅ done | `cache/gemma1b_v1`, 20 shards, ~min on L4 |
| 2b — train skill (4k steps, d_k=512, batch 8) | held-out recall≥3, top5≥0.8 | ✅ **PASS** | final held-out recall **+6.39 nats**, top5 **0.98**; gates healthy (no collapse). λ_KL=0.5. |
| 2c — live acceptance (24 facts) | **A1 + A4 PASS** | ✅✅ **A1 PASS, A4 PASS** | **A1**: lift **+16.25 nats**, top5 **1.00**, empty 0.00, wrong-fact **−0.22**. **A4**: retention **1.00** (real disk round-trip into a fresh session). |
| 3 — A2 abstention | A2 PASS without losing A1 | 🔄 in progress | λ_KL=0.5 (skill_v1): A1 PASS, A2 FAIL (neutral_kl 0.10, unrelated 0.17). λ_KL=2.0 (skill_kl2): A2 improved (kl 0.072, unrel 0.33) but **A1 broke** (wrong-fact −1.84) — confirms branch table "raise λ_KL → A1 degrades → capacity is the issue". Both A2 + wrong-fact = key **interference** → fix via wider d_k + deeper encoders (skill_dk1024). |
| 4 — A3 capacity (d_k=512) | interference curve | ✅ curve obtained | recall lift/top5 vs N facts: n1 16.7/1.0, n2 10.1/1.0, n4 9.6/0.50, n8 10.2/0.75, n16 10.5/0.44, n32 6.0/0.09, n128 0.8/0.01. **d_k=512 cleanly holds ~2–8 facts**; interference dominates by n≥16–32. → wider d_k needed for capacity + A2. |

### Step-3/4 notes (2026-06-10)
- **A2 attempts:** λ_KL=0.5 (skill_v1): A1✓ A2✗ (neutral_kl 0.10, unrel 0.17). λ_KL=2.0 (skill_kl2): A2 a bit better but **A1 broke** (wrong-fact −1.84). Read-gate **magnitude feature** (skill_gate, branch-table fix): A1✓ A4✓ but A2 not fixed. So neither λ_KL nor gate-features crack A2 → it's **key interference** (capacity).
- **Metric noise:** A2 `neutral_kl` swung **0.013 ↔ 0.22** for the *same* skill across eval seeds/fact-counts; A1 wrong-fact control −1.0 @ 8 facts vs +0.28 @ 24. The acceptance evals need **more facts** (denoise) before treating A2 as a hard fail.
- **Bugs fixed mid-run:** read-gate magnitude feature overflowed fp32 (pow²) → NaN; `train_skill` didn't seed torch → non-deterministic encoder init. Both fixed.
- **d_k=1024 result (skill_dk1024):** held-out recall 6.72/top5 0.91 (≈ d_k=512). A3 curve **≈ identical** to d_k=512 (n1 10.9/1.0, n4 16.6/1.0, n8 11.3/0.62, n16 6.5/0.31, n64 1.4/0.03) and wrong-fact control *worse* (−1.46). With 24 a2-facts, **neutral_kl 0.014 (PASS)** but unrelated-query 0.17 (FAIL).
- **KEY CONCLUSION:** **wider d_k does NOT extend capacity or reduce interference** → the bottleneck is the **effective dimensionality of the keys** (the encoder maps Gemma hiddens into a limited subspace), not d_k. The branch-table quick levers (λ_KL → breaks A1; read-gate features → no help; d_k → no help) are **exhausted**. A2/A3-at-scale need a **design change: multi-head store and/or richer key representations** (per-fact keys spanning more independent directions), or accept the ~4–8-fact regime where A1/A4 are excellent.
- **A3 curves obtained for two d_k (512, 1024)** — DoD requirement met.
- **Remaining (need a design decision):** multi-head store for capacity/A2; A1b paraphrase; welded-MLP A3 baseline; base scaling.

### Step 5 — capacity ROOT CAUSE + multi-head build (2026-06-10)
Triggered by a 6-agent design panel (see `docs/handoff_capacity_and_multihead.md`): 3 critics
unanimously proved a naive **linear multi-head store is a no-op** (`Σ_h M_h q_h = M q`).
- **Oracle-key diagnostic** (`scripts/memory/diagnose.py`, CPU, no model): the delta-rule store
  recalls **256/256 facts with orthogonal keys** (recall@1 = 1.00) but collapses like the real A3
  curve at **mean key-cosine ≈ 0.6** (n2 .50, n8 .12, n16 .06). ⇒ **the ceiling is KEY OVERLAP,
  not delta-rule erosion** — decorrelated keys fix it; no store/memory-as-tokens redesign needed.
- **Multi-head built (informed/minimal):** math shows M stays `[d_v,d_k]`, so the only change is
  **per-head RMS-norm** (`rms_norm_heads`) + **independent per-head encoders** (`_MultiHeadEncoder`,
  the capacity bet) with `shared_encoder=True` as the predicted-no-op **A/B control**. `n_heads=1`
  byte-identical; 46 CPU tests green. Flags `--n-heads/--shared-encoder`.
- **GPU plan (L4, batch 8, when runtime back):** rebuild cache → Part B real-key geometry (measure
  the actual cosine) → train arms {independent multi-head, shared-encoder control, richer encoder
  `encoder_layers=2`, + key-whitening} → A1/A2/A3. The shared-vs-independent A/B *is* the test of the
  panel's no-op claim; richer-encoder is the cheapest decorrelation lever.

## Bottom line
A frozen Gemma-3-1b **can** be given useful persistent parametric memory: **A1 + A4 PASS** (single-fact cross-session recall +16 nats, retention 1.0). Capacity is the open frontier — limited by key effective-dimensionality, not by d_k or KL or the read gate, so the next step is a multi-head / richer-key design, which is a decision point rather than a sweep.

## 🎯 MILESTONE (2026-06-10): the core question is answered for A1-verbatim.
A frozen Gemma-3-1b + a meta-trained decoupled delta-rule memory recalls a nonce fact
**across a session boundary** (context gone; retrieved from the persisted store) by
**+16 nats, top-5 = 1.0**, with **perfect (1.00) save→reload retention**. The skill/store
(slow/fast-weights) factorization + post-final-norm tied-embedding injection works on a
frozen base. Remaining: A2 specificity (λ_KL sweep), A3 capacity, A1b paraphrase.

## Notes
- Step 0 (2026-06-10): post-final-norm injection of the tied-embedding gold direction steers
  gemma-3-1b cleanly; random-direction control stays flat through λ≤1.0, degrades by λ≥2.
  Whole-approach de-risked at the injection splice. → proceed to Step 1.

## Capacity plan (Fable design review) — C0 PASS, C1/C2 running (2026-06-10)
Fable adjudicated the handoff (`docs/design_review_capacity.md`): **A1+A4 IS the milestone**;
the multi-head no-op proof was wrong (delta-rule heads *do* ensemble) but multi-head is inert
for our clustered geometry; **the lever is ZCA key whitening**. Patch integrated (whitening in
MemorySkill, `diagnose_keys.py`, data-pressure generator knobs) alongside the multi-head build;
50 CPU tests green.
- **C0 (`diagnose_keys` on the real gemma-3-1b cache) — diagnosis PROVEN.** Addressing hiddens:
  **participation ratio = 10** (of 1152), r50=4, top-1 dir = 25% of var, **mean |cos| = 0.77**.
  Predicted A3 top5 (N=4..128): raw 0.67/0.42/0.33/0.25/0.11/0.04 (matches observed collapse);
  **whitened 1.00/1.00/0.94/0.95/0.99/0.98** (holds to 128); oracle 0.92–0.98 (erosion ruled out).
- **C1/C2 running** (`whiten_sweep.sh`, detached, keepalive defeats idle-reclaim): base vs
  `--whiten` A/B on the cache, then data-pressure cache + whiten + 2-layer encoder. C1 gate:
  A3 knee ≥ 32 clean facts (pre-registered from C0's whitened column).

### C1 result — whitening fixes key geometry but reveals a SECOND bottleneck (2026-06-10)
C1 base vs `--whiten` (A3 top5 vs N): base 1/1/1/0.50/0.38/0.19/0.12/0.02; whiten 1/1/1/**0.88**/0.44/0.25/0.09/0.03 (n=1..128). Whitening helped at n=8 but did NOT hit the ≥32 gate, and underperformed C0's clean-simulator prediction (whitened predicted 0.94–0.98 to 128).
- **Why:** `diagnose_keys --skill skill_whiten` shows the TRAINED whitened keys are now *well-spread* — **participation ratio 10→46, mean |cos| 0.77→0.10**. So whitening + training **solved bottleneck #1 (key collision)**.
- **But A3 still collapses → bottleneck #2:** the A3 multifact eval streams N facts through ONE session with filler, and the lr/forget **gate selectivity over long sessions was never trained** (multifact capped at k≤6 — "the data never asked"). C0's simulator did direct writes (no filler/gates/streaming) so it modeled only geometry and over-predicted.
- **C2 (`pressure_whiten`: multifact k≤32 + same-relation hard negatives + 2-layer encoder + whiten) targets exactly bottleneck #2** — running now. If A3 still stalls after C2, the residual is the in-session-write vs standalone-query context mismatch.

### Milestone-3 (real text) — BUILT, pending GPU (2026-06-10)
All five items implemented + committed (CPU); run on the L4 via `scripts/memory/real_text_sweep.sh`
once C2 frees the GPU:
- **#1 real-corpus recall (distribution shift):** `RealCorpusGenerator` (subclasses EpisodeGenerator,
  overrides make_fact → real entity/statement/question/answer; SQuAD via `load_squad_records` or 40
  bundled facts), strict **held-out-by-entity** split; `build_cache --real-corpus`, train on it.
- **#2 A1b content-addressable:** the fact's paraphrase is a natural **question**; `eval_acceptance
  --paraphrase` / `eval_real --paraphrase` query with it, not the verbatim prefix.
- **#3 baselines** (`eval_real.py`): no-memory floor, in-context ceiling, embed+kNN retrieval@1 vs memory.
- **#4 generative EM** (generate_greedy, gold∈output) + neutral-KL **no-regression** at scale.
- **#5 scale+persistence:** ingest the held-out bank one fact at a time, **save→reload between each**
  (deployment regime), query ALL-so-far at checkpoints, report **worst-case** frac_top5 + min_lift to N=256.
Stats protocol: fixed seeded held-out bank (distinct entities), single base; the harness reports the
worse-side metrics. **Highest-risk question (per the goal): does a real-text-meta-trained skill recall
held-out real entities, or does the PR≈10 / cos-0.77 Gemma answer-prefix geometry cap it post-whitening.**
