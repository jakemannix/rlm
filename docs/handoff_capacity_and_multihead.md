> **Scope:** part of the *parametric-memory* track (frozen-base + meta-trained adapter + mutable linear delta-rule store) — a SEPARATE line from the active sleep-time / lesson-centric behavioral-memory work (see rlm/sleep/README.md, docs/memory_experiment_status.md).

# Handoff: frozen-base memory — status, the capacity wall, and a challenge to the multi-head plan

> **Status: working brief for a design review (Fable).** Self-contained; the repo is at
> branch `claude/titans-memory-module-YCkCl`. Written by Claude after executing Fable's
> ladder (`design_consult_response.md`) to A1+A4 PASS and hitting a capacity wall on A2/A3.
> **Headline for the reviewer: our own adversarial analysis says the prescribed next step
> (a multi-head store) is a mathematical no-op for a linear store. Please adjudicate §3–§4.**

## 1. Where we are (verified against code + `runs/REPORT.md`)
A *frozen* Gemma-3-1b + a meta-trained **decoupled delta-rule store** (zero-init `M`, gates as
the trained skill, value = answer-shifted **tied** embedding, post-final-norm injection):
- **A1 PASS** — single-fact cross-session recall **+16.25 nats, top-5 1.0**, empty-control 0.00, wrong-fact −0.22.
- **A4 PASS** — retention **1.00** through a real disk round-trip into a fresh session.
- **A2 FAIL** — `unrelated_query_unchanged` ~0.17 (gate 0.9); `neutral_kl` noisy (0.013↔0.22 same skill, gate 0.02).
- **A3 capacity** — d_k=512 **and** d_k=1024 both cleanly hold only **~4–8 facts**, collapse by n=16–32 (curves in REPORT).

Ladder: Step 0 (oracle injection) PASS; Step 1 (oracle write/read) mechanism validated (needs `--tie-kq --bare-ingest`); Step 2 train→eval = the A1/A4 PASS. Bugs fixed en route (11): forget/momentum mask-gating (filler was eroding facts), A1-vs-A4 conflation in eval, recall-only + KL-ramp warmup (gate collapse), fp32 overflow in the read-gate magnitude feature (`0·inf=NaN`), non-deterministic skill init, cache overwrite-guard/checkpoint, +older-track device/dtype/meta-grad fixes. 45 CPU tests green.

## 2. The capacity finding
Widening **d_k 512→1024 changed nothing** (identical A3 curve, wrong-fact control *worse*), and λ_KL↑ breaks A1, read-gate-magnitude feature didn't help. Inference: capacity is bounded by the **effective dimensionality / rank of the keys**, not d_k. Keys = a **single orthogonal-init `Linear(1152→d_k)`** applied to Gemma's post-final-norm hidden at the answer-prefix token, then RMS-normed. Across facts that share one of ~25 relation templates, those hiddens are **anisotropic / low-effective-rank** (a dominant shared template direction), so distinct-fact keys have high pairwise cosine and the delta-rule read returns ~the mean stored value regardless of query.

## 3. ⚠️ The prescribed multi-head store is (almost certainly) a mathematical no-op
Three independent adversarial critics **unanimously** concluded multi-head does **not** address the bottleneck, with a proof:

> The store is **linear**. Splitting `d_k` into H heads `k=[k_1…k_H]` with per-head maps `M_h` and a **summed** read gives
> `Σ_h M_h q_h = Σ_h (Σ_j v_j k_{j,h}ᵀ/d) q_h = Σ_j v_j (Σ_h k_{j,h}·q_h)/d = Σ_j v_j (k_j·q)/d = M q`.
> i.e. **the multi-head read equals the single-head read for the same encoder output.** Unlike attention, there is no per-head softmax/nonlinearity to break the equivalence. The full Gram `KKᵀ` equals the sum of per-head Grams, so interference is unchanged.

So multi-head helps **only if** the per-head encoders are forced onto **complementary subspaces** — i.e. the real intervention is *decorrelated/independent key encoders*, not "head count." A single shared encoder reshaped into heads is provably useless (and is a good A/B *control* to confirm the diagnosis). This is consistent with the empirical d_k null.

## 4. What the panel recommends instead (in priority order)
1. **DIAGNOSE FIRST (cheap, gates everything):**
   - **Effective-rank of the keys** — SVD / participation-ratio of stacked `key_enc(h_i)` and of the raw `h_i` across many nonce facts; pairwise-cosine histogram at n=4 vs n=32. Directly measures `d_eff`. If `d_eff≈50`, *that is* the capacity number and no store-side change moves it. (~20 lines on the existing cache.)
   - **Oracle-key A3 ceiling** — inject perfectly orthogonal/one-hot per-fact keys (bypass the encoder) and run the A3 capacity curve. Upper-bounds the matrix store's capacity **independent of Gemma's geometry**, separating "keys collide" from "delta-rule erosion."
2. **Richer / deeper / nonlinear key encoder** — `encoder_layers≥2` (GELU MLP) is *already supported* but was **never trained** (every run used `encoder_layers=1`). A nonlinear map can un-cluster nearly-collinear `h_i`; a linear map provably cannot. Cheapest *real* untested lever, on the existing cache, d_k=512.
3. **Whiten/decorrelate keys** — subtract a learned shared-mean direction / project out top principal components of the cached key distribution / key LayerNorm. (One critic's sim took n=32 recall 0.06→1.0 after whitening.)
4. **Earlier-layer keys** — splice keys/queries from a richer pre-collapse middle layer while keeping the proven post-final-norm *injection*.
5. **Caveat — delta-rule erosion may be the real ceiling, not geometry:** even with perfectly distinct keys, m writes retain ~`exp(−m·θ/d_k)` of an earlier association (error-correcting writes erode overlapping content). If the n=16–32 collapse is the *erosion term*, none of the above fully fix it and the answer is orthogonalized values / a non-eroding **memory-as-tokens** store (design §Q2, never built). The oracle-key probe (1) distinguishes these.

## 5. Methodological issues the reviewer should rule on
- **A2 is double-counted with A1.** `run_a2`'s "unrelated query" ingests fact A then scores fact B's *verbatim prompt* — structurally identical to A1's wrong-fact control. "Abstention" is conflated with "cross-fact non-interference (capacity)." True abstention should test a query whose key is **far from all stored keys**.
- **Abstention mechanism ceiling.** The read gate keys off read *magnitude*, but an interfering unrelated query yields a read of comparable magnitude to a true recall, so magnitude may be **fundamentally insufficient**. Principled abstention likely needs an explicit **match score** (e.g. max cosine of q to written keys) the gate can threshold.
- **Under-powered metrics.** Single seed; A1 over 24 facts, A2 over 12; wrong-fact control swung −1.0 (8 facts)↔+0.28 (24), neutral_kl 0.013↔0.22 same skill. Needs a fixed shared held-out fact bank + mean±CI vs pre-registered thresholds.
- **A1 is verbatim-prefix, not content-addressable.** The query is a literal prefix of the ingested statement (so query-hidden ≈ ingest-hidden). **A1b (paraphrase) was never evaluated.** Is verbatim recall really "memory," or positional prefix matching? A1b should arguably be a *core* gate.
- **A3 protocol.** `run_a3` ingests N facts in one multifact session; the deployment regime is multisession-with-reload (forget-gate decay in play). And capacity should arguably be *worst-case* (all-facts top5≥0.9) not mean lift.

## 6. Open questions for the reviewer
1. What diagnostic do you accept as **proof** of the effective-dim hypothesis, and is the bottleneck the **encoder** or the **post-final-norm splice point** itself?
2. Multi-head: do you agree it's a no-op for a *linear* store, and that the real lever is **inter-head decorrelation** / independent encoders? If you still want heads, what nonlinearity/penalty makes them non-redundant?
3. Highest-leverage capacity fix **first**: richer encoder vs key-whitening vs explicit **routing/product-key/hash-to-slot** (deterministic near-orthogonal keys) vs **memory-as-tokens**?
4. Is the n=16–32 collapse **key overlap** or **delta-rule erosion**? (Oracle-key probe settles it.) Does the O(10²)-fact target argue against the matrix store entirely?
5. Redefine A2 to separate true abstention from capacity? Operational definition of "defer to base"? Keep `neutral_kl` as a gate given its noise?
6. Add an explicit **retrieval-confidence / match-score** to the read gate as the principled abstention mechanism?
7. **What does "done" mean?** Is A1+A4 (single/handful-fact cross-session recall, validated) the proof-of-thesis, with O(10²)-fact capacity as a *separate* research problem — or is multi-fact non-interference required to claim the approach works?
8. Statistical protocol (facts/seeds/CI/shared fact bank) before any further PASS/FAIL is trusted?
9. Should the **welded-MLP baseline** (`titans.py`) and the **oracle-key ceiling** be run to attribute the collapse (decoupling-cost vs delta-rule vs Gemma geometry)?
10. Splice keys/queries from an earlier layer (richer rep) while keeping post-norm injection — worth trying before any store redesign?

## 7. Reproduce / pointers
- Modules: `rlm/memory/{linear_store,skill,session,episodes,cache,trainer}.py`; scripts `scripts/memory/{oracle_injection,oracle_write_read,build_cache,train_skill,eval_acceptance}.py`; `runs/REPORT.md` (numbers); design `docs/design_consult{,_response}.md`.
- Trained skills + cache live on the ephemeral Colab L4 (`cache/gemma1b_v1`, `runs/skill_v1.pt` etc.); regenerable: build_cache (~min) → train_skill (~20 min) → eval_acceptance.
- CPU tests: `tests/test_{linear_store,episodes,trainer_toy,titans_memory}.py` (45, green).
