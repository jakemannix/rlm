> **Scope:** part of the *parametric-memory* track (frozen-base + meta-trained adapter + mutable linear store) — a SEPARATE line from the active sleep-time / lesson-centric behavioral-memory work (see rlm/sleep/README.md, docs/memory_experiment_status.md).

# Design review: the capacity wall — adjudication of `handoff_capacity_and_multihead.md`

> From Claude (Fable), after pulling the branch, re-running the CPU suite (green),
> reviewing the diffs (mask-gated forget, A1/A4 separation, KL ramp, overflow fix —
> all sound), and **testing the contested claims numerically** (simulators now live
> in `scripts/memory/diagnose_keys.py`; tables below are from those runs).
>
> First, plainly: **A1 + A4 passing is the milestone.** A frozen base recalling a
> nonce fact across a destroyed context at +16 nats with perfect disk-round-trip
> retention is the thesis of the whole brief, demonstrated. Everything below is
> about the second claim (capacity/specificity), which is now a well-posed problem
> with — I believe — a one-line-of-math fix.

## 1. Adjudication of §3 (the multi-head no-op "proof")

The panel's **conclusion is upheld; the proof is wrong**. The algebraic identity
holds for *Hebbian* superposition (`M = Σ v kᵀ/d`, where per-head Grams sum exactly
to the full Gram). The delta rule breaks it: each head's error-correction term uses
that head's *own partial prediction* (`v − M_h k_h`), not the block of the full-space
error, so the recurrences diverge (verified: max read deviation 0.25 on a 12-write
example). And the difference can be *large and favorable*: with **isotropic**
rank-limited keys (r=32, d_k=512, m=128 facts, real-geometry values), measured top-5
went **0.34 (H=1) → 0.87 (H=4) → 0.96 (H=8)** — heads act as an ensemble over key
sub-projections whose interference errors partially decorrelate while the signal sums
coherently. Multi-head is *not* mathematically inert for this store.

But it **is empirically inert for the failure you actually have.** Under
template-clustered keys (25 centroids, ~5% entity energy, noisy queries — the
diagnosed geometry), interference is a *shared, structured bias*: every head's
projection contains the same dominant centroid component, the errors are correlated
across heads, and the ensemble averages nothing away:

```
template-clustered keys, top-5 vs m:        m=4    8    16    32   128
  H=1, raw                                  0.95  0.90  0.81  0.57  0.20
  H=8, raw                                  0.95  0.90  0.81  0.57  0.22
  H=1, ZCA-whitened                         1.00  1.00  1.00  1.00  1.00
  H=8, ZCA-whitened                         1.00  1.00  1.00  1.00  1.00
```

Two more things this table settles. The raw H=1 row **reproduces the observed A3
curve's shape** (clean ≤8, degraded by 16, collapsed by 32–128) from nothing but the
clustered-key assumption — strong support for the diagnosis. And **whitening alone
takes the same store to a clean 128**, with noisy queries, no heads, no retraining of
anything structural. The panel's instinct ("the real intervention is decorrelated
keys, not head count") was exactly right; the cheapest decorrelator is closed-form.

Erosion (handoff Q4): ruled out as the binding constraint. Full-rank keys at
d_k=512 hold m=256 at top-5 = 1.00 in the same simulator — and the observed
d_k-invariance is itself anti-erosion evidence (the erosion exponent is m·θ/d_k;
doubling d_k would have visibly moved an erosion-limited curve).

## 2. The mechanism, stated once

Capacity of this store is governed by the **spectrum of the addressing population**,
not d_k. Post-norm hiddens at the answer-prefix position are dominated by shared
template/position directions; the entity-discriminative components are a small
fraction of variance. A linear (or orthogonal-init) encoder preserves that skew, so
distinct-fact keys collide; widening d_k embeds the same low-effective-rank cloud in
a bigger space (identical Gram ⇒ identical curve ⇒ your d_k=1024 null). ZCA
whitening equalises the spectrum so the entity tail carries addressing. Note the
information-theoretic point is gentler than "capacity ≤ rank": retrieval against a
*discrete codebook* (the tied embedding) error-corrects at readout, so even m > r is
recoverable once interference is whitened from structured bias to isotropic noise.

Two caveats, honestly held: whitening amplifies small directions *including noise*
(`fit_zca`'s `eps_frac` is the dial; sweep {1e-2, 1e-3, 1e-4}, and expect the
paraphrase regime to want more shrinkage), and whitening cannot conjure entity
information that is truly absent at the splice. The diagnostic distinguishes:
if raw-hidden within-relation cosine ≈ across-relation cosine (no entity signal at
all), branch to earlier-layer keys (keep the proven post-norm *injection*).

## 3. Revised plan (replaces the handoff's §4 ordering)

**Step C0 — diagnose + predict (hours, no training).** `scripts/memory/diagnose_keys.py
--cache cache/gemma1b_v1 --skill runs/skill_v1.pt` reports: raw-hidden and
trained-key spectra (participation ratio, r50/r90), within- vs across-relation
cosines (needs a cache rebuilt with the relation-labels patch — included; the old
cache still gives spectra + overall cosines), the **predicted A3 curve raw vs
whitened from the measured covariance against the real head**, and the oracle-key
ceiling. Gate: raw prediction matches the observed knee (±1 octave) and oracle is
clean. If yes, the diagnosis is proven and the whitened column is the pre-registered
expectation for C1.

**Step C1 — whitening (the decisive experiment).** Retrain on the existing cache
with `train_skill.py --whiten` (fits ZCA on ~50k cached addressing vectors, installs
it in the skill; serializes with it). Run A3. Gate: knee moves to ≥32 clean facts.
Sweep `--whiten-eps` if paraphrase or noise degrades A1.

**Step C2 — training pressure (do with or right after C1).** The current data never
*asks* for capacity: multifact caps at k≤6 and same-relation hard negatives occur
~4% by chance, so the observed clean range (~4–8) suspiciously equals the demanded
range. Rebuild the cache with the new knobs: `--multifact-k 2,32
--same-relation-control-prob 0.5 --paraphrase-prob 0.25` (the last also serves A1b).
Retrain (`--encoder-layers 2` is the right companion here — nonlinear capacity is
only used if the data demands it). Gate: A3@32 worst-case-leaning (fraction of facts
at top-5, not mean lift) plus the A1b-paraphrase number reported separately.

**Step C3 — abstention done right (after geometry).** Adopt the panel's critique:
current A2 conflates abstention with cross-fact specificity. Redefine **A2a** (true
abstention: neutral text + queries from held-out relation templates never in
`RELATIONS`) and fold cross-fact specificity into A3 as the wrong-fact arm vs N. For
the mechanism, the read-magnitude feature was the right idea on the wrong statistic;
the principled feature is a **match score**: a second auto-associative store
`M₂ += (k − M₂k)kᵀ/d` written with the same gates, giving `qᵀM₂q ≈ Σ cos²(q, kⱼ)` —
"is q near anything I stored" — parametric, serializes with M, ~15 lines. Sequence
it here because match scores are uninformative while all keys collide (which is why
the magnitude feature couldn't work).

**Step C4 — only if C1+C2 stall:** heads (now genuinely useful — post-whitening keys
are near-isotropic, the regime where the ensemble effect measured 0.34→0.96), then
earlier-layer keys, then routed/product-key stores. Implementation note if heads are
built: use the self-consistent convention (per-head `1/d_h` scaling, *mean* of head
reads) — the naive `1/d_k`-per-head variant over-deposits repeated writes by H.

**Statistical protocol (adopt before any further PASS/FAIL):** the handoff's
critique is correct and the swings prove it. Commit a fixed held-out fact bank
(seeded JSON, ≥64 facts), evaluate A1/controls over all of it with ≥3 eval seeds,
report mean ± sd, gate on the worse side of the interval. `neutral_kl` stays but
over ≥64 neutral sequences.

## 4. Answers to the ten questions, compactly

1. **Accepted proof of the hypothesis:** C0's triple — measured spectrum (PR, r90),
   same- vs cross-relation cosine gap, and the raw-prediction matching the observed
   knee while the oracle ceiling is clean. Encoder vs splice: decided by whether the
   *raw hidden* has a same/cross cosine gap (signal present → encoder/whitening
   problem; absent → splice problem → earlier-layer keys).
2. **Multi-head:** not a no-op in general (proof above), but a no-op for clustered
   interference — so deferred behind whitening, after which it becomes live again.
3. **Highest-leverage first:** whitening (C1), with data-pressure + encoder depth
   (C2) as the same sprint. Routing/product-key and memory-as-tokens stay parked.
4. **Overlap vs erosion:** overlap. Erosion ruled out at these m (table + d_k-null
   argument); the matrix store is *not* disqualified for O(10²) — the whitened
   simulation holds 128 with the real head.
5. **A2 redefinition:** yes — A2a as true abstention (held-out templates + neutral
   text), cross-fact specificity moves into A3. Keep `neutral_kl` only under the
   statistical protocol.
6. **Match-score gate:** yes, via the auto-associative `M₂` (C3), replacing the
   magnitude feature rather than stacking on it.
7. **"Done":** A1+A4 *is* the proof of thesis — say so without hedging. Capacity is
   milestone 2 with its own gates (A3@32 worst-case-leaning, A2a, A1b ≥ 3 nats).
   RLM integration stays gated behind milestone 2, per the scope guard.
8. **Stats protocol:** as above; pre-register C1's expectation from C0's whitened
   column.
9. **Welded-MLP baseline + oracle-key:** oracle-key is now free inside C0; the
   welded-MLP A3 arm is worth one overnight run *after* C1 so the comparison is
   against the fixed system, not the broken one.
10. **Earlier-layer keys:** branch C4, triggered by C0's no-entity-signal outcome,
    not before.

## 5. What changed on the branch with this review

`scripts/memory/diagnose_keys.py` (C0, smoke-tested end-to-end on synthetic
clustered data); ZCA whitening in `MemorySkill` (+ `fit_zca`, persistence,
`train_skill.py --whiten/--whiten-eps`, tests); episode-generator capacity-pressure
knobs (`multifact_k`, `same_relation_control_prob`) with `build_cache.py` flags and
relation labels recorded in new caches (tests); this document. CPU suite green
(now 30 memory-track tests + the legacy 19).

One nit from the code review, no action needed: a fully-masked chunk zeroes the
momentum buffer rather than carrying it (`S ← 0·S − 0`). `M` is exactly preserved,
which is the contract that matters, and resetting momentum across masked gaps is
arguably the *better* semantics — but the comment says "exact no-op," so make the
choice explicit in the docstring when next touching the file.
