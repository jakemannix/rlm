# Review of `memory_capacity.md`: corrections verified, attribution fixed, the decade plan

> Fable, 2026-06-XX. I re-ran the contested probes under harsher protocols before
> adjudicating (full-vocabulary argmax against an anisotropic clustered codebook,
> not among-N random-unit values; harness now in `capacity_probe.py fullvocab_arm`
> — run it with the real `head.pt` as D0). Verdict up front: **both ⚠️ corrections
> are right and survive the harsher scoring; the §3 attribution table is wrong in
> one important place; milestone-3 is the win it claims to be, with one metric
> (paraphrase lift) that must be specificity-adjusted before it is quoted.**

## 1. The two corrections — confirmed, and they correct *my* numbers

The "capacity ≈ d_k/3" planning heuristic came from my original design doc, and it
was a *reconstruction*-error bound applied to a *retrieval* problem. Against a
codebook, crosstalk only has to lose an argmax, not vanish: capacity scales like
d_k·d_v (divided by log-vocab-ish factors), which is why probe A grows with d_v.
Under my harsher re-run (32k anisotropic clustered codebook, full-vocab argmax):
Hebbian d_k=256 still holds **N=2000 at 0.98 top-1** — the among-N protocol was
*not* materially inflating this regime, which surprised me and strengthens the
claim. Run D0 with the real Gemma head to make it load-bearing; my prior is now
that it holds.

Correction #2 is exact, with a sharper mechanism than the doc states: the delta
rule at θ=1 is a **recency window of width ≈ d_k/θ̄**. Probe B's 0.13 at N=2000,
d_k=256 *is* 256/2000 — the store recalls the last ~d_k writes and has actively
erased the rest. And note what follows: **at θ→0 the delta rule IS Hebbian** (the
correction term is second-order in θ). So "we chose the wrong update rule"
overstates it — we chose a gate-parameterized *family* whose init point (θ=1,
right for one-shot A1) is wrong for large N. The fix may be five lines, not a new
rule: a store-load feature into the existing lr gate (log‖M‖_F or a write
counter), trained on large-N episodes, lets the meta-learner discover θ(load) —
high while empty, Hebbian-like as it fills. Run the fixed-θ and Hebbian arms as
the doc proposes, but run the **learned θ(load)** arm beside them; my bet is on it.

## 2. One attribution fix: key rank is *not* "the binding cap today" — and it binds harder than probe C says

The §3 table and §1 bullet 3 conflict with the doc's own evidence. Probe C says
rank-46 supports 2000 facts at 0.79 — if that were the binding cap, the observed
ceiling would be hundreds, not 8–16. The observed ceiling is set by **streaming
dynamics** (C1/C2 proved keys are now well-spread, cos 0.10/PR 46, yet streamed A3
still collapses at ~8 while the same components hold ~10² static). Meanwhile my
clustered-codebook re-run moves probe C the *other* way: rank-46 at N=2000 scores
**0.28**, not 0.79 — against a clustered codebook, rank-limited crosstalk aligns
with codebook clusters and confuses much earlier. So the corrected ladder, one
binding constraint per decade:

| Regime | Binds | Evidence | Lever |
|---|---|---|---|
| **8–16 → ~10² (streamed)** | streaming dynamics: write/query context mismatch + whole-statement & filler erosion | C1/C2 + static-vs-streamed gap | D1 below |
| **~10² → ~10³ (static then streamed)** | update rule at θ≈1 (recency window) | probe B; 0.12 confirmed full-vocab | θ(load) gate policy / Hebbian+decay |
| **~10³ → ~10⁴** | key rank (≈46 today; binds earlier than probe C implied) | rank-46 full-vocab 0.28@2000 | earlier-layer keys, nonlinear encoders, multi-token keys |
| **beyond / orthogonal** | linear-readout superposition | probe D | parametric softmax — see §4 |

## 3. Milestone-3: real, with one metric that needs surgery

Verbatim is the headline and it is clean: +8.24 over floor = **81% of the
in-context ceiling with the context destroyed and the store persisted**, wrong-fact
0.06, A4 = 1.0. Quote it.

The paraphrase number must not be quoted as-is. **Memory (+11.13) beating its own
in-context ceiling (+8.05) is diagnostic, not impressive** — a *specific* memory
cannot beat handing the model the actual passage; only a *prior-shifting* effect
can, by sharpening the answer-type distribution beyond what context provides. The
wrong-fact control (4.77) says ~⅔ of the paraphrase lift is exactly that:
question-shaped prompts open the read gate, and the smeared read boosts
plausible-answer tokens generically (SQuAD answers share type structure — names,
dates, numbers). Three changes, all cheap:

1. **Primary A1b metric = specific lift** `= lift(target) − lift(wrong-fact)`.
   Today that's ≈ 2.4 nats — real, and honest.
2. **Invariant: memory ≤ in-context ceiling.** Any violation flags prior-shifting;
   wire it into `eval_real` as a printed warning, not a gate.
3. The same pathology is the growing `neutral_kl` (0.19→0.45 with N): the gate
   cannot tell "relevant read" from "big read." This is the **match-score store**
   from `design_review_capacity.md` C3, still unbuilt and now the top
   specificity lever: a second auto-associative matrix `M₂ += (k − M₂k)kᵀ/d`
   written with the same gates; `qᵀM₂q` ≈ "is q near anything I stored" feeds the
   read gate. Parametric, serializes with M, ~60 lines including tests.

## 4. The fork in §4 is a false binary: the parametric softmax exists

"To exploit the softmax you must keep k/v pairs = slot/retrieval memory" skips the
option that preserves every property on the doc's own list: an **online-written
product-key / slot table** (Lample-style PKM, but with the value table written at
inference). Fixed parameter budget (e.g., 16–64k slots × d_v ≈ 75–300 MB),
sub-key product addressing or learned hashing over the *whitened* encoder output,
top-k routed **writes** (scatter delta into selected slots) and softmax-weighted
reads. Persisted-in-weights ✓, no text at query time ✓, fixed footprint ✓,
editable ✓, O(top-k) reads ✓. Probe D already measured the readout side; the
experiment is making the pairs parametric. This is the 10³–10⁴ architecture that
does **not** concede to retrieval, and it should be D4 — built only if D1–D3
stall, but designed now so stalling has a destination.

And the honest answer to Q5 (where does the dense store win): O(1) read cost
independent of N, fixed footprint, end-to-end differentiability, and —
unique to superposition — facts stored in a *shared* geometry that can compose
and generalize across items. That regime is the **per-user / per-project working
set, ~10¹–10²** — which is precisely the agentic-memory regime that motivated
this project. "Dense parametric memory is the right organ at working-set scale,
with a measured 81%-of-ceiling recall and perfect persistence; PKM extends it
parametrically; retrieval wins beyond" is a defensible, publishable claim that
does not require winning at 10⁴.

## 5. The plan (supersedes §6's ordering; L0/L1/L2 attribution ladder retained as the instrument)

**D0 — probe hygiene (hours).** `capacity_probe.py cache/<dir>/head.pt` re-scores
A–C against the real head, full-vocab (arm added this commit). Locks the theory
numbers. Also rerun probe D's softmax arm with real W for the PKM design point.

**D1 — the streaming decade (8–16 → ≥64 streamed). Run first; it is the binding
constraint and mostly eval-side.**
- **D1a** (`diagnose_keys`, pairing block added this commit, needs one cache
  rebuild for `answer_mask`): same-fact vs cross-fact cosine between the key
  *written in session context* and the key the *standalone query* produces. A
  collapsed gap ⇒ context mismatch is the gun ⇒ D1d. A healthy gap ⇒ erosion/
  selectivity ⇒ D1b/c suffice.
- **D1b** (`eval_real --write-mode answer`, added this commit): oracle
  selectivity — write only the ~2–5 positions whose *target* is an answer token.
  No retraining; if the streamed curve jumps (my prediction: 8 → 30–50), the gap
  is erosion from whole-statement + filler writes and the production fix is a
  learned gate that reaches the same sparsity.
- **D1c**: the learned version — surprisal + store-load lr-gate features (exact
  spec already in `real_data_plan` history / §1 above; default-off, zero-init),
  retrain on the SQuAD cache, target θ̄_filler ≤ 0.01 at unchanged recall.
- **D1d** (only if D1a fires): write/query key-consistency auxiliary loss
  (InfoNCE over the cache's paired keys — the pairs are already cached).

Gate: `eval_real --scale` worst-case frac_top5 ≥ 0.8 @ 32 and min_lift > 0 and
neutral_kl flat in N, ≥3 seeds.

**D2 — the update-rule decade.** Arms on L0/L1/L2 with real W: θ=1 (control),
θ=0.2, Hebbian+forget, **learned θ(load)**. Large-N training episodes (SQuAD has
3,883 train facts; multifact k up to 256). If the single store stalls, the
two-timescale variant (fast delta window + slow near-Hebbian store, summed reads
— complementary learning systems in ~30 lines) before anything exotic.
Gate: L2 ≥ 0.8 worst-case @ 128.

**D3 — rank lifting**, now known to matter from mid-10² (my re-run): earlier-layer
keys first (cache-builder variant storing layer ≈⅔ hiddens — likely the
highest-leverage single change, per the doc's own Q2), then encoder depth +
decorrelation, then multi-token keys. The "hard rank ceiling of a frozen base"
worry in Q2 dissolves once keys stop being a single post-norm position: available
rank multiplies across positions and layers; 46 is an artifact of the extraction
choice, not the base.

**D4 — parametric softmax (PKM, §4)** if D2/D3 stall before 10³, at matched bytes
vs dense on the L2 protocol.

**S (parallel, small, unblocks "useful"):** match-score store M₂; specific-lift as
the primary A1b metric + the ceiling invariant in `eval_real`; abstention
episodes drawn from cross-title SQuAD questions (data already exists).

**Sequencing:** D0 + D1a/D1b this session (no retraining), S in parallel, D1c/D1d
retrain next, D2 sweep after, D3/D4 gated. The doc's "fastest informative
experiment = update-rule swap" is right for the *static* decade but wrong for
*today*: the deployed ceiling is streamed, and D1b is cheaper (zero retraining)
and aimed at it.

## 6. Answers to the six open questions, on the record

1. **Update rule:** θ(load) as a learned gate policy in the existing architecture;
   fixed-θ and Hebbian as the controls. The "rule" and the "schedule" are the same
   lever because delta→Hebbian as θ→0.
2. **Key rank:** liftable — single-position post-norm extraction is the artifact;
   earlier-layer + multi-position keys multiply rank. No intrinsic frozen-base
   ceiling worth fearing before 10³⁺.
3. **Value side:** not binding through 10³ on my clustered-codebook runs; revisit
   only if D0-real shows codebook crowding. Untying the values sacrifices the
   closed-form readout — don't, until forced.
4. **Streaming:** decomposed into D1a/b/c/d with a measurement that attributes the
   gap before any retraining.
5. **Where parametric wins:** working-set regime (10¹–10²), O(1)/fixed-footprint/
   composable — claim it explicitly; stop implicitly competing with RAG at 10⁴.
6. **Architecture by scale:** dense (≤10², proven) → dense + θ(load) + rank
   (10²–10³) → online-written PKM (10³–10⁴, still parametric) → retrieval beyond.

## 7. What changed on the branch with this review

`capacity_probe.py` full-vocab/real-head arm (D0); answer-span segmentation with
target-aligned `answer_mask` through generator → cache → session, plus
`eval_real --write-mode {all,fact,answer}` (D1b, zero retraining required);
`diagnose_keys` write-vs-query pairing block (D1a; needs one cache rebuild to
populate `answer_mask`). 325 tests green including the SQuAD/real-corpus path.

## 8. Addendum: adjudication of 79b4a0f (probe E — "lower lr is the dominant ~10× lever")

This commit landed independently of the review above and **converges with §1's
prediction** (θ→0 ⇒ delta→Hebbian; the update-rule fix is a θ policy). Probe E is
internally consistent — I re-derived its numbers: at rank 46, θ=0.03, N=1024 the
expected early-item retention is exp(−Nθ/d_eff) = exp(−0.67) ≈ 0.51, matching the
measured 0.60 vs Hebbian's 0.98. The finding stands. Three amendments before it
drives a GPU run:

**A1 — the probe metric is scale-invariant in θ; deployment is not.** Probe E
scores `argmax(read · V)`, which is unchanged if the read shrinks 33×. Deployment
scores `Δlogit = ⟨W_gold, g·out_proj(read)⟩` against the base's prior, and a
one-shot write at θ deposits a read of magnitude ∝ θ — so an **eval-only
`max_lr` drop on the existing skill will collapse the lift** (verified in a toy:
θ=0.03 at unit gain ≈ 1/33 the injected nats; restored exactly at gain 1/θ).
The §1 bullet's "one-line drop … no retraining of structure" is therefore the
wrong takeaway; the doc's own Q1 experiment (**retrain** at low `max_lr` — CE
learns the compensating read gain automatically) is the right one. Do not let a
quick eval-only sweep "refute" the lever.

**A2 — the gain that restores recall amplifies crosstalk identically.** Signal
and interference in the read both scale with θ·g, so injection-SNR is
θ-invariant; the capacity win comes solely from reduced *erosion*. Consequence:
at low θ + compensating gain, wrong-fact lift and neutral-KL pressure are **at
least as large as today** — the match-score store M₂ (§3) moves from
"specificity fix" to **prerequisite of the low-lr arm**. Schedule S before or
with the low-θ retrain, and gate the arm on specific-lift, not raw lift.

**A3 — absolute Ns in probe E are among-N / isotropic-value optimistic.** Under
full-vocabulary argmax against a clustered codebook (the §2 re-runs; the
`fullvocab_arm` added here), rank-46 numbers shrink ~2–4× (0.79→0.28 at N=2000).
The θ-lever survives — erosion is orthogonal to codebook geometry — but "rank-46
holds ~1000" should be read as "high hundreds, before streaming." One nuance in
probe E's favor: isotropic-within-rank keys are a *good* model of
**post-whitening** keys, so probe E ≈ the post-whitening static ceiling under
the optimistic readout. Re-score E with the real head (one command now:
`capacity_probe.py cache/<dir>/head.pt`).

**The design rule, replacing "~0.06":** for a uniform policy and a target
lifetime of N facts, the optimum tracks **θ* ≈ 0.3 · d_eff / N** (CPU-verified:
recall-vs-θ peaks at ≈0.054 for N=256 and ≈0.013 for N=1024 at d_eff=46). The
learned θ(load) gate from §1/§5-D2 is exactly the mechanism that can realize
this schedule online — and note the doc's "novelty-gated lr" serves a different
purpose (update-vs-add semantics): erosion is *diffuse* (every write erodes
everything ∝ θ/d_eff), so the capacity fix is the load schedule, not novelty
detection.

One more interaction worth wanting: low θ relaxes the **streaming** selectivity
budget by the same 1/θ factor (filler writes erode 33× less at θ=0.03), so the
low-lr retrain and D1 attack overlapping mass — but the write↔query context
mismatch is θ-independent, so **D1a still decides** what remains. Revised first
GPU session: D1b (eval-only, hours) → low-θ retrain with S-track metrics → D1a
on the rebuilt cache → `eval_real --scale` with specific-lift as primary.