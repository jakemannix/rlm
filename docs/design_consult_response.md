# Design response: persistent parametric memory on a frozen LLM

> **Status:** plan-of-attack from Claude (Fable), responding to `docs/design_consult.md`,
> after reading `rlm/memory/{titans,miras,config,modeling}.py`, the wrapper, and the
> regression tests on branch `claude/titans-memory-module-YCkCl`. Everything here is
> argued, not asserted; each recommendation states what evidence would flip it.

---

## 0. Headline diagnosis: the binding-geometry problem

The brief frames the open risks as (a) store capacity (welded MLP), (b) whether a frozen
base can emit memory-shaped keys, and (c) whether residual injection is readable. All
three are real, but none of them is the thing that fails A1 first. The thing that fails
A1 first is **where the value comes from**.

The current wrapper computes `k, v, q = k_proj(h_t), v_proj(h_t), q_proj(h_t)` — all
three from the *same* hidden state, at the *same* position, at the *same* (last) layer.
Walk the ingest of `"Vault 7's access code is QX-4471."` through that scheme:

- At the position of `"is"`, the last-layer hidden state encodes the base's *prediction
  of the next token*. The base does not know the fact, so this state encodes the base's
  prior — generic continuations — **not** `" Q"`. (If the base could predict the gold
  token here, the fact wouldn't be novel and A1's control would fail anyway.)
- At the position of `" Q"`, the last-layer hidden state predicts `"X"`. The identity of
  the *current* token is largely rotated away by the last layer — late residual streams
  are next-token-shaped (the logit-lens/tuned-lens observation); current-token identity
  is linearly strong in embeddings and early/middle layers and weak at the top.

So under the welded scheme, **no writable (k, v) pair at the last layer contains the
gold answer in head-readable form.** The memory can faithfully autoassociate the base's
own prediction states (which is exactly what the rel-MSE ≈ 0.01 result certifies — write
fidelity in value space), and still produce zero logit lift on A1, because value-space
fidelity ≠ logit-space usefulness unless the values are head-aligned. This is why the
existing in-isolation result, while a necessary mechanical milestone, does not yet
predict anything about A1.

**The fix is an offset binding:**

> key at position *t* from a deep representation of the *context so far*;
> **value at position *t* = the input embedding of token *t+1*** (answer-shifted).

Gemma ties input and output embeddings (`tie_word_embeddings=True` across the family —
verify on the loaded config), so `e[" Q"]` *is* (up to scale) the unembedding row of
`" Q"`. A read that returns ≈ `e[gold]`, injected into the head's input, shifts the gold
logit by ≈ `⟨W_U[gold], out⟩` — directly, by construction. The meta-learner's job
collapses from "discover a steering code the frozen base happens to interpret" to "learn
key/query projections and gates; the value codebook and the readout are given."

In one sentence: **v1 is a meta-learned, cross-session induction head.** During ingest
it binds (deep representation of context at *t*) → (embedding of token *t+1*); at recall,
the query "…access code is" retrieves the stored next-token embedding from across the
session boundary, exactly as an induction head retrieves it from within the window. This
also reframes Q2/Q3 (below): the base doesn't need to learn to read a perturbed state,
because the perturbation is delivered in the one code the head natively reads.

What would flip this: if Step-1's oracle test (§4) shows that hand-constructed
(k = deep h, v = e_gold) writes produce no logit lift even with hand-tuned scale, the
tied-embedding readout assumption is wrong in practice (e.g., final-logit softcapping or
norm effects kill it) and we move to the middle-layer / soft-token arms.

---

## 1. Answers to the six design questions

### Q1 — Decouple store from controller? Yes, but the clean version is cheaper than you think.

Your slow/fast-weights instinct is correct, and the welded-MLP capacity competition
(θ₀-as-skill vs θ−θ₀-as-content) is real — but it is an **A3 phenomenon, invisible at
N=1**, so don't let it block A1. The clean decoupling:

- **Store = a zero-init linear map** `M ∈ R^{d_v × d_k}`, `M₀ = 0`. All the *skill* lives
  in the frozen surround: key/query encoders, value codebook (= the base's embedding
  table, frozen for free), gates, and a scalar/linear readout gain. With θ₀ ≡ 0 there is
  no init-vs-content competition *at all* — the decomposition isn't relaxed, it's
  dissolved.
- **Your update rule on a linear store is exactly the delta rule.** For
  `L = ½‖Mk − v‖²` (per token), `∇_M L = (Mk − v)kᵀ`, so the Titans update becomes
  `S_t = η S_{t−1} − θ_t (M_{t−1}k − v)kᵀ`, `M_t = (1−α)M_{t−1} + S_t` — DeltaNet with
  momentum and decay (Yang et al. 2024; the lineage Titans/Miras themselves cite via
  Schlag's fast-weight programmers). Consequences:
  - The inner gradient is **analytic**. No `torch.func`, no `functional_call`, no
    per-batch Python loop: the whole inner loop is a few GEMMs, batch-parallel, and
    differentiable w.r.t. projections/gates by construction (BPTT through the chunk
    sequence). This sidesteps the slowest, most fragile part of the current code for
    the arm that matters first.
  - Capacity is *interpretable*: with RMS-normalized keys in R^d, random keys have
    pairwise cosine ~N(0, 1/d); recall noise after m one-shot writes scales like
    √(m/d)·‖v‖. Planning numbers: clean top-1 recall to **m ≈ d/10** conservatively,
    **m ≈ d/3** with delta-rule error correction and a second write pass. For O(10²)
    facts: **d_k = d_v = 512–1024** (0.26–1.0 M floats — trivially serializable, A4 is
    free). This answers "how to size the store" with arithmetic instead of vibes.
- **Width over depth.** If/when the linear store saturates (A3), scale by widening d_k
  and adding *heads* (h independent stores with d/h-dim keys, mirroring attention), not
  by deepening the store. Depth buys compositional generalization — which the frozen
  controller and the base already provide — while width buys addressable slots; depth
  also worsens inner-loop conditioning and re-introduces sequential autograd. Keep the
  deep-MLP store as the *comparison arm* in A3, not the default.

Convenient fact: `MemoryMLP` with `n_layers=1` is already a single `Linear(d_k→d_v)`, so
the linear arm is nearly config-reachable today (`n_layers=1`, zero-init weights, zero
bias). Two cautions: (i) exclude the store's own parameters from the outer optimizer, or
meta-training will drift θ₀ off zero and quietly re-weld skill into the store; (ii)
`cfg.init_scale` is shared with the wrapper's k/v/q/out projection init (`modeling.py`),
so `init_scale=0` would zero-init the projections and kill all gradients — the episode
trainer (§3) should own its projection init separately (e.g., orthogonal or std 1/√d).

### Q2 — Integration: residual injection vs memory-as-tokens. Injection first — but at a sharper point than the current hook.

Your suspicion ("attention knows how to read tokens; it was never trained to interpret a
perturbed hidden state") proves too much: *every* component of the transformer
communicates by adding vectors into the residual stream — additive perturbation is the
native medium, which is why activation-addition steering and ROME-style edits work on
frozen models at all. The genuine concern is quantitative, not categorical: late-layer
residuals in Gemma carry massive activations (your measured abs-max ~66k), which inflate
the pre-final-norm RMS, so an additive write of moderate norm gets relatively squashed
by the final RMSNorm.

So make the injection point **post-final-norm** (hook `model.model.norm`'s output, not
the last block): `head_input ← norm(h) + g · out_proj(read)`. At the last layer there
are no downstream blocks to lose, the head reads the injection linearly
(`Δlogit = W_U · g·out_proj(read)`, massive-activation-proof), and with v = tied
embeddings the required `out_proj` is ≈ scaled identity. This is the maximal-probability
v1.

Memory-as-tokens stays in the plan as the **fallback arm**, in its cheapest form: a
*two-pass soft token*. Pass 1 encodes the prompt and produces q; retrieve
`read = M(q)`; pass 2 prepends `s = mem_proj(read)` as one (or a few) `inputs_embeds`
token(s) and re-runs. This reduces to retrieval-conditioned soft prompting on a frozen
LM — heavily precedented, very likely to work — at 2× inference cost and more plumbing
(no KV surgery needed, which full per-layer memory-token splicing would require). Trigger
for switching: oracle injection (§4, Step 0) fails, or meta-training shows consistent
rank improvement that never crosses into top-k (the "steering moves logits but can't
beat a confident prior" signature).

One asymmetry to note: last-position injection delivers ~one token of steering per
forward. That's *sufficient for A1 by your own metric* (first distinguishing token),
and during generation each new step re-queries, with emitted tokens becoming context the
base completes from. Multi-token-answer robustness is an upgrade, not a prerequisite.

### Q3 — Can a frozen base emit memory-shaped keys? For A1, yes by construction; for paraphrase, add depth *outside* the base before touching it.

Keys don't need to be "memory-shaped" in any absolute sense — they need to be
**consistent between ingest phrasing and query phrasing**. A1 as specified shares the
literal prefix ("Vault 7's access code is"), so `h` at the query position in session 2 ≈
`h` at the same relative position in session 1: key consistency is nearly free, and
*no* base intervention is needed. Use post-final-norm h as the projection input (apply
the frozen `model.model.norm` functionally inside the memory path) — this also fixes the
input-scale problem at the source instead of relying on the memory's internal RMS-norm to
clean up O(10⁴) activations after a randomly-initialized projection has already smeared
the massive dims across all key dims.

For paraphrase robustness (call it **A1b**, and label your verbatim result A1-verbatim so
it doesn't overclaim), the escalation ladder is: (1) train k/q on paraphrase-augmented
meta-episodes; (2) deepen the key/query *encoders* — small MLPs on top of frozen h,
which are part of the skill and live entirely outside the base; (3) only then LoRA near
the splice. Paraphrase-invariant semantic content is linearly decodable from mid/late
layers (standard probing result), so I'd bet on (1)+(2) sufficing and on never needing
(3) — which also preserves the headline claim "the base is frozen" in its strong form.

### Q4 — Splice point: keys/queries from deep post-norm h; injection post-final-norm; middle layer is the second arm, not the first.

The middle-layer intuition ("more retrievable") is right about *where token identity and
relational structure live*, but middle-layer **injection** buys you a harder problem for
v1: the read must imitate states the mid-layer distribution would produce, and credit
assignment for meta-training must flow through the frozen suffix (backprop through half
the base — cheap-ish since weights are frozen, but it re-couples the trainer to the base;
see §3 for why decoupling is the velocity unlock). The last-layer/post-norm arm has a
near-closed-form solution and a base-free trainer. Sequence accordingly.

The middle-layer arm becomes interesting precisely when: answers are multi-token and
compositional, queries are paraphrases the top-layer keys don't capture, or oracle
injection shows the head can't be steered against a confident prior. Its natural form
then is "inject retrieved token identity at layer ℓ and let the frozen suffix do the
grammar" — which converges with the soft-token fallback. Multi-scale (both) is an A3+
luxury.

### Q5 — Meta-training: episode-structured CE through the frozen head, trained entirely from cached activations.

**Objective.** Two stages:

1. *Warmup (optional but cheap):* direct retrieve-the-value loss in value space —
   cosine/L2 between `M(q)` and `e[gold]` — to get projections/gates into the right
   basin. Runs entirely on cached tensors.
2. *Main:* next-token **CE on the answer tokens of recall episodes**, computed through
   the frozen final norm + LM head, **plus** an abstention term:
   `KL(p_augmented ‖ p_base)` on positions/episodes where the memory contains nothing
   relevant. The KL-to-base form is sharper than CE-to-data for abstention: it directly
   optimizes "defer to the base," which is your A2 in loss form. Total:
   `L = CE(answer tokens) + λ·KL(non-recall positions)`.

CE through the head is non-negotiable as the final objective — the entire open question
is whether reads move *logits*, and only CE measures that. But note what the post-norm
splice buys you: the gradient path is
`CE → head → norm-output (+ injection) → out_proj → read → inner loop → projections/gates`,
and **never enters a single transformer block**. Both the ingest forward and the recall
forward can run under `no_grad`; you cache, per episode:
`(h_ingest, e_ingest_shifted, h_query, answer_ids)` — and meta-train the ~1–10 M-param
memory machinery with the base completely out of the loop. On the L4: ~20k episodes ×
~200 tokens ≈ 4 M tokens of Gemma-3-1B forward ≈ tens of minutes to build the cache once;
each meta-training config then runs at toy-model speed (minutes). This is the
experimental-velocity unlock, and it's also why I'd build the trainer as a **standalone
module that does not use `TitansAugmentedLM` at all** — the hook's train/eval semantics
(fresh state per forward in train mode) were designed for single-forward meta-training
and actively fight the two-forward episode structure. Keep the wrapper for deployment;
give the trainer raw tensors.

(If you later move to mid-layer injection, the cache trick degrades — gradients must
traverse the frozen suffix — which is one more reason the last-layer arm goes first.)

**Episode structure** (the "context gone" condition, no mask artifice):

- Forward 1 (ingest): run base over fact-bearing text; collect (h, e_shifted); run the
  inner loop (analytic delta rule) to produce state S, **keeping the graph** w.r.t.
  projections/gates.
- Forward 2 (recall): independent text containing the query; read `M(q)` at answer
  positions; inject post-norm; CE on gold tokens. The fact tokens are simply absent from
  forward 2 — the eval condition and the training condition are the same condition.

**Data distribution.** Synthetic, generated programmatically — the skill must
generalize over *associations*, not over one task:

- ~50 relation templates (access code, ticker, birthday, codename, count, color, …) ×
  nonce entities × **nonce answers** with varied tokenization lengths (1–4 tokens), so
  the value distribution spans the embedding table rather than a corner of it.
- Facts embedded in filler prose (wikitext-ish), so write-gates learn selectivity; the
  learnable per-input lr gate is exactly the right mechanism, but see §5 on its
  per-chunk scalar pooling.
- Episode mix: single-fact recall; k∈{1..8}-fact ingest with one queried (teaches
  non-interference + selective recall); **abstention episodes** (memory holds A, query
  is unrelated text / a fact the base knows → KL term); **wrong-fact controls** (your
  A1 control (ii), as a training signal, not just an eval).
- Curriculum: recall-only warmup (prevents early gate collapse, §5) → mixed with
  abstention → multi-fact → paraphrased queries → **multi-session episodes** (state
  carried across 2–3 ingest forwards with filler between) — this last is what trains the
  forget gate to *not* forget fact 1 while later text streams through, which is the
  actual cross-session persistence skill.

### Q6 — The ladder (see §4).

---

## 2. v1 specification (concrete)

Base: `google/gemma-3-1b-it`, bf16, frozen, `no_grad` everywhere. (Read `hidden_size`,
`tie_word_embeddings`, and any `final_logit_softcapping` off the loaded config rather
than trusting recall; if a final softcap exists, fold it into the oracle sweep.)

Memory machinery (the *skill*, meta-trained then frozen):

- `key_enc`, `query_enc`: Linear (later: 2-layer MLP) from `norm(h)` ∈ R^{1152} → R^{d_k},
  d_k = 512. RMS-normalize outputs (keep `normalize_qk=True` semantics).
- Value pathway: `v_t = e[token_{t+1}]` from the frozen embedding table (mind Gemma's
  input-side √d scaling — use raw table rows, normalized; `normalize_v=True` is fine,
  `out` rescales).
- Store (the *content*, mutable, persisted): `M ∈ R^{d_v × d_k}`, d_v = 1152 (embedding
  dim) or a compressed d_v with a learned frozen decoder back to embedding space —
  start with full embedding dim; compress only if A3 forces it. `M₀ = 0`, excluded from
  the outer optimizer. Momentum buffer same shape.
- Update: delta rule with momentum + decay, **analytic**, `inner_lr ≈ 1.0` modulated by
  the learned gate (see §5 — the current default of 1e-2 cannot one-shot-write),
  `chunk_size` small (1–4) for ingest.
- Readout: `Δ = g(q, read) · out_proj(read)` added to `norm(h)` at the read position
  (post-final-norm). `out_proj` init ≈ identity-scaled; `g` a learned scalar gate
  (sigmoid over [q; read] features), trained to ≈0 by the KL/abstention term — this is
  the explicit "abstain" mechanism an MLP-store lacks (an MLP returns *something* for
  unseen queries; injecting `out_proj(garbage)` into every token of every session is the
  A2 failure mode, so abstention must be an explicit trained behavior, not a hope).
- **Modes:** ingest = write-only (no injection during ingest — the current `read_write`
  perturbs the residual *while* ingesting, which adds noise and lets the write loop see
  its own outputs); recall = read-only (writing the query itself would delta-rule
  *overwrite* the stored association at the same key with a wrong value — the query's
  "next token" is not the gold answer until after the model has emitted it). Learned
  write-gates can subsume this asymmetry later; v1 hard-codes it.
- Persistence (A4): serialize `M` (fp32) only; **zero the momentum on load** — carrying
  momentum across a session boundary applies a stale update direction to the first
  writes of the next session. (Decide empirically, but default to params-only.)

Deployment batch is B=1 per memory owner; the batch dimension exists only in training.

---

## 3. Trainer architecture

A standalone `EpisodeTrainer` consuming cached tensors:

```
cache:   for each episode: h_ingest [T1, d], e_shift [T1, d], h_query [T2, d],
         answer_positions, answer_ids, episode_type ∈ {recall, abstain, control, multifact, multisession}
forward: S = inner_loop(key_enc(norm(h_ingest)), v(e_shift); gates)      # analytic, batched
         read = M_S(query_enc(norm(h_query[answer_positions])))
         logits = lm_head(norm(h_query) + g·out_proj(read))               # frozen norm+head in graph
loss:    CE(answer_ids) + λ·KL(p_aug ‖ p_base)  [KL uses cached base logits or recomputes head]
```

The frozen norm + head are the only base components in the graph (head matmul over a
262k vocab at a handful of positions — cheap). Everything else is GEMMs over cached
activations. This is also where the current code's two perf issues disappear for the
linear arm: the per-batch Python loop with `functional_call`+`grad` in `write()`, and
the O(B) full-tensor clones inside it.

---

## 4. The ladder — A1 pass-or-die ordering

Each step has a gate; do not proceed past a failed gate, branch instead.

**Step 0 — Oracle injection (hours, zero training).** Bypass memory entirely: for the
query forward, add `λ·e[gold]` (and separately `λ·W_U[gold]` if untied in practice) to
the post-norm hidden at the final position; sweep λ. *Gate:* ∃ stable λ-range where gold
enters top-5 with Δlogprob ≥ +3 nats and the rest of the distribution isn't shredded
(spot-check perplexity on neutral continuations). *If it fails:* residual injection is
dead at this splice regardless of memory quality → jump to the two-pass soft-token arm
(re-run Step 0 there: prepend `λ·e[gold]`-ish soft token, measure lift).

**Step 1 — Oracle write/read (hours).** Hand-construct the binding with no
meta-learning: run ingest, take k = key_enc₀(norm(h)) with a *random orthogonal* key_enc,
v = e_shift, write with inner_lr=1, chunk=1; run the query forward, read with the same
encoder as q, inject with hand-tuned scale. *Gate:* A1 metric passes with both controls
(empty memory; wrong fact). This factorizes A1 into (injection works) × (store binds) ×
(meta-learning can find it) and certifies the first two. *If Step 0 passed but Step 1
fails:* the failure is key consistency or delta-rule interference at T tokens — debug
with k from the literal answer-position only, then widen.

**Step 2 — Meta-train the skill (≈1 day incl. cache build).** Cache 10–50k synthetic
episodes; warmup retrieval loss; main CE objective; recall-only curriculum first.
Evaluate A1-verbatim + controls + A4 (serialize/reload mid-eval). *Gate:* A1 numerical
pass — Δlogprob(gold | S vs ∅) ≥ +3 nats, gold ∈ top-5, controls |Δ| ≤ 0.5 nat. **This is
the pass-or-die moment, and it arrives within ~a day of starting Step 2.**

**Step 3 — A2 (abstention).** Mix KL/abstention episodes in; re-eval A1 (watch for gate
collapse, §5) and A2 (unrelated query B: no fabrication, distribution ≈ base).

**Step 4 — A3 (capacity).** Recall@N for N ∈ {1, 2, 4, …, 256}; arms: linear store
d_k ∈ {128, 256, 512, 1024}, multi-head variant, welded-MLP baseline (current code).
This is where the store/controller question is *settled by data*; my prediction is the
linear store dominates the MLP on interference-onset per parameter and is vastly easier
to reason about when it doesn't.

**Step 5 — A1b (paraphrase).** Paraphrase-augmented episodes → deeper key/query encoders
→ only then discuss LoRA.

**Step 6 — Scale the base.** Gemma-3-4B fits the L4 in bf16 with room; and because the
base is forward-only in *both* caching and deployment, a 4-bit-quantized 12B is viable
for cache generation. Re-fit projections per base (dims differ); the interesting
question is whether the *skill recipe* (not weights) transfers — bigger bases should
yield better keys, i.e., paraphrase robustness improves before anything else.

---

## 5. Failure modes and landmines (ordered by expected pain)

1. **inner_lr is 100× too small for one-shot writes.** With the repo's normalization
   conventions, one delta-rule step contracts the residual error by ≈ (1 − θ·d_k/d_v);
   at d_k=d_v that's (1−θ). Default `inner_lr=1e-2` (gate-scaled ×(0,2)) means a single
   pass writes ≤2% of an association — fine for Titans' long-context regime where a
   "fact" spans many tokens, fatal for single-shot episodic facts. Set `inner_lr` so the
   gated range includes ≈1.
2. **Gate collapse to "never write / never read".** Early in training, injected reads
   are noise; CE is minimized by g→0 and the lr-gate→0, a stable local optimum that
   silently disables the memory. Mitigations: recall-only warmup before abstention
   episodes enter the mix; loss weighting; monitor gate statistics as a first-class
   training metric.
3. **Per-chunk scalar gates average away selectivity.** `_gates()` mean-pools the chunk
   (and batch) to a scalar; with chunk_size>1 one informative token inside filler is
   diluted. For ingest use chunk_size 1–4, and plan a per-token gate variant — for the
   linear store it's trivial (per-token θ_t in the analytic update).
4. **Confidence smearing instead of recall.** Injection raises *many* logits; your
   control (ii) catches it — also track entropy at the answer position and the margin
   to the base's argmax, not just gold rank.
5. **Momentum across session boundaries** applies a stale direction to the next
   session's first writes; serialize params only (see §2). Relatedly, A4 should reload
   into a *fresh process* to count as literal persistence.
6. **`init_scale` coupling** (config) zero-inits the wrapper's projections if you set it
   to 0 for the store; the standalone trainer should own projection init.
7. **Logit softcapping / embedding scaling.** Gemma-2 had final-logit softcap; Gemma-3
   reportedly dropped it — *check the config of the actual checkpoint*; a softcap
   compresses injection effect and changes the oracle-λ sweep. Gemma scales input
   embeddings by √d at the input — use raw table rows for the value codebook and let
   normalization handle scale.
8. **Train/deploy skew in the inner loop.** Meta-training must run the inner loop with
   the deployment chunk_size, normalization, and write/read asymmetry. The current
   wrapper's train-mode (fresh state per forward) vs eval-mode (persist + detach) split
   is exactly the kind of skew to design out — another argument for the standalone
   trainer + deployment-only wrapper.
9. **Overclaiming A1-verbatim.** The shared literal prefix makes k≈q by construction.
   Report A1-verbatim and A1b-paraphrase as separate results from day one. (Entity-side
   priors — "Vault 7" is a real CIA-leak referent — don't matter much since the answer
   is nonce; the nonce-answer instinct in the brief is the right one.)

---

## 6. Related anchors (brief, for grounding not tutoring)

Fast-weight programmers and the linear-attention duality (Schmidhuber; Schlag et al.);
DeltaNet / parallelized delta rule (Yang et al. 2024) — the exact update the linear arm
implements; Titans/Miras (Behrouz et al.) for the momentum/forget formulation being kept;
ROME for the "write (k,v) that downstream consumes" intuition and the unembedding-aligned
readout; logit/tuned lens for why last-layer h is next-token-shaped; soft-prompt steering
literature for the two-pass fallback's prior probability of working; memory-layers /
product-key memories for where the multi-head wide store goes if A3 demands scale.

---

## 7. What I'd tell Opus to build first, in order

1. `scripts/oracle_injection.py` — Step 0 sweep, post-norm hook, report table.
2. `rlm/memory/linear_store.py` — analytic delta-rule store (params/momentum dicts →
   plain tensors), with the wrapper's post-norm read/inject path and write-only /
   read-only modes.
3. `scripts/build_episode_cache.py` + `rlm/memory/episodes.py` — synthetic generator +
   cached (h, e, positions) episodes.
4. `rlm/memory/episode_trainer.py` — standalone trainer (§3), CE + KL objective,
   gate-statistics logging.
5. `scripts/eval_acceptance.py` — A1/A2/A4 harness with the numerical gates from §4,
   wrong-fact and empty-memory controls built in, JSON output so runs are comparable.

The wrapper (`TitansAugmentedLM`) survives as the deployment shell; the welded-MLP
path survives as the A3 baseline arm. Nothing already built is wasted — the
normalization, the gate machinery, the meta-training-gradient fixes, and the regression
harness all carry over; the changes are *where v comes from, where the read lands, and
where the training loop lives*.
