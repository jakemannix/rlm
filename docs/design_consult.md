# Design consult: persistent parametric memory on a *frozen* LLM

> **Status: working brief, not authority.** This is a request-for-design written by
> Jake + Claude to hand to a stronger model (Claude "Fable") for a plan of attack.
> Everything here is a hypothesis to pressure-test, not a settled decision.

## Goal (reframed)
Give a **frozen, pretrained** LLM (Gemma-3, 1B now; we can go bigger on our hardware) a
**persistent, long-term memory** via a Titans/Miras-style module updated by online gradient
steps at inference. We train **only** the memory machinery; the base stays frozen. The point
is **not** long context — it's **cross-session memory**: information absorbed while the model
runs, persisted as state, and **recalled in a later session after the original context is gone.**

## Core thesis we want pressure-tested: separate the *skill* from the *store*
A Titans memory secretly bundles two things:
- **The skill ("how to memorize")** — the k/v/q projections, the gate networks
  (lr/momentum/forget), `out_proj`, and the initial memory weights `θ₀`. These are *fixed
  after training*. This is meta-learned and should be trained on **broad/generic data**, then
  frozen.
- **The store ("what is memorized")** — the running memory weights `θ` evolving from `θ₀`.
  *Fully mutable at deployment; persisted across sessions.*

Titans welds both into one small MLP: `θ₀` is the skill's init and `θ−θ₀` is the content,
sharing one tiny tensor — a capacity bottleneck (a better learner-init competes with content
capacity). This is exactly the **slow-weights / fast-weights** decomposition (Schmidhuber;
Schlag et al., "Linear Transformers are Secretly Fast Weight Programmers"): a frozen *slow*
controller programs a higher-capacity *fast* store. **Open question: should we decouple them —
a frozen meta-learned controller driving a separately-sized mutable store (matrix or larger
net) — rather than one welded MLP? How should the store be structured/sized?**

## What we already have (mechanics work; usefulness unproven)
- `TitansMemory`/`MirasMemory`: MLP-as-memory with the online surprise/momentum/forget update,
  differentiable through the inner loop (`torch.func`) so `θ₀`+gates+projections can be
  meta-trained.
- `TitansAugmentedLM`: wraps a frozen HF causal LM, hooks one block (default last), projects
  hidden→(k,v,q), writes (k,v), reads M(q), adds `out_proj(read)` back into the residual.
  Memory state **persists across forward calls in eval mode** (that's the cross-session
  substrate) and re-inits fresh per sequence in train mode (for meta-training).
- We hardened it on real Gemma: RMS-normalize K/Q/V inside the update (Gemma's residual has
  **massive activations, abs-max ~66k**, which otherwise diverges to NaN), fixed the
  meta-training gradient and dtype/device handling. In isolation it **memorizes real Gemma
  hidden states to rel-MSE ~0.01**. Full regression suite passes.

## Constraints / priors
- Single **L4 (23 GB)**. Base frozen + memory at the last layer ⇒ training forwards through the
  base but **backprops only through the memory + LM head** (no backprop through the base's
  layers) ⇒ a bigger base is affordable.
- Synthetic data is unlimited; "how much data" = "meta-train the skill to convergence on a
  broad enough distribution that it *generalizes* to unseen associations" (NOT overfit one task).

## The eval regime (this is clean, no artifice)
Because the claim is cross-session persistence, the test writes itself:
- **Session 1 (ingest):** forward the model over a fact; the memory writes it;
  **snapshot/persist the memory state**; discard the context.
- **Session 2 (recall):** a fresh forward with the **fact's tokens absent from the window**,
  memory state reloaded; ask. Only the persisted memory can answer. (Attention literally cannot
  — the tokens are gone.) This replaces any "handicap attention" trickery.

## Acceptance tests (our target — define "it works"; design toward these)
**A1 — minimal cross-session recall (the north star).**
- Use a fact **unknowable from pretraining** so any recall is unambiguously from memory —
  ideally a *nonce* answer with low prior, e.g. `FACT = "Vault 7's access code is QX-4471."`
  (the "2029 Superbowl → Green Bay Packers" framing is the intuition, but a real NFL team is
  guessable, which muddies attribution — prefer a random answer).
- Ingest FACT (session 1) → persist state `S` → fresh session with query `Q` = "What is Vault
  7's access code?" / "Vault 7's access code is" and memory `=S`.
- **Metric:** logprob/rank of the gold answer token under `Q`. **Pass:** with `S`, P(gold|Q) is
  materially higher than with empty memory, and the gold token lands in top-k. **Recalling even
  the first distinguishing token suffices** — a competent base completes the rest in-context.
- **Controls:** (i) empty memory ⇒ no lift (rules out base already knowing); (ii) ingest a
  *different* fact ⇒ no lift on `Q` (rules out "memory just makes everything more confident").

**A2 — specificity:** ingest A; recall A; an unrelated query B (never ingested) must NOT be
fabricated (memory defers to base).
**A3 — capacity curve:** ingest N facts, measure recall vs N (interference point).
**A4 — literal persistence:** serialize `S` to disk, reload, A1 still passes.

## Questions for the designer (challenge our hypotheses in parens)
1. **Decouple store from controller?** Frozen controller + higher-capacity mutable store
   (fast-weights matrix vs deeper net)? How to size/structure the store for O(10²) facts?
   *(We suspect the welded tiny MLP is the main thing that won't scale.)*
2. **Integration.** Additive residual injection at one layer vs **memory-as-tokens the frozen
   attention can attend to** (attention already knows how to read tokens; it was never trained
   to interpret a perturbed hidden state). *(We suspect memory-as-tokens may be the difference
   between signal and noise on a frozen base.)*
3. **Can a frozen base emit memory-shaped keys at all,** or is a thin learnable adapter required
   (LoRA near the splice / train final norm / learned k/q tied to the base's attention
   subspaces)? Where's the minimal effective intervention?
4. **Where to splice** — last layer (next-token-shaped) vs middle (more retrievable) vs
   multi-scale?
5. **Meta-training the skill:** what data distribution + curriculum yields a controller that
   generalizes to unseen associations (not a task-specific trick)? Objective — next-token CE on
   recall episodes, or a more direct retrieve-the-value loss?
6. **Scale ladder:** the smallest experiment that gives an unambiguous A1 pass/fail on one L4,
   then what to scale first.

## The ask
A concrete **plan of attack**: recommended store/controller factorization, integration
mechanism, splice point, the minimal base-intervention (if any), the meta-training data +
objective, and an **experimental ladder ordered to make A1 pass-or-die as fast as possible** on
a single L4 — plus the failure modes you'd watch for. Assume we implement competently; optimize
for learning whether this works *soon*.

## Pointers (for the designer pulling this repo)
- Memory module: `rlm/memory/titans.py`, `rlm/memory/miras.py`, `rlm/memory/config.py`
- Frozen-base wrapper: `rlm/memory/modeling.py` (`TitansAugmentedLM`)
- Tests / current behavior: `tests/test_titans_memory.py`,
  `tests/test_titans_gemma_integration.py`
- The fixes that made it run on real models: recent commits on
  `claude/titans-memory-module-YCkCl` (normalization, meta-training gradient, device/dtype).
