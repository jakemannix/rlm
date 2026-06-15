> **Scope:** part of the *parametric-memory* track (frozen-base + meta-trained adapter + mutable store) — a SEPARATE line from the active sleep-time / lesson-centric behavioral-memory work (see rlm/sleep/README.md, docs/memory_experiment_status.md).

# Direction proposal: continual parametric memory (semantic, not episodic)

> **For a fresh review (Fable).** Read `docs/memory_START_HERE.md` first for the full
> map; this doc proposes a *change of target* for the memory track and asks for concrete
> architecture/method/eval proposals. Self-contained: the background you need is summarised
> below, with pointers for depth.

## 0. The ask, up front
We have **proven** that a *frozen* LLM can host a **persistent, meta-trained, save/reload-able
memory** — but we built and measured it as an **episodic fact store** (write a fact, recall it
verbatim across a session boundary). That target caps at a handful of facts and, at scale,
converges toward an in-VRAM vector database (differentiable RAG). The proposed pivot:

> **Aim instead at *semantic* memory — deeply lossy, *continual* parametric compression of an
> experience stream (what pretraining does) — on a frozen base (the anti-forgetting anchor),
> using an *MLP* substrate (where LLM knowledge actually lives) rather than the linear
> associative store.**

**Please propose concretely:** the substrate (single MLP vs a small stack, which layers, width/
depth), the online write rule, the anti-catastrophic-forgetting strategy, the smallest experiment
that would validate it, the continual-learning eval suite, and where (if anywhere) the episodic
dense store still fits. Push back hard if the framing is wrong.

## 1. Where we are (so the pivot is grounded)
Summary; full detail in `runs/REPORT.md` + `docs/memory_capacity.md` (+ its review).
- **Proven:** frozen Gemma-3-1b + a meta-trained-then-frozen "skill" (small key/query encoders +
  gates) + a mutable, persisted linear store recalls a *held-out real fact* across a destroyed
  context at **~81% of the in-context ceiling**, generates the exact answer ~**45%** of the time,
  with **perfect save→reload**. The *mechanism* — frozen base hosting a persistent online memory —
  works on real text.
- **The capacity ceiling we hit (~8–16 facts) and dissected** (`docs/memory_capacity.md`,
  `scripts/memory/capacity_probe.py`): the *store* is nowhere near full (a linear associative
  memory holds ~`d_k·d_v`, i.e. thousands); the binding limits are the *write rule* (error-correcting
  writes erode prior content — fixable by gentler/`θ(load)` writes), *streaming dynamics*, and key
  geometry. At *millions* of facts a dense store is the wrong structure entirely → sparse
  slot/product-key memory, which is a differentiable vector DB ≈ RAG.

**Conclusion driving this doc:** episodic recall is a solved-in-principle, scale-limited target.
The *interesting* target is the other kind of memory.

## 2. Episodic vs semantic — why the target should change
- **Episodic** (what we built): store discrete items, pull them back. Caps with the store's
  parameter budget; at scale ⇒ retrieval. Competes with MemGPT/Letta (token-space memory) and
  long-context — and largely *loses* at the million-item scale, where text retrieval is trivial.
- **Semantic / continual** (proposed): lossily compress a *stream of experience* into weights, over
  time — exactly what pretraining does to the web. This is the thing a 1M-context window and a
  vector DB **cannot** do: it abstracts, composes, and generalises across what it has seen, and it
  is paid for *once* (in weights), not re-supplied every query. **This is the regime where parametric
  memory has a defensible reason to exist.**

## 3. The key technical insight: our store is a *degenerate MLP-memory*
This reframes "should we use an MLP like Titans?" — we essentially already are, minus the
nonlinearity:
- Transformer **MLP layers *are* key-value memories** (Geva et al. 2021, *FF layers are key-value
  memories*): first matrix = keys (pattern detectors), second = values (written to the residual).
- The fact-editing literature edits **MLP weights** to store/change facts (ROME, MEMIT) — direct
  evidence that *knowledge lives in MLPs*.
- Our store is `read = M·q`, `M = Σ vᵢkᵢᵀ` — a **single, *linear* key-value layer**. An MLP memory
  is `W₂·σ(W₁x)` — the *same* structure **plus a nonlinearity `σ`**.
- That `σ` is exactly what removes our capacity wall: a *linear* key-value store caps at ~rank `d_k`;
  the nonlinearity breaks linear superposition and buys **super-linear → exponential** capacity per
  parameter (modern Hopfield; our own probe D: softmax readout held ~32× the linear one).

So: **a nonlinear MLP (ideally a small stack across a few layers) is the natural substrate for
compression** — it matches where LLMs store knowledge, it breaks our cap, and depth gives the
surface→semantic abstraction a single layer can't. The repo already contains an MLP memory
(`rlm/memory/{titans,miras,modeling}.py`) — the thing this project originally started from and
pivoted *away* from; this proposes pivoting back, but in the frozen-base + persistent + meta-trained
framing.

## 4. The proposed architecture
**Frozen base + a plastic MLP memory, online-updated, persisted, meta-trained to not forget.**
- **Frozen base** = the stable fallback / catastrophic-forgetting anchor (this is the piece Titans
  lacks — Titans co-trains the memory with the base).
- **Plastic MLP memory** at one or a few layers, written into the residual stream; updated online by
  a surprise/gradient rule (Titans-style) or a meta-learned update.
- **Persisted across sessions** (Titans resets its memory per sequence; we save/reload it).
- **A meta-trained "skill"** that learns *how* to update so writes compress without eroding prior
  knowledge.

vs the alternatives: **Titans** — co-trained + per-sequence reset (long-context, not persistent
knowledge). **RAG/MemGPT** — text in an external store (scales, but no abstraction/composition,
re-injected into context every call, not differentiable). The proposed system is *in-weights,
composable, zero-query-time-context, end-to-end differentiable, and persistent.*

## 5. The honest costs / open problems (don't paper over these)
1. **Writing stops being one-shot/analytic.** Our delta rule writes a fact exactly in one step; an
   MLP memory writes by **online gradient/surprise steps** — slower, lossier, and the meta-learning
   that keeps them stable is genuinely fragile (this project's early bugs were all
   meta-training-through-the-inner-loop). *Upside:* lossy iterative writes are the right tool for
   *compression*.
2. **The hard problem moves from capacity → catastrophic forgetting** (continual learning: EWC,
   replay, gated/sparse updates, etc.). The frozen base anchors it; the plastic part can still erode
   itself.
3. **Evaluation changes** — *not* "did I recall fact X." It becomes continual-pretraining-style:
   perplexity on a held-out stream over time, forward/backward transfer, retention curves.

## 6. What carries over (this is not a restart)
The memory track's *scaffolding is substrate-agnostic*: the frozen-base post-norm splice, cached-
activation meta-training (`rlm/memory/{trainer,cache}.py`), persistence/A4, the whitening + match-
gate machinery (`skill.py`), and the eval harness. Swapping the linear store for an MLP memory is a
**component change inside the existing framework**, and `rlm/memory/titans.py` already implements an
MLP memory to build from. Treat the dense linear store as the *finished proof that the plumbing
works* (a frozen base **can** host a persistent, meta-trained, online memory — demonstrated), and
make the next milestone the MLP-substrate continual-compression version.

## 7. Questions for the proposal
1. **Substrate.** One MLP vs a stack at K layers? Which layers (early-surface vs mid-semantic)?
   Width/depth/value-space? Fixed-size vs growing? Does sparsity (product-key / mixture-of-MLPs)
   belong here for the long tail, with a dense MLP for the working set?
2. **Write rule.** Surprise/gradient (Titans) vs a meta-learned update operator? One inner step or
   several? How do we keep the inner loop stable enough to meta-train through (our recurring pain)?
3. **Anti-forgetting.** Frozen-base anchor + what — replay from the activation cache, a retention
   regularizer (EWC-like), gated/sparse writes, a fast/slow two-store split (complementary learning
   systems)? What's the minimal mechanism?
4. **Smallest decisive experiment.** What single run would show "continual lossy compression on a
   frozen base beats the frozen base alone, without catastrophic forgetting"? On what stream
   (a domain corpus the base is weak on? a temporal stream?), with what metric?
5. **Eval protocol.** The continual-learning metric suite + dataset + the comparison baselines
   (frozen base; base + online LoRA; RAG) that make a *novel* claim falsifiable.
6. **The honest novelty question.** How is this distinguishable from "continue-pretraining a LoRA
   online with replay"? What is the *specific* claim that isn't just that? (If the answer is
   "nothing," that's worth knowing now.)
7. **Where does the episodic dense store fit** — as a working-set front-end to the semantic store,
   or is it retired?

## 8. References to ground the proposals
Geva et al. 2021 (FF layers = key-value memories); Meng et al. 2022 ROME / MEMIT (editing factual
knowledge in MLP layers); Behrouz et al. 2024 Titans (test-time neural memory; momentum + forgetting)
+ the Miras follow-up; Ramsauer et al. 2020 (modern Hopfield, exponential capacity via softmax);
Schmidhuber fast-weight programmers; Kirkpatrick et al. 2017 EWC + the continual-learning /
test-time-training literature.
