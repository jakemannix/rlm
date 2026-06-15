> **Scope:** part of the *parametric-memory* track (frozen-base + meta-trained adapter + mutable neural store + persistent writes across sessions) — a SEPARATE line from the active sleep-time / lesson-centric behavioral-memory work (see rlm/sleep/README.md, docs/memory_experiment_status.md). Though §8 endorses pivoting to the sleep-lesson approach, this doc's primary contribution is establishing the intellectual case for the parametric-memory design space.

# Field positioning: why the design point this repo occupies is empty

> **Status: literature positioning + research-program justification (2026-06).**
> Companion to `continual_memory_direction.md`. That doc proposes *what to build next*;
> this doc establishes *why nobody else has built it*, who the nearest published
> neighbors are, and what that implies for claims, baselines, and evals. Synthesized
> from a broad 2023–2026 literature sweep (online/continual LoRA, Titans/MIRAS/TTT,
> KV-cache injection, lifelong model editing, token-space agent memory).

## 0. The design point, stated precisely

This project's target is the composite:

> **Frozen base model** + **meta-trained write/read skill** ("the adapter that knows
> how to inject information") + **gradient-written neural store** (Titans-like MLP
> substrate) that is **updated online and persisted across sessions**.

The claim of this doc: that exact composite is unoccupied in the literature — not
because nobody thought of it, but because it has been approached from at least four
directions that each stop **exactly one coordinate away**, blocked by two genuine
technical walls, one economic squeeze, and one sociological gap. None of these amount
to an impossibility result.

## 1. Nearest neighbors — each differs in one coordinate

| System | Frozen base | Trained writer | Neural store | Online writes | Persists cross-session | The missing coordinate |
|---|---|---|---|---|---|---|
| [Titans](https://arxiv.org/abs/2501.00663) / [ATLAS](https://arxiv.org/abs/2505.23735) | ✗ (co-trained) | ✓ | ✓ MLP | ✓ | ✗ resets per sequence | persistence |
| [GRACE](https://arxiv.org/abs/2211.11031) (NeurIPS 2023) | ✓ | partially | discrete codebook | ✓ | ✓ | store is a lookup table, not superposed weights |
| [WISE](https://arxiv.org/abs/2405.14768) (NeurIPS 2024) | ✓ | ✓ router | ✓ side-FFN copies | ✓ sequential edits | ✓ | writes are supervised *edits*, not stream compression |
| [MemoryLLM](https://arxiv.org/abs/2402.04624) (ICML 2024) / M+ | ✗ (trained jointly) | ✓ | latent token pool | ✓ | ✓ (~10⁶ updates) | base must be trained for the memory |
| Larimar (IBM, ICML 2024, arXiv:2403.11901) | ✓ | ✓ | Kanerva-style matrix | ✓ one-shot | partially | episodic *editing* target; linear store |
| [StreamAdapter](https://arxiv.org/abs/2411.09289) (MSR 2024) | ✓ | ✓ meta-trained | LoRA-shaped | ✓ | ✗ explicitly "temporary" | persistence |
| [Doc-to-LoRA](https://arxiv.org/abs/2602.15902) / [DyPRAG](https://arxiv.org/pdf/2503.23895) | ✓ | ✓ hypernetwork | LoRA artifact | ✗ one-shot per doc | ✓ (as files) | no online accumulation |
| [TTT-E2E](https://arxiv.org/abs/2512.23675) / [In-Place TTT](https://arxiv.org/abs/2604.06169) (ICLR 2026 Oral) | ~ | ✓ | fast weights in MLPs | ✓ | ✗ per-context | persistence |

Two observations:

1. **The lifelong-model-editing community has been building this repo's linear-store
   track under a different name.** GRACE = frozen base + adapter at one layer holding
   a key-value store written during deployment and persisted. WISE = side-FFN memories
   with a router, knowledge *sharded* across parameter subspaces and merged. These are
   the real baselines for this work — not just RAG.
2. **Every system that has persistence gave something up to get it:** a discrete/
   symbolic store (GRACE), supervised delimited writes (WISE), or a co-trained base
   (MemoryLLM). Nobody combines *meta-learned gradient writes* + *indefinite
   persistence* + *frozen base*. The next sections explain why.

## 2. Technical wall #1: per-sequence reset is load-bearing for the meta-learning

Titans resetting its memory per sequence is not an oversight — it is what makes the
meta-objective well-posed. The skill (gates, projections, θ₀) is trained over episodes
drawn i.i.d. from a **bounded horizon**; every stability property of the update rule
is a statement about state distributions *reachable within that horizon from θ₀*.
Persist the store for weeks and it random-walks into state-space regions the skill
never saw during meta-training: the meta-learned gates are off-distribution **with
respect to their own memory**, and errors compound with no reset to flush them.

This is the unbounded-horizon RNN stability problem, and there is no published
technique for bounding the drift of a meta-learned update rule over horizons 100–1000×
its training episodes. Evidence that even the architecture's own authors keep adding
resets rather than removing them:

- [ATLAS](https://arxiv.org/abs/2505.23735) criticizes purely online updates as
  "myopic" — and fixes it with a *sliding-window* optimization, not persistence.
- [TNT](https://arxiv.org/abs/2511.07343) (ICLR 2026) introduces **periodic local
  memory resets** as a training necessity for parallelization.
- The whole Titans/TTT lineage thereby silently converts "memory research" into
  "long-context research" — state lives exactly as long as one sequence.

This repo's own run history (fragile inner-loop meta-training; NaN cascades from
Gemma's abs-max ~66k residual activations, see `design_consult.md`) is this wall in
miniature. **Implication:** the eval must *measure* horizon drift rather than hope
past it — see §8.

## 3. Technical wall #2: a frozen base was never trained to read the injection

A co-trained Titans base *learns* to consume the memory read added to its residual
stream. A frozen base has no training signal telling it what that perturbation means;
the injected signal must mimic distributions the base already consumes. Observed
pattern across the field: every working frozen-base system solves injection by
collapsing into an interface the base natively understands —

- **tokens** → RAG / token-space agent memory;
- **KV states** → trained/precomputed KV caches
  ([Cartridges](https://arxiv.org/abs/2506.06266),
  [Knowledge Packs](https://arxiv.org/abs/2604.03270),
  [ReasonCACHE](https://arxiv.org/abs/2602.02366));
- **weight deltas in the base's own parameter basis** → LoRA / model editing.

When researchers start near this repo's design point and iterate, they slide down the
gradient into an existing paradigm (soft prompts become trained KV caches; adapter-
writers become hypernetwork-LoRA; online stores become codebooks). The point that
*doesn't* collapse — novel residual-stream injection from an online store — inherits
the open problems of all its neighbors simultaneously. The `design_consult.md`
question "residual injection vs memory-as-tokens" is exactly this wall; the field's
revealed answer is "pick an interface the base already speaks."

## 4. The economic squeeze: the niche is eaten from both sides

Knowledge demand has two timescales, both already served:

- **Immediate** ("remember what I said an hour ago") — served essentially perfectly by
  context, token-space memory (MemGPT/Letta lineage,
  [arXiv:2310.08560](https://arxiv.org/abs/2310.08560);
  [Mem0](https://arxiv.org/abs/2504.19413)), and KV reuse. Cheap, exact, auditable.
- **Durable** ("internalize this corpus/persona/skill") — served by **offline
  consolidation**: batch distillation, Cartridges-style self-study, scheduled LoRA
  fine-tuning with replay ([continual-pretraining recipe](https://arxiv.org/abs/2403.08763)).
  Controllable, replayable, evaluable *before* deployment, with rollback.

An online parametric store's unique selling point is covering the gap between "now"
and "tonight's batch job" — and that gap has little commercial value because context
already covers it. This is the complementary-learning-systems story with "sleep
consolidation" implemented as a cron job instead of an online write rule. The squeeze
fails — i.e., the niche becomes real — only where:

1. **No batch infrastructure exists** (on-device / personal AI), or
2. **The abstraction claim holds**: parametric memory composes and generalizes over
   what it absorbed in ways retrieval cannot. Indirect support:
   [LoRA-vs-KV-cache analysis](https://arxiv.org/abs/2606.05698) finds document-LoRA
   acts as *decoding-time parametric memory*, strongest exactly where retrieved
   context is absent; conversely
   [context suppresses parametric knowledge](https://arxiv.org/abs/2410.08414) shows
   context-resident knowledge *displaces* rather than integrates with weights.

`continual_memory_direction.md`'s pivot from episodic recall to semantic compression
is precisely a retreat to the territory the squeeze cannot reach.

## 5. Deployment blockers (why no product ships it)

- **Serving:** per-user mutable neural state breaks request batching. Per-user LoRA at
  least has serving infrastructure ([S-LoRA](https://arxiv.org/abs/2311.03285));
  per-user evolving MLP state has nothing.
- **Security:** a persistent online-written memory is a **persistence mechanism for
  prompt injection** — write once, influence every future session, invisibly.
- **Compliance:** right-to-erasure is near-impossible in a superposed store; token-
  space memory is the only paradigm with native audit/delete.

## 6. The sociology: an empty Venn intersection plus a benchmark vacuum

The people who can train architectures (the Titans/TTT groups) do *sequence-model*
research where per-sequence reset is definitionally fine; their metric is perplexity
vs. context length. The people who care about cross-session agent memory standardized
on API-served frontier models where backprop is unavailable, so they build token-space
systems by necessity; their benchmarks (LoCoMo,
[LongMemEval](https://arxiv.org/abs/2410.10813)) are constructed so text retrieval can
ace them. **No benchmark anywhere rewards weeks-long streams, retention curves, or
generalization over absorbed experience** — and benchmarks steer fields. The
intersection of "trains models" and "cares about persistent agents" is approximately:
academic labs with small open models. Some of the emptiness is graveyard (negative
results unpublished), but much is simply an unpopulated intersection.

## 7. The adjacent alternative: sequential LoRA (and its one missing component)

The obvious competing design — train a LoRA online during inference, monitor
saturation, freeze it, spawn the next, offload, route over the library — exists in
pieces but has never been assembled as a closed loop:

| Component | Exists as |
|---|---|
| Online training + spawn-on-signal | [Online-LoRA](https://arxiv.org/abs/2411.05663) (WACV 2025): new LoRA at each detected loss plateau |
| Sharding + routing | [WISE](https://arxiv.org/abs/2405.14768); MELO (arXiv:2312.11795, AAAI 2024): LoRA blocks indexed by an internal vector DB, activated per-input |
| Library retrieval at inference | [LAG](https://arxiv.org/abs/2507.05346) (SVD-based routing over large adapter libraries); [COLA](https://arxiv.org/abs/2510.21836) (autoencoder retrieval of adapters) |
| Serving thousands of adapters | [S-LoRA](https://arxiv.org/abs/2311.03285) |
| **Saturation monitor ("is this adapter full?")** | **does not exist** — raw material: [spectral imbalance](https://arxiv.org/pdf/2602.00722), [subspace-geometry forgetting law](https://arxiv.org/abs/2603.02224) F = α(1−cos²θ_min)+β |

Why the loop stays open: (a) **the objective problem** — an unlabeled inference stream
gives next-token loss on user text, teaching the adapter typos, the assistant's own
outputs, and boilerplate; even curated small-stream continual pretraining shows
[marginal gains](https://arxiv.org/html/2501.17840v1); (b) **the ops problem** —
backprop on the serving path roughly doubles memory and wrecks latency; (c) **the
composition problem** — independently-trained adapters interfere when merged or
hot-swapped ([Cartridges-at-scale collapse](https://arxiv.org/abs/2606.04557);
[task/knowledge subspace entanglement](https://arxiv.org/abs/2604.26768)), and
choosing which adapter to load is itself a retrieval problem — at which point one has
rebuilt RAG with a lossier, costlier, less auditable artifact.

Each attempted closed-loop version quietly reduces to either offline consolidation or
retrieval; the residual advantage has not been demonstrated on any benchmark anyone
reads.

## 8. Implications for this repo

1. **The pivot in `continual_memory_direction.md` is correct, and it is also the
   field's revealed preference.** Episodic exact recall on a parametric store loses to
   retrieval at every easily-evaluated scale — which is why the survivors (GRACE,
   MELO) made their stores discrete, i.e., *became* retrieval. The defensible claim is
   the one a vector DB structurally cannot make: **lossy compression that generalizes**
   — answering questions whose answers were never stored verbatim but are entailed by
   the absorbed stream.
2. **Baseline against the editing literature, not just RAG.** GRACE, WISE,
   MemoryLLM/M+ are the real comparators. The differentiating claim to make explicit:
   they all require *supervised, delimited edit events* ("here is a fact; install
   it"), while this project's skill is meta-trained to extract what is worth writing
   from an undifferentiated stream.
3. **Budget for wall #1 and measure it.** Meta-train the skill on episodes of length
   L, then run retention/perplexity curves at 10L–1000L: report where drift sets in
   and which mechanisms (forget-gate floor, write-rate annealing with load, periodic
   "sleep" re-anchoring toward θ₀) extend the stable horizon. **Nobody has published
   that curve.** A clean negative result is also a contribution — it would be the
   first quantification of why everyone resets.
4. **The saturation monitor is a separable, low-risk paper.** "How full is this
   store/adapter?" with spectral telemetry, validated against
   `scripts/memory/capacity_probe.py` — useful to this repo's roadmap and to the
   sequential-LoRA design space that the field has not yet assembled.

## 9. One-sentence summary

The void exists because the one point in design space this repo claims — meta-learned
online writes, persisted indefinitely, on a frozen base — is the point whose stability
nobody knows how to guarantee, in a market whose two profitable halves are already
served by context and cron jobs, studied by two communities whose intersection is
nearly empty; that is a hard niche, but it is a niche, not a tombstone.

## References (primary)

- Behrouz, Zhong, Mirrokni — *Titans: Learning to Memorize at Test Time* — [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- Behrouz et al. — *It's All Connected* (MIRAS) — [arXiv:2504.13173](https://arxiv.org/abs/2504.13173)
- Behrouz et al. — *ATLAS* — [arXiv:2505.23735](https://arxiv.org/abs/2505.23735)
- Li, Behrouz et al. — *TNT: Improving Chunkwise Training for Test-Time Memorization* — [arXiv:2511.07343](https://arxiv.org/abs/2511.07343)
- Sun et al. — *Learning to (Learn at Test Time)* (TTT) — [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)
- Tandon, Dalal et al. — *End-to-End Test-Time Training for Long Context* — [arXiv:2512.23675](https://arxiv.org/abs/2512.23675)
- Feng et al. — *In-Place Test-Time Training* (ICLR 2026 Oral) — [arXiv:2604.06169](https://arxiv.org/abs/2604.06169)
- Hartvigsen et al. — *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors* — [arXiv:2211.11031](https://arxiv.org/abs/2211.11031)
- Wang et al. — *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing* — [arXiv:2405.14768](https://arxiv.org/abs/2405.14768)
- Wang et al. — *MemoryLLM: Towards Self-Updatable Large Language Models* — [arXiv:2402.04624](https://arxiv.org/abs/2402.04624)
- Larimar (IBM) — episodic memory-conditioned LLM — arXiv:2403.11901
- Muhtar et al. (MSR) — *StreamAdapter* — [arXiv:2411.09289](https://arxiv.org/abs/2411.09289)
- Sakana AI — *Doc-to-LoRA* — [arXiv:2602.15902](https://arxiv.org/abs/2602.15902)
- *DyPRAG: Dynamic Parametric RAG* — [arXiv:2503.23895](https://arxiv.org/pdf/2503.23895)
- Wei, Li, Marculescu — *Online-LoRA* (WACV 2025) — [arXiv:2411.05663](https://arxiv.org/abs/2411.05663)
- Fleshman, Van Durme — *LoRA-Augmented Generation* — [arXiv:2507.05346](https://arxiv.org/abs/2507.05346)
- Sheng et al. — *S-LoRA* — [arXiv:2311.03285](https://arxiv.org/abs/2311.03285)
- HazyResearch — *Cartridges* — [arXiv:2506.06266](https://arxiv.org/abs/2506.06266); *Cartridges at Scale* — [arXiv:2606.04557](https://arxiv.org/abs/2606.04557)
- *Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache Injection* — [arXiv:2604.03270](https://arxiv.org/abs/2604.03270)
- *ReasonCACHE* — [arXiv:2602.02366](https://arxiv.org/abs/2602.02366)
- Zuo et al. (JHU) — *Rethinking LoRA Memory Through the Lens of KV Cache Compression* — [arXiv:2606.05698](https://arxiv.org/abs/2606.05698)
- Cheng et al. — *Interplay between Parametric and Contextual Knowledge* — [arXiv:2410.08414](https://arxiv.org/abs/2410.08414)
- Steele — *Subspace Geometry Governs Catastrophic Forgetting in LoRA* — [arXiv:2603.02224](https://arxiv.org/abs/2603.02224)
- *Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation* — [arXiv:2602.00722](https://arxiv.org/pdf/2602.00722)
- Packer et al. — *MemGPT* — [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- Chhikara et al. — *Mem0* (ECAI 2025) — [arXiv:2504.19413](https://arxiv.org/abs/2504.19413)
- Wu et al. — *LongMemEval* (ICLR 2025) — [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)
- Ibrahim et al. — *Simple and Scalable Strategies to Continually Pre-train LLMs* (TMLR) — [arXiv:2403.08763](https://arxiv.org/abs/2403.08763)
- Tiwari et al. — *Learning, Fast and Slow* (fast/slow weights framing) — arXiv:2605.12484
