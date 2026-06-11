# The learning signal: what to learn from in an agent's daily experience

> **Status: literature positioning + pipeline proposal (2026-06).**
> Third in a series. `memory_field_positioning.md` established *why* the design point
> this repo occupies is empty; the consolidation discussion established *how* a nightly
> "sleep" stack should operate. This doc addresses *what to learn from* — the
> supervision-signal problem: given a day of raw agent experience, which fragments
> deserve gradient, and where does the training signal come from when nobody labeled
> anything? Synthesized from the 2022–2026 literature on data selection, self-taught
> reasoning, agent self-evolution, and self-adapting models.

## 0. Core thesis: next-token prediction on raw experience is the wrong objective

Pretraining's magic — "just predict the next token and intelligence falls out" — works
only at massive scale, over a corpus whose diversity launders its noise. Everything the
field does *after* pretraining already concedes the point: instruction tuning is
supervised; RLHF/RLAIF injects a curated, low-entropy reward signal on top of the
high-entropy text distribution. Nobody continues raw next-token training on whatever a
deployed model happens to see, because at small scale the objective teaches typos,
boilerplate, and the model's own outputs back to itself.

Learning from an agent's daily stream is therefore a *signal extraction* problem, not
a training-loop problem. The proposed pipeline:

1. **Cheap surprise-based filter** over the day's episodes — which moments were
   informative at all?
2. **Expensive test-time-compute reflection** on the flagged subset — what is the
   actual lesson?
3. **Verification gate** — is the candidate lesson true, or merely plausible?
4. **Distill into weights** — consolidate what survives.

This mirrors human fast/slow cognition: most experience is "nothing new here" and is
processed reflexively; deliberate reflection is expensive and applied selectively. The
remainder of this doc maps each stage onto its published neighbors and locates the
gaps.

## 1. The enabling principle: the generation–verification gap

Humans do not require smarter-than-self teachers to improve; neither do the methods
below. The asymmetry that makes self-improvement possible is that **checking is easier
than doing**: a model that cannot reliably *produce* a correct solution can often
*recognize* one — via outcomes ("the test passed"), consistency ("three samples
agree"), executable checks, or hindsight ("now that I know the answer, this reasoning
clearly led to it"). Every method in this doc is a different machine for converting
that asymmetry into training signal. In the self-supervised setting, verification
asymmetry plays exactly the role that curated reward plays in RLHF — it is the
low-entropy injected source without which the loop is just a model talking to itself.

## 2. Stage 1 — the cheap filter: surprise / learnability selection

The data-selection literature has already established, below the episode level, that
selective training beats uniform training:

- **RHO-LOSS** ([Mindermann et al., ICML 2022](https://arxiv.org/abs/2206.07137))
  formulates the criterion directly: select points that are *learnable, worth
  learning, and not yet learnt* — high loss under the current model but low loss under
  a reference model. Surprising **and** tractable; irreducible noise is excluded
  because the reference model also fails on it.
- **RHO-1 / Selective Language Modeling**
  ([arXiv:2404.07965](https://arxiv.org/abs/2404.07965)) applies this at token level:
  compute excess loss against a reference model, train **only on those tokens**, and
  beat uniform next-token training — a direct empirical refutation of "just do
  next-token prediction on everything."
- **SuRe** ([arXiv:2511.22367](https://arxiv.org/abs/2511.22367)) uses surprise-driven
  prioritized replay for continual LLM learning — the same principle on the
  consolidation axis.
- **TLM** ([arXiv:2505.20633](https://arxiv.org/abs/2505.20633)) finds that
  high-perplexity samples are the most informative for test-time adaptation.

Titans' surprise metric ([arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) — the
write signal already implemented in this repo's memory stack — is the same idea
operating *reflexively, per-token, inside the forward pass*: System 1's flinch. The
open move is to **elevate selection from the token level to the episode level**:
score whole interactions ("this debugging session was surprising; this routine lookup
was not") and let the score gate downstream reflection compute. No published treatment
of episode-level selective learning exists.

## 3. Stage 2 — slow reflection on the flagged subset

Given the small flagged subset, the second stage spends real test-time compute asking
*what the lesson is*. The lineage:

- **STaR** ([arXiv:2203.14465](https://arxiv.org/abs/2203.14465)): generate
  rationales, keep those that led to correct answers; for failures, *rationalize
  backward* from the known answer — hindsight converts failure into supervision.
  "Things done wrong or things that required multiple tries" is exactly STaR's signal
  source: the eventual success labels the earlier failures.
- **Quiet-STaR** ([arXiv:2403.09629](https://arxiv.org/abs/2403.09629)): interleave
  thought generation with prediction and apply REINFORCE on whether the thought
  improved next-token prediction — the model *learns where thinking pays off*. This is
  the closest existing thing to a learned fast/slow gate.
- **Reflexion** ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) and **ExpeL**
  ([arXiv:2308.10144](https://arxiv.org/abs/2308.10144)): verbal reflection over agent
  trajectories, extracting lessons from successes *and* failures — but the lessons are
  stored in token space (memory buffers, insight lists), never written into weights.
- The 2025–2026 agentic wave industrialized the pattern: **EvolveR**
  ([arXiv:2510.16079](https://arxiv.org/abs/2510.16079)) runs a full experience
  lifecycle — offline self-distillation of trajectories into reusable strategic
  principles, retrieved online to guide future behavior. **STeP** (May 2025) builds
  self-reflective error-corrected trajectories with explicit error marking and loss
  masking, so the model trains on the correction without imitating the mistake.
  **SE-Agent** (Aug 2025) maintains a pool of trajectories that are revised,
  recombined, and refined against each other. Overviews:
  [self-evolving LLM-based agents](https://www.emergentmind.com/topics/self-evolving-llm-based-agents)
  and [Learning to Self-Evolve](https://arxiv.org/pdf/2603.18620). Relatedly,
  [Retrieval-Augmented LLM Agents: Learning to Learn from Experience](https://arxiv.org/pdf/2603.18272)
  shows SFT plus experience retrieval beats either alone — weights and token-space
  memory are complements, not rivals.

## 4. Stage 3 — distill the reflection into weights

Reflection output is still tokens. The final stage makes it permanent:

- **Distilling System 2 into System 1** (Yu et al., Meta, 2024,
  [arXiv:2407.06023](https://arxiv.org/abs/2407.06023)): run expensive deliberate
  reasoning offline, then train the model to produce the conclusions *directly*,
  without the intermediate tokens. Applied on a daily cadence, this is the literal
  mechanism of "sleep on it and wake up knowing" — yesterday's laborious chain of
  thought becomes tomorrow's reflex.
- **SEAL — Self-Adapting Language Models**
  ([arXiv:2506.10943](https://arxiv.org/abs/2506.10943), MIT;
  [project page](https://jyopari.github.io/posts/seal),
  [code](https://github.com/Continual-Intelligence/SEAL)): given new information, the
  model writes its own "self-edit" — restructured training data, augmentations, even
  optimization hyperparameters — which is then applied as an actual weight update. The
  self-edit *policy* is trained by RL with post-update performance as the reward:
  the model learns what kind of writing-to-self actually sticks. Notably, the model's
  self-generated training data outperformed GPT-4.1-written synthetic data for
  updating itself. SEAL is essentially this repo's "meta-trained skill that knows how
  to write into the store," implemented in token space with SFT as the write
  primitive.

## 5. What remains uncovered — the open territory

Three gaps, in increasing order of ambition:

1. **Everything above runs where ground truth exists** — QA accuracy, unit tests,
   environment reward. A deployed agent's daily stream offers only *weak* verification
   signals: user corrections, retry-until-success patterns, downstream task
   completion, tool errors, cross-session consistency. The full
   surprise→reflect→verify→distill loop on **unlabeled lived experience** has not been
   built; every published instance smuggles in a benchmark.
2. **The episode-level two-stage filter has no formulation.** A cheap surprise gate
   passing the flagged ~2% of episodes to expensive reflection is both the cognitive
   architecture (System 1 deciding when to wake System 2) and the *economic enabler*:
   targeted reflection compute at 3am is cheap precisely because the gate keeps it
   targeted. RHO-LOSS gives the criterion; nobody has lifted it from token to episode.
3. **The closed loop with persistence is missing.** EvolveR-class systems distill
   into retrievable token-space principles, not weights; SEAL updates weights per-task,
   not as an accumulating lifelong process with gating and rollback. Silver & Sutton's
   ["The Era of Experience"](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
   (2025) is the manifesto for this direction — agents must learn from their own
   experience streams rather than ever-larger human datasets — but it is a position
   statement, not a system.

## 6. The caution: self-reinforcing loops entrench their own errors

A loop that selects its own data, generates its own lessons, and consolidates them
into its own weights will amplify whatever it gets wrong — and there is now a
*security* literature on exactly this failure mode.
[Zombie Agents](https://arxiv.org/pdf/2602.15654) demonstrates persistent control of
self-evolving agents via injections that survive **because** the agent consolidates
them: the consolidation machinery is the persistence mechanism. (This is the
weaponized version of the prompt-injection-persistence concern raised in
`memory_field_positioning.md` §5.)

Conclusion: **the verification gate is the load-bearing component** of the pipeline.
Surprise tells you where to look; reflection generates a candidate lesson; only
verification earns it a gradient. Any implementation that weakens stage 3 to ship
stages 1–2 faster has built an error amplifier with a cron schedule.

## 7. Tie back to this repo

- **Titans-style surprise is the System-1 per-token write signal already implemented
  in `rlm/memory`.** This doc's pipeline is its System-2 counterpart at the
  episode/day timescale: the same "high gradient norm means informative" instinct,
  promoted from a forward-pass reflex to a nightly triage policy.
- **The two-stage filter is RHO-LOSS's criterion lifted from token to episode level**
  — high surprise under the current self, but tractable under reflection. That
  formulation does not exist in the literature and is small enough to build and
  evaluate independently of the parametric-store track.
- **Engineering as data collection for the research program.** If the practical stack
  gets built — surprise gate, reflection jobs, verification, distillation — it
  generates exactly the supervised traces ("what we chose to consolidate, and how it
  went") that a meta-learned parametric writer would need for training. The cron-job
  version is not a detour from the meta-learning agenda; it is the instrument that
  collects its training data.

## 8. One-sentence summary

Raw next-token prediction on lived experience is noise at agent scale; the viable
objective is a pipeline — surprise to find the moments, reflection to extract the
lesson, verification to earn the gradient, distillation to keep it — and while every
stage exists in isolation under benchmark supervision, the episode-level filter and
the closed loop on unlabeled experience are unbuilt, which is precisely the opening.

## References (primary)

- Mindermann et al. — *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt* (RHO-LOSS, ICML 2022) — [arXiv:2206.07137](https://arxiv.org/abs/2206.07137)
- Lin et al. — *RHO-1: Not All Tokens Are What You Need* — [arXiv:2404.07965](https://arxiv.org/abs/2404.07965)
- *SuRe: Surprise-driven Prioritized Replay for Continual LLM Learning* — [arXiv:2511.22367](https://arxiv.org/abs/2511.22367)
- *TLM: Test-time adaptation via high-perplexity sample selection* — [arXiv:2505.20633](https://arxiv.org/abs/2505.20633)
- Behrouz, Zhong, Mirrokni — *Titans: Learning to Memorize at Test Time* — [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- Zelikman et al. — *STaR: Self-Taught Reasoner* — [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- Zelikman et al. — *Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking* — [arXiv:2403.09629](https://arxiv.org/abs/2403.09629)
- Shinn et al. — *Reflexion: Language Agents with Verbal Reinforcement Learning* — [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- Zhao et al. — *ExpeL: LLM Agents Are Experiential Learners* — [arXiv:2308.10144](https://arxiv.org/abs/2308.10144)
- *EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle* — [arXiv:2510.16079](https://arxiv.org/abs/2510.16079)
- *Learning to Self-Evolve* — [arXiv:2603.18620](https://arxiv.org/pdf/2603.18620)
- *Retrieval-Augmented LLM Agents: Learning to Learn from Experience* — [arXiv:2603.18272](https://arxiv.org/pdf/2603.18272)
- Yu et al. (Meta) — *Distilling System 2 into System 1* — [arXiv:2407.06023](https://arxiv.org/abs/2407.06023)
- Zweiger et al. (MIT) — *SEAL: Self-Adapting Language Models* — [arXiv:2506.10943](https://arxiv.org/abs/2506.10943); [project page](https://jyopari.github.io/posts/seal); [code](https://github.com/Continual-Intelligence/SEAL)
- Silver & Sutton — *The Era of Experience* (2025) — position paper
- *Zombie Agents: Persistent Control of Self-Evolving LLM Agents* — [arXiv:2602.15654](https://arxiv.org/pdf/2602.15654)
- Overview: [Self-Evolving LLM-Based Agents](https://www.emergentmind.com/topics/self-evolving-llm-based-agents)
