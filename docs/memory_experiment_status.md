# Personal-memory consolidation: status & next steps (2026-06-12)

Where the "do remembered lessons actually change a small model's behavior"
experiment stands, and the plan from here. Branch
`claude/clever-hopper-b6orfo`. All code is committed; all data artifacts
live under the gitignored `runs/sleep/` and `personal_chat_archive/`.

> Provenance note: the day's runs used a 1.5B local policy/judge
> (Qwen2.5-1.5B) on Apple MPS, and an OpenRouter model ladder. The gold
> memory set is `personal_chat_archive/frontier_quality_audit/` — 71
> typed, scoped, evidence-backed memories curated by a frontier model
> over Jake's full claude.ai archive (694 conversations).

## Pipeline (built & validated end-to-end)

| Stage | Module / script | State |
|---|---|---|
| Transcript → episodes (Claude Code) | `rlm/sleep/cc_traces.py` | done, tested |
| claude.ai export → episodes | `rlm/sleep/claude_export.py` | done, tested |
| Frontier memory candidates → resolved evidence | `rlm/sleep/frontier_memories.py` | done (203/203 IDs resolve) |
| Dataset assembly (distill / apply / held-out probes / day-shards) | `scripts/sleep/assemble_memory_datasets.py` | done — 71 distill, 142 apply, 71 probes, 48 day-shards |
| Local judge stack (verify→rubric→refute, cached) | `rlm/sleep/gold.py` | done, canary-validated |
| Judge calibration vs frontier tiers | `scripts/sleep/calibrate_judges.py` | done |
| Model-quality ladder (OpenRouter) | `rlm/sleep/model_ladder.py` + `scripts/sleep/run_model_ladder.py` | done |
| Nightly LoRA + behavioral probes | `scripts/sleep/run_memory_nights.py` | done |

Bugs found and fixed while getting here (all regression-tested): markdown
verdict parsing at 1.5B (×2), refute-gate framing collapse, training-side
all-masked-label NaN (the twin of the eval-side `response_nll` bug),
cross-night model leak → MPS OOM at 172 GB, stale-from-memory model IDs,
and empty-content cache poisoning from hybrid-thinking models.

## Results so far

### 1. Behavioral uptake — the headline (`runs/sleep/memory_nights/`)

48 cumulative nightly LoRAs over the day-sharded memories (final adapter
has seen all 213 apply+distill examples), evaluated on 71 **held-out**
probe scenarios (probe #3 of each memory — never trained on):

| metric | base | adapted | Δ |
|---|---|---|---|
| behavior uptake (does the response exhibit the memory) | 59/71 (83%) | **65/71 (92%)** | **+9 pp** |
| CoT surfacing (principle keywords in reasoning) | 34/71 (48%) | 27/71 (38%) | **−10 pp** |

Behavior flips: +8 gained, −2 lost (net +6). Strongest on
`technical_principle` (13→16) and `workflow_rule` (11→14); flat on
`project_fact` and `behavior_correction`. **The trained behavior shows up
more, but the model talks about it less** — consistent with SFT baking a
lesson into a reflex rather than explicit reasoning.

**Caveats that bound this result (and drive the plan):**
- Base is already at 83% — the frontier-authored probes are too easy for
  a competent 1.5B instruct model, leaving little headroom.
- The grader is the noisy 1.5B base judge (`behavior_marker` YES/NO).
- Single run, cumulative-only, no seed variance, no negative control.
- No held-out-*memory* control: we can't yet separate "learned this
  specific memory" from "got generally more cautious."

### 1b. A2 RE-GRADE — the +9pp headline was a grader artifact (2026-06-12, Opus session)

Re-grading the **same cached responses** with the ladder-proven
`qwen3.6-35b-a3b` judge (`runs/sleep/memory_nights/regraded_summary.json`):

| metric | base | adapted | grader |
|---|---|---|---|
| behavior uptake | 59/71 (83%) | 65/71 (92%) | 1.5B (lenient — discredited) |
| behavior uptake | **3/71 (4%)** | **8/71 (11%)** | 35b-a3b (trusted) |
| CoT surfacing | 34/71 (48%) | 27/71 (38%) | 1.5B |
| CoT surfacing | **0/71 (0%)** | **0/71 (0%)** | 35b-a3b |

The 1.5B grader rubber-stamped almost everything. Under the trusted
grader the real story is: **a 1.5B base almost never exhibits these
behaviors (4%), training roughly triples that (→11%, still low absolute),
and it never visibly reasons about the principle (0% CoT both arms).**
The "+9pp" was noise in a bad instrument; the true effect is a small,
real lift off a near-zero floor, on a model below the capability bar for
the nuanced behaviors and CoT we're probing.

**Decision (drives Phase B):** the 1.5B is too weak a *policy* to be the
main testbed — it can't exhibit nuanced behaviors or produce CoT. Phase B
runs primarily on a stronger small policy (Qwen2.5-7B-Instruct: CoT-capable,
LoRA-trainable on the Colab L4), with the 1.5B kept as a cheap second point
so the curve answers "does memory uptake scale with policy capability?".
All Phase B grading uses 35b-a3b, never the policy itself.

### 2. Curation-quality elbow (`runs/sleep/model_ladder/`)

71 matched + 71 derangement-mismatched triplets per model. Classification
= single-call keep/reject+tier (isolates judgment from stack design);
generation = evidence→memory graded vs the frontier reference by a fixed
non-Claude referee (gemini-3.1-pro).

| model | balanced acc | gold-tier agree | gen coverage | gen equiv |
|---|---|---|---|---|
| qwen3.6-35b-a3b (3B active) | **0.894** | **0.98** | 2.33 | 0.16 |
| qwen3.7-max | 0.887 | 0.88 | 2.16 | 0.17 |
| z-ai/glm-5.1 | 0.845 | 0.88 | 2.34 | 0.27 |
| qwen3.5-122b-a10b | 0.838 | 0.82 | 2.22 | 0.22 |
| anthropic/claude-sonnet-4.6 | 0.782 | 0.96 | 2.10 | 0.18 |
| deepseek-v4-flash | 0.775 | 0.84 | 2.20 | 0.14 |
| qwen3.5-9b / qwen3.6-27b | (errored — None content; fix landed, rerun pending) |

**Full 15-rung ladder (2026-06-12, Opus session)** — `docs/data/model_ladder_full.csv`,
`docs/figures/model_ladder_elbow.png`. Judging balanced-accuracy, top→bottom:
Opus-4.8 0.908 · qwen3.6-27b 0.901 · qwen3.6-35b-a3b 0.894 (tier-agree 0.98) ·
qwen3.7-max 0.887 · glm-5.1 0.845 · qwen3.5-122b-a10b 0.838 · qwen3.5-35b-a3b
0.831 · gpt-5.5-pro 0.796 · sonnet-4.6 0.782 · qwen3.5-9b 0.775 ·
deepseek-v4-flash 0.775 · **then a cliff** to ministral-3b 0.609, granite-4.1-8b
0.535, ministral-14b 0.528, ministral-8b 0.507 (≈ chance). The judging elbow is
sharp: a ~27–35B MoE (qwen3.6) **matches Opus-4.8**, and below ~9B it collapses.
Generation stays hard everywhere (equiv ≤0.27) except **gpt-5.5-pro (0.38 equiv,
coverage 2.87)** — the one model meaningfully better at producing the reference
memory.

Two clean findings (original 8-model run):
- **Curation *judging* has a low, local elbow.** `qwen3.6-35b-a3b` — a
  3B-active MoE that runs on this machine — tops the ladder at 0.894
  balanced / 0.98 tier-agreement, beating both qwen3.7-max and
  sonnet-4.6. For *classification*, you do not need a frontier model.
- **Curation *generation* is unsolved at every scale.** Coverage of the
  reference insight tops out ~2.3/5 and exact-equivalence ~0.2 even at
  frontier. Distilling the *right* durable memory from raw evidence is
  genuinely hard — which is exactly why the frontier_quality_audit
  hand-curation has value. (Partial artifact: evidence often supports
  several valid memories, so equiv under-counts good-but-different ones.)
- Local 1.5B stack baseline for comparison: ~0.58 balanced accuracy
  (`runs/sleep/judge_calibration/`) — 42% matched-keep, blind to tiers,
  worst on behavioral memory types.

## Phase B result — the causal uptake test came back NEGATIVE (2026-06-12)

Policy **Qwen3-4B-Instruct-2507** (latest text-causal Qwen), random 49/22
train/heldout memory split, 3 seeds, graded by 35b-a3b. The controls the
dry-run lacked, finally in place:

| probe set | n | base | adapted (±σ over seeds) | Δ |
|---|---|---|---|---|
| **train** (trained memory, novel probe) | 49 | 0.204 | 0.136 ± 0.025 | **−0.068** |
| **heldout** (memory never trained) | 22 | 0.045 | 0.091 ± 0.037 | +0.045 |
| **negative** (no memory should fire) | 30 | 1.000 | 1.000 ± 0.000 | 0.000 |

- **Specific-learning contrast (train Δ − heldout Δ) = −0.113.** The opposite
  sign of the hypothesis. Training did **not** install the trained memories'
  behaviors; it *lowered* targeted behavior on the trained set while nudging the
  untrained set up from near-zero.
- Reading: trained behaviors (base 0.20) fell and untrained ones (base 0.045)
  rose, both converging to ~0.10–0.14 — a **generic style shift, not
  memory-specific learning**. The adapter changed *how* the model writes without
  installing *which* behavior to exhibit.
- CoT ≈ 0 throughout (the 4B doesn't visibly reason about the principle).
- Negative-control delta is 0 — no over-application (the one clean positive).

**Caveats:** small n (49/22) with wide error bars; an unexplained base-rate
asymmetry (train 0.20 vs heldout 0.045) muddies the train-vs-heldout contrast;
absolute rates are low (4–20%). But the direction is clear and consistent with
A2: **the distill+apply LoRA-SFT recipe does not produce targeted behavioral
uptake at 1.5B–4B.** Artifacts: `runs/sleep/uptake_q3_4b/`.

## Phase C result — judging is NOT solved on realistic candidates (2026-06-12)

Two parts, both negative, and together they overturn the optimistic A1 read:

1. **Two-stage curator** over 163 cc-archive episodes kept **163/163** — the
   35b-a3b judge rubber-stamps candidates *generated by its own family*; the
   "filter" stage doesn't filter. Output is frontier-polished but unfiltered.
2. **8B judge fine-tune** (Qwen3-8B, on Modal/Colab) on 280 frontier-labeled
   examples: balanced accuracy **0.48** (keep-recall 0.96, reject-recall 0.00 —
   collapses to "always keep"). The keep-heavy trainset (72% keep) wasn't
   class-balanced. **But the zero-shot 35b-a3b scores the same 0.485 on this
   testset** (keep-recall 0.92, reject-recall 0.05).

**The reframe:** the Phase-A1 ladder measured judging on an *easy* task —
matched memories vs **derangement** negatives (a memory paired with a totally
unrelated memory's scenario/response), which is trivially separable. On
*realistic* candidates (memories generated from real episodes, Opus-labeled
keep/reject), **even the 35B is at chance (0.485, keeps everything)**. So the
"judging elbow at 27–35B matching frontier" was an artifact of the negatives;
on the distribution that actually matters for curation, the cheap judge does
**not** discriminate. Reject-recall ≈ 0 across the board is the tell.

### The emerging cross-phase picture (revised)
- **Curation (judging memory-worthiness): only looks solved on easy negatives.**
  35B matches frontier on derangement negatives (0.89) but is at chance (0.485)
  on realistic generated candidates — reject-recall ≈ 0 (Phase A1 + C). The
  cheap judge keeps everything when the negatives are plausible.
- **Generation of good memories: hard** — only frontier-ish (gpt-5.5-pro best
  at 0.38 equiv).
- **Uptake (training a small model to behave per memories): does not work**
  with this recipe at 1.5B–4B (Phase A2 + B).

**Net:** across judging, generation, and uptake, the cheap/small-model versions
of every stage underperform once the task is realistic. The day's honest
finding is that the simple distill→judge→SFT memory loop does not work at these
scales — each stage's apparent success was partly a measurement artifact
(lenient grader in B's first pass; easy negatives in the curation ladder). The
controls (A2 re-grade, held-out-memory split, derangement-vs-realistic
negatives) are what surfaced this — and are the main durable contribution.

## Phase D result — temporal holdout confirms B: no uptake (2026-06-12, Modal H100)

The consolidation thesis test: train cumulatively on memories dated ≤2026-01-01
(33 train), probe memories dated *after* (38 future-heldout, never seen).
Qwen3-4B-Instruct-2507, 3 seeds, 35b-a3b grader. Ran on a **Modal H100** (~7 min
vs ~40 on the Colab L4; artifacts on the durable Modal Volume + HF).

| set | n | base | adapted (±σ) | Δ |
|---|---|---|---|---|
| train (≤cutoff) | 33 | 0.091 | 0.071 ±0.014 | −0.020 |
| **future** (>cutoff, never trained) | 38 | 0.132 | 0.123 ±0.025 | **−0.009** |
| negative | 30 | 1.000 | 1.000 | 0.000 |

Future-memory uptake **−0.009**, specific-learning **−0.011** — both
indistinguishable from zero. Training on past memories does **not** transfer to
future situations; combined with Phase B (random split, −0.113), the
weight-baking consolidation loop shows no measurable behavioral effect at this
scale, whichever way the data is split.

## Final conclusion (all phases complete)

Across A→D, the honest result is **negative for the simple memory-consolidation
loop at 1.5B–4B**, and the value is in the *controls* that revealed why the
naive versions looked like they worked:

1. **Uptake doesn't happen** (A2, B, D): SFT on distilled-memory exemplars does
   not install the target behaviors; effects are flat-to-negative under a
   trustworthy grader, on random and temporal splits alike. No over-application.
2. **Cheap judging doesn't discriminate on realistic candidates** (A1 vs C):
   the 27–35B "elbow" was an artifact of easy derangement negatives; on
   generated+frontier-labeled candidates even 35B keeps everything
   (reject-recall ≈ 0). Fine-tuning an 8B judge on the imbalanced set collapses
   to majority-class.
3. **Generation of good memories is hard** everywhere except the frontier
   (gpt-5.5-pro 0.38 equiv).

What works: the *measurement scaffolding* — held-out-memory and negative
controls, a trusted-grader audit (which overturned the +9pp headline), and
derangement-vs-realistic negative comparison. These are reusable and are the
real contribution. The infrastructure (cc/export loaders, frontier-memory
pipeline, two-phase uptake harness, Modal+Colab GPU runners, HF-durable
adapters) is committed and reproducible.

### Where to take it next (not run)
- Uptake is the bottleneck, not curation: try far-more-examples-per-memory,
  preference/RL on the behavior itself, or in-context memory vs weight-baking.
- Class-balance the judge trainset; build a realistic (non-derangement) judging
  eval as the standard going forward.

## Plan from here (superseded — all phases executed above)

### Phase A — close the loops we already opened (cheap, ~hours)
1. **Finish the ladder curve.** Rerun with the None-content fix to fill
   `qwen3.5-9b` / `qwen3.6-27b`, and add a sub-10B rung or two + a
   frontier ceiling (opus/gemini-pro) so the elbow is unambiguous.
   Commit the curve as a docs figure.
2. **Harden the probe grader.** Re-grade the existing 71×2 probe
   responses with `qwen3.6-35b-a3b` (the ladder-proven judge) instead of
   the 1.5B base — the +9pp uptake number is only as trustworthy as its
   grader. Cached responses make this judge-only, no retrain.

### Phase B — make the uptake result *mean* something (the real experiment)
The day's probe run was a plumbing dry-run; it has measurement problems.
Tighten into a real causal test:
3. **Held-out-memory control.** Split the 71 memories: train on set A,
   probe on set B (never-trained). Behavior gain on B = "got generally
   more careful"; gain on A−B gap = "learned the specific lesson." This
   is the load-bearing experiment.
4. **Negative control.** Probe scenarios where *no* memory should fire —
   measure whether the adapter over-applies (false-positive behavior /
   becomes uniformly hedgy).
5. **Harder probes + independent-vs-cumulative.** Author tempting probes
   that beat an 83% base; run both training modes; ≥3 seeds for variance.
6. **Recover the CoT signal.** The −10pp CoT drop is interesting — test
   whether training on the *apply* exemplars (which show reasoning) vs
   *distill* examples changes it, and whether a "think before answering"
   probe prompt restores it.

### Phase C — productionize curation, scale the corpus
7. **Two-stage curator:** local `qwen3.6-35b-a3b` for bulk keep/reject
   (0.894, free-ish) + a frontier pass only for generation/borderline.
8. **Fine-tune a sub-35B judge** against the 142-decision calibration set
   + ladder labels — can we get 0.89 cheaper than 35B? (Jake's original
   "fine-tune SLMs against gold" goal.)
9. **Scale gold:** run the validated pipeline over the *full* CC-traces +
   chat archive (thousands of episodes) for a large training set.

### Phase D — the continual-memory thesis
10. **True temporal holdout:** train through day N, test whether the
    agent behaves per a day-N memory on a day-N+1 task *without being told
    the memory* — the actual "consolidation works" claim. The behavioral
    probe harness is the eval; this just runs it as a real time-split.

## How we'll know it worked (success criteria)
- **Phase B is the bar:** adapted beats base on held-out-memory probes by
  a margin that clears seed noise, *and* the trained-memory set shows a
  further gain over held-out (specific learning, not just generic
  caution), *and* the negative control does **not** regress (no
  over-application). A frontier-judge win-rate (adapted vs base responses,
  blind) on the probe scenarios is the human-legible version.
- **Curation is "good enough to scale"** when the two-stage curator's
  keep/reject agrees with frontier tiers at ≥0.9 balanced accuracy at a
  cost that makes thousands of episodes affordable — already in reach
  with 35b-a3b.

---

# Phase E — breadth + scale (the reversal), 2026-06-13

Phases A–D were a **negative** result for the *narrow* recipe (49 memories ×
~3 exemplars, 4B only): specific-learning −11pp, and the +9pp headline was a
grader artifact. Phase E changes two variables Jake flagged: **breadth** (many
*distinct* memories, ~1 exemplar each, not depth-per-memory) and **policy
scale** (sweep sizes, not 4B only).

## Recipe
- **Breadth data** (`scripts/sleep/gen_breadth_data.py`): one frontier (Opus)
  call per claude.ai export episode → a `(durable memory, novel scenario,
  ideal response)` triple. 800 episodes → **555 distinct-memory apply-exemplars**
  (`runs/sleep/breadth/sft_breadth.jsonl`). One exemplar per memory.
- **Train**: LoRA (`all-linear` for dense policies; attention-only for the MoE)
  on the 555 exemplars, 3 seeds, on Modal H100 (`scripts/memory/modal_gpu.py`).
- **Eval = transfer**: the **71 held-out frontier-memory probes** (`fm_*`),
  ID-disjoint from the `cai_*` training memories — the model is graded on
  applying memories it was *never trained on*. Plus 30 negative-control probes.
- **Grading**: the trusted `qwen3.6-35b-a3b` judge **and** an independent
  `deepseek-v4-flash` judge (different family) — behavior, blind win-rate,
  over-application. (`scripts/sleep/run_uptake_experiment.py --train-file ...`.)

## Result — uptake is real and scales with policy size

| policy | size | base | Δ behavior (qwen) | Δ (deepseek) | win-rate (qwen) | win-rate (ds) | over-apply (q/ds) |
|--------|------|------|-------------------|--------------|-----------------|---------------|-------------------|
| Qwen3-4B | 4B | 0.127 | +0.061 | +0.042 | 0.592 | 0.681 | −0.022 / −0.011 |
| Olmo-3-7B | 7B | 0.127 | +0.085 | +0.094 | 0.671 | 0.710 | −0.011 / +0.000 |
| Olmo-3.1-32B | 32B | 0.141 | +0.108 | +0.127 | 0.747 | 0.732 | +0.000 / +0.000 |

Figure: `docs/figures/breadth_size_sweep.png`; data: `docs/data/breadth_size_sweep_2026-06-13.csv`.

## Verdict — **robust with caveats** (adversarially pressure-tested, medium confidence)

A 4-lens adversarial panel (arch-confound, low-absolute-rate, shared-judge-bias,
transfer-leakage) tried to refute the finding; all four returned *minor* — none
overturns it, but each sharpens the framing. What the evidence **supports**:

- **A transferable, selective disposition is installed.** On 71 ID-disjoint
  probes never trained on, the adapted policy applies the target behavior more
  than its own base, and the **blind win-rate is significant at every size under
  both judges** (deepseek binomtest p = 0.0018 / 0.0003 / 0.00006 for 4B/7B/32B).
  The **win-rate is the load-bearing metric.**
- **The effect strengthens with policy size** — win-rate and behavior-Δ both rise
  monotonically under both independent judges. The **cleanest size-only contrast
  is within-Olmo 7B→32B** (same architecture family), where it still rises.
- **Not a verbosity artifact** (adapted is *shorter* than base at every size),
  **not generic over-firing** (negative-control over-application ≈ 0, exactly
  0.000 at 32B), and the two judges agree at the *item* level (91.3%, κ = 0.714).
- This is a genuine **reversal** of the Phase A–D negative.

Honest **caveats** (do not overstate):

- **Low absolute ceiling**: base ~13%, best adapted ~25% — the policy still fails
  the strict binary test ~75% of the time.
- **The binary Δ is fragile at the two smaller rungs**: paired-bootstrap 95% CIs
  cross/touch zero (4B [−0.033, 0.122], 7B [−0.000, 0.188]); only 32B
  [0.052, 0.207] is individually significant. Net gains are single-digit
  probe-flips (~3 / 6.7 / 9 of 71). **Headline the win-rate, not the binary Δ.**
- **Arch × size only partially crossed**: 3 points with one vendor switch on the
  4B→7B leg; the 4B-Qwen rung corroborates but is arch-confounded. Even the clean
  leg is Olmo-3 (7B) vs Olmo-3.1 (32B, a point release).
- **"Held-out" overstates independence**: `cai_*` train and `fm_*` probes are from
  the *same user's history* and share disposition families (~32/71). Leakage can
  inflate the gain's *level* but not the *size trend* (which survives on the
  most-novel subset). Frame as "transfer to held-out-but-overlapping probes."
- **Residual LLM-judge-modality risk**: two LLM judges can't exclude a shared LLM
  prior; under strict AND-consensus the gain stays positive at every size but the
  exact 7B-vs-32B ordering is within noise (growth solid, ranking not).

## Next (ranked, from the panel)
1. **Cross arch × size** — add a Qwen point at larger size (Qwen3-30B-A3B run in
   flight) so each architecture spans ≥2 sizes; converts "monotonic from 4B" from
   partially-confounded to a properly crossed design. *Highest value.*
2. **Human/non-LLM spot-check** on ~30–50 probes to close the LLM-judge modality.
3. **More seeds + pre-register win-rate as primary**, binary Δ as secondary.
4. **Embedding-based leakage analysis**; re-report the gain on a semantically-novel subset.
5. **Graded/partial-credit behavior score**; test whether more data/exemplars
   raises the ~25% adapted ceiling (practical usefulness vs mere detectability).

**Compute/infra notes**: Modal H100 for the sweep; the OpenRouter judge needs a
`max_tokens` cap or it 402s on a low balance (fixed in `OpenAIClient`); `all-linear`
LoRA OOMs/crawls on large MoE — use attention-only there. See the
`openrouter-and-moe-lora-gotchas` memory.

---

# Phase F — lesson-centric reframe + within-lesson validation (2026-06-14)

## Why we reframed (Phase E was the wrong instrument)

Phase E's held-out-memory transfer (+6–11pp, scaling with size) was real but
measured the **wrong thing**. The memory text is **never in the prompt** (neither
training nor eval), so any held-out gain can only be a learned **disposition
prior**, not memory-conditional recall. And transfer only happened because the 71
"held-out memories" aren't different things — they collapse onto a handful of
recurring dispositions that the training set also hammers. So Phase E measured
**generic disposition uptake (≈ instruction tuning), not consolidation of specific
memories**, and "transfer to held-out memories" oversold it.

The fix: relabel the unit of analysis from **"memory" → "lesson"**. Tag every
memory with a hidden lesson label; build **within-lesson** hold-out splits (train
on some instances of lesson L, test on *held-out instances of L*) and per-lesson
learning curves; add a **leave-one-lesson-out ablation** to separate "learned
lesson L" from "got generically better."

## The lesson taxonomy (`docs/lesson_taxonomy.md`)

All 626 memories tagged by **Opus sub-agents** (a deliberate *pristine-data
high-water-mark* — establish whether the approach works with best-possible labels
before scaling to cheaper taggers) into a nested taxonomy: **16 coarse (l1) / 68
medium (l2) / 234 fine (l3)** + 14 domains + a disposition/content flag. **552
disposition / 74 content.** Usable signal lives at **l1 (every lesson ≥19
instances)** and the **l2 head (~29 lessons ≥10)**; l3 is too sparse (median 2).
Data: `docs/data/lesson_taxonomy.json`, `docs/data/lesson_tags.jsonl`.

## First validation — 7B within-lesson uptake (RunPod A40)

Trained **Olmo-3-7B-Instruct** (LoRA all-linear, 3 seeds) on **387** within-lesson
train instances; evaluated on **168 held-out instances of the SAME 16 l1 lessons**.
Graded by **Opus sub-agents** (blinded behavior + win-rate) — the pristine grader,
**not** the OpenRouter judge (grader quality is load-bearing; a weak grader gave a
false +9pp in Phase A2).

**Overall: base 0.26 → adapted 0.61 (+35.7pp); blind win-rate 0.72 (105–40);
negative control clean (over-application 0.033 → 0.000).** That's ~4× the Phase-E
cross-lesson effect — within-lesson generalization is large and real.

**Per-lesson confirms the theory — three buckets:**
- **Dispositions transfer strongly:** precise-distinctions +100, quantitative-rigor
  +90, interaction-discipline +88, intellectual-pushback +60, step-by-step +60,
  mechanism-rigor / options / explanatory-style +43.
- **The content lesson does NOT transfer:** `contextual-tailoring` **+0** (applying
  Jake's *specific facts* — you can't generalize a fact you never saw). Exactly as
  predicted for the `content` kind.
- **A real cost:** `voice-and-style` **−40pp** — disposition SFT homogenizes voice,
  hurting "preserve Jake's style / avoid LLM tics."
- **Hard/subtle lessons stay weak:** verification +19 (real verifying needs tools it
  can't use at eval), epistemic-honesty +5.

Caveats: Opus grader is primary (OpenRouter second-judge cross-check is a
nice-to-have); seed-0 graded; a few lessons are small-n (voice 5, workflow 3); the
within-lesson eval is "easier" than cross-lesson **by design** — that's the point.

## Plan from here

1. **Infra:** bake a RunPod **Docker image** (deps + `WANDB_API_KEY`) + a small
   `runpod_gpu.py` runner; add `wandb.log` to the SFT loop in `rlm/sleep/lora.py`.
   Stops the per-pod reinstall/whack-a-mole; gives live tracking. (See the
   `runpod-gpu-mechanics` memory.)
2. **13B point:** Mistral-Nemo-12B (clean non-thinking dense) on the image, with
   wandb — confirm the effect holds/strengthens at 13B.
3. **The real experiment:** the **{lesson × #instances × leave-one-lesson-out}**
   learning-curve + ablation at l1 (and the l2 head). The ablation separates
   *specific learning* from generic uplift; the learning curve shows how many
   instances per lesson are needed before a lesson reliably installs.
4. **Follow-ups the per-lesson view surfaced:** (a) the **voice-and-style
   regression** — does disposition SFT trade off style preservation? a targeted
   probe; (b) **verification / epistemic-honesty weakness** — unlearnable via plain
   SFT, or do they need tools / CoT / preference data? (c) the **content 14%** —
   route to a different method (in-context, or knowledge-editing); don't pool it into
   the curve.
5. **Then scale down:** smaller policies + cheaper taggers/graders, measuring how
   much quality degrades from the pristine Opus high-water-mark.

> **Where to start (fresh reader):** `rlm/sleep/README.md` (what this is, plain
> language) → this doc (full running log + plan) → `docs/lesson_taxonomy.md` (the
> lesson index). This is the **active** track. The `docs/memory_START_HERE.md`
> /capacity/Titans docs are a **separate** parametric-memory line — see their banners.
