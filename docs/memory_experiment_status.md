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

## Plan from here

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
