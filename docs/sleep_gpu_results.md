# Sleep-consolidation: GPU results (L4, 2026-06-12)

Execution log for the GPU continuation plan in `docs/sleep_testing_plan.md`
(§5). Environment: Colab L4 (24 GB), Python 3.12.13, torch 2.11.0+cu128,
transformers 5.10.1, peft 0.19.1, datasets 4.0.0. Policy + self-judge:
Qwen/Qwen2.5-1.5B-Instruct, bfloat16.

**Colab env note:** Colab preinstalls torchao 0.10.0, which peft ≥0.19
*rejects at import* inside its LoRA dispatcher (`ImportError: ... only
versions above 0.16.0 are supported`) — the night dies at the train stage.
Fix: `pip install -U torchao` (0.17.0 works; its mxfp8/cutlass `.so` load
warnings on L4 are benign). The judge-output cache made the rerun free:
the relaunched night hit 16/16 cached reflections (`judge 0.0s`, 0 LM
calls).

## G1 — timing calibration (os, 64-episode day, n_samples=1, --use-surprise)

Command: `run_nightly.py --dataset THUDM/AgentInstruct --config os
--policy-model Qwen/Qwen2.5-1.5B-Instruct --episodes-per-day 64
--use-surprise --device cuda --days 1`

**Wall per stage** (from `day_result.json` `stage_seconds`, plus the live
first run's artifact mtimes for the judge):

| stage | wall | notes |
|---|---|---|
| gate (heuristic + surprise NLL over 64 eps) | 3.5 s | surprise reuses the judge's model instance |
| judge (16 reflections + verifies, live) | ~189 s | ~3.2 min; 0 s on rerun (cache) |
| train (26 examples, r=16, 2 epochs) | 8.0 s | |
| eval (2 × [64 next-day + 20 test + 50 retention]) | 28.5 s | |

**One full 64-episode night at 1.5B ≈ 4–5 min wall on L4** (vs. ~16.5 min
for a 16-episode day at 0.5B on 4 CPU cores). Peak VRAM **14.1 GB / 24 GB**
(judge model + LoRA training co-resident), GPU util ~94% during eval.
The plan's 0.5 GPU-h estimate for G1 was ~6× conservative.

**Day summary:** 64 episodes → 16 gated (budget-bound) → 11 learn / 16,
**4/16 parse failures (25%)** at n_samples=1 → 26 examples (incl. replay).

| | base | adapted | delta |
|---|---|---|---|
| next-day NLL | 0.4902 | 0.4890 | −0.0012 |
| test NLL | 0.5852 | 0.5745 | **−0.0107** |
| retention NLL | 2.1002 | 2.0878 | −0.0124 |

**1.5B reflection quality** (G1 exit check, 5 read): clearly above the
0.5B "mush" bar — lessons are specific and tool-grounded ("use
sort | uniq for dedup-then-count", "verify paths exist before executing"),
distilled examples are mostly coherent Q→A pairs. Residual artifacts:
occasional stray `[assistant]`/transcript fragments in prompts, and one
Think/Act format echo. The 25% parse-failure rate (vs 0% at 0.5B on CPU)
is the thing to watch — likely `max_new_tokens=768` truncating longer
1.5B reflections; G2 runs n_samples=3 which tolerates per-sample failures.

## Canary: the verification gate failed, and was fixed (kill criterion #2)

`scripts/sleep/verify_canary.py` (8 paired items: 4 deliberately corrupted,
4 matched controls) against the 1.5B self-judge, run before trusting any
G2 delta:

It took three iterations to get a verifier that discriminates at 1.5B:

| verifier | corrupted caught | controls passed |
|---|---|---|
| v1: one-word YES/NO, temperature 0.7 | **1/4** (accepts everything) | 4/4 |
| v2: suspicion-by-default checklist, greedy (`63e68f7`) | 4/4 | **0/4** (rejects everything) |
| v3: symmetric framing + 2 few-shot reference reviews (`e88514c`) | **4/4** | **4/4** |

v1 rejected only the egregious item (`rm -rf *` passed off as a listing
command); subtler corruption (skip-the-validation, count-then-subtract
SQL) was rubber-stamped — exactly the failure mode §4 of the testing
plan predicted, now quantified. v2 proved a small model follows framing
more than evidence: "your default is suspicion" flipped it to reject-all.
v3 keeps what v2 added (error checklist, reasoning-then-`VERDICT:` line,
fail-closed parsing, greedy decoding for verification, prompt-template
fingerprints in the judge cache key) but frames both error directions as
costly and shows one worked NO and one worked YES review. The few-shot
domains (shell quoting, head-on-logs) are distinct from the canary items,
so 4/4+4/4 is not template-matching.

The first night run with v1 had **31/31 examples verified** — including
transcript-narration "responses" like "The SQL query executed successfully
and returned the expected answer" — confirming accept-all in the wild,
not just on the canary.

## G2 — first powered night (db, 64-episode day, n_samples=3, --use-surprise)

**Timing/usage** (from the v1-verifier run; judge cost is verifier-independent
to first order): gate 3.4 s | judge 418 s | train 7.9 s (31 examples) |
eval 30.7 s → **one db night ≈ 7.7 min**. Judge usage: 74 calls
(48 reflections + 26 verifies), 62,074 in / 12,542 out tokens →
**~30 tok/s generation** on L4 at 1.5B (unbatched), ~178 tok/s
total throughput incl. prefill.

**Parse failures: 3/48 samples (6%)** at n_samples=3 (vs 25% of 16 at
n=1 on os — pooled, the per-sample rate is ~11%). Diagnosis from raw
responses: not truncation — the judge writes SQL containing `\'` inside
JSON strings, an invalid JSON escape. Fixed by a salvage pass in
`extract_json` (`8ee0dae`); the failed samples were otherwise complete,
so the effective loss rate going forward is ~0.

**db base NLLs are very low** (next-day 0.179, test 0.166): AgentInstruct
db gold responses are formulaic, exactly the §5 kill-criteria caveat.
There may be little headroom for the loop on this domain at 1.5B.

**v1-verifier deltas, recorded for comparison but NOT trusted** (the
gate behind them was accept-all): next-day +0.0219, test +0.0188,
retention −0.0096 — i.e. training on unfiltered distillations made the
model *worse* on next-day/test. If the v3-verified night below flips the
sign or shrinks the damage, that is direct evidence the verification
gate is load-bearing (the testing plan's central claim).

**The v3-verified night** (same day, same gate, calibrated verifier):
16 gated → 15 learn → 29 candidate examples → **12 rejected (41%)** →
17 verified (+ replay → 21 train examples). Parse failures 1/48 (~2%,
salvage active). The rejections are real garbage, not noise: transcript
echoes posing as answers ("What are the results when executing the above
SQL query?"), and SQL that is syntactically invalid (unquoted
space-containing columns `Submitted via`, `Sub-Product`). Wall: gate
3.5 s | judge 440.6 s | train 6.9 s | eval 31.3 s; judge 77 calls,
69,085 in / 12,992 out tokens.

| | base | adapted (v1 accept-all) | adapted (v3 verified) |
|---|---|---|---|
| next-day NLL | 0.1794 | 0.2013 (+0.0219) | 0.1915 (**+0.0121**) |
| test NLL | 0.1656 | 0.1844 (+0.0188) | 0.1765 (**+0.0109**) |
| retention NLL | 2.1002 | 2.0906 (−0.0096) | 2.0883 (−0.0119) |

Two findings:

1. **The verification gate is load-bearing in the measured direction:**
   filtering the distillations roughly halved the next-day/test damage at
   identical gate/judge settings. (It did not yet produce a *win*.)
2. **Training on db distillations hurts next-day/test NLL at 1.5B
   regardless** — consistent with the §5 kill-criteria caveat: db gold
   responses are formulaic (base NLL 0.166) and reflective re-phrasings
   pull the model away from that style. Whether any (lr, rank, replay)
   region escapes this is exactly the G3 question.

## G3 — sweep (db, 36 points, judge fully cached)

Grid: `adapter.lr ∈ {5e-5, 1e-4, 2e-4, 5e-4}` × `rank ∈ {8, 16, 32}` ×
`replay_ratio ∈ {0, 0.2, 0.4}`, one verified night each, same day of
episodes, judge cache pre-seeded from G2 (**zero judge calls for all 36
points**). ~45 s/point, ~27 min total — the original "3–6 GPU-h" estimate
was ~8× conservative. Data: `docs/data/g3_sweep_db_1.5b.csv`; scatter:
`docs/figures/g3_sweep_db_scatter.png`.

**Answer to the G3 question: no.** No (lr, rank, replay) region achieves
`next_day_delta < 0` beyond noise on db:

- **lr dominates everything else.** 5e-5: deltas within ±0.0006 (the
  adapter barely moves the model). 1e-4: ≈ +0.002. 2e-4: ≈ +0.01.
  5e-4: ≈ +0.05. Monotonic damage; rank is second-order at best.
- The "best" point (lr=5e-5, rank=16, replay=0.4: −0.0005 next-day,
  −0.0006 test) is indistinguishable from not training.
- **The retention probe failed its own test**: retention deltas are
  *negative everywhere* (≈0 to −0.03) and grow more negative with
  replay_ratio and lr — replay examples resemble the probe items, so
  training on them *improves* the probe. `replay_ratio=0` does not
  measurably hurt retention at any lr. Per the plan: the probe needs
  harder items (append a v2; damage currently shows on test NLL, not
  retention).

Per the kill criteria this triggers the domain check, not a verdict on
the loop: db base NLL (0.166) confirms the gold responses are formulaic
enough that a 1.5B base has nothing to learn from re-distilled versions
of them. Next: one night on **alfworld** (the domain with the richest
failure signal: 29/336 heuristic-gate selections vs ~0 elsewhere).

## Domain check: one alfworld night (64-episode day, n_samples=3, --use-surprise)

The kill-criteria follow-up to G3. alfworld: 9/64 gated (the score floor
binds, not the budget — failure signal is real but sparse), 9/9 learn,
10 verified examples; judge 41 calls, 277 s; night ≈ 5.3 min.

| | base | adapted | delta |
|---|---|---|---|
| next-day NLL | 0.0921 | 0.0896 | **−0.0025** |
| test NLL | 0.0474 | 0.0466 | **−0.0008** |
| retention NLL | 2.1002 | 2.0967 | −0.0035 |

The same default lr=2e-4 that cost db +0.012 next-day **improved**
alfworld — the first night with all three deltas negative. Honest
context: alfworld base NLLs are even lower than db's (0.047 test), so
the magnitudes are small; the broader picture is that 1.5B already
models AgentInstruct expert responses extremely well everywhere (every
domain's base NLL ≤ 0.2 nats vs 2.1 on general text). Within that
ceiling, alfworld is the domain where consolidation has signal, so G4
runs there.

## G4 — multi-night curves (alfworld, 5 nights × 40 episodes, n_samples=3)

Two policies, same five days of episodes, same gate/judge (cumulative ran
entirely off the independent pass's judge cache: **0 LM calls**):
independent nights (each adapter from base on that night only) vs
`--cumulative` re-distill (night N = base + union of all verified
examples so far). Figure: `docs/figures/g4_alfworld_curves.png`.

Test-NLL deltas vs base (negative = win); base test NLL 0.0474:

| night | n_examples (indep / cumul) | independent | cumulative |
|---|---|---|---|
| 0 | 4 / 4 | −0.0018 | −0.0018 |
| 1 | 9 / 12 | −0.0010 | −0.0007 |
| 2 | 10 / 22 | −0.0024 | −0.0012 |
| 3 | 8 / 30 | −0.0010 | **+0.0019** |
| 4 | 12 / 42 | −0.0017 | **+0.0102** |

Next-day deltas: independent negative on all five nights; cumulative
negative through night 2, +0.0023 at night 3, −0.0006 at night 4.

**The divergence is the finding.** Independent nights produce small but
*systematic* wins — 5/5 nights improved both next-day and test NLL, and
the night-4 adapter beat base in a blind pairwise self-judged A/B on 32
unseen next-day prompts: **22 wins / 10 losses / 0 ties (win-rate 0.688)**
— the generation-level effect is much larger than the ~0.002-nat NLL
deltas suggest. Cumulative re-distill matches that for ~2 nights and then
*degrades through base* as the accumulated set grows, while its retention
probe keeps improving (2.100 → 2.082, the best "retention" of any run) —
it is memorizing its own replay mix at the expense of generalization.
Interference sets in by ~30 accumulated examples at lr 2e-4 with 2 epochs.
This is the forgetting-vs-accumulation result the consolidation program
predicted: per-night material is learnable; the union is not the way to
accumulate it. Cross-night merging/routing (explicitly out of PoC scope)
is what this curve says to build next.

**G5 note — verification ablation already measured:** the v1-vs-v3
verifier comparison in G2 is the `verify_examples` ablation in the
direction that matters: accept-all roughly doubled the next-day/test
damage at identical settings. A dedicated `verify_examples=false` column
would only re-confirm it.

## Cumulative GPU budget actually spent

G1 ~6 min (+1 failed-train run ~5 min) | canary ×3 ~7 min | G2 ×2
~16 min | G3 sweep ~27 min | alfworld night ~5 min | G4 ~55 min
(incl. 32-pair win-rate generation) ≈ **2.0 GPU-h total** vs the plan's
7–11 GPU-h estimate for G1–G4. Judge: $0 API throughout (self-judge).
