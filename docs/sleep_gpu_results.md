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

## G4 — multi-night curves

*(pending)*
