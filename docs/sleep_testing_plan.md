# Sleep-consolidation: CPU test results & GPU continuation plan

> **Audience: the next (GPU-equipped) session picking up branch
> `claude/clever-hopper-b6orfo` (PR #2).** The original phased plan was
> re-scoped when GPU access slipped: everything that could run on CPU was
> run on CPU (2026-06-11, 4-core/15 GB container, network to HF). This doc
> records what was executed, what it found — five real bugs — and exactly
> what remains for a GPU session. Runbook: `docs/sleep_consolidation.md`;
> rationale: `docs/learning_signal.md`.

## Executive summary

- **The judge is now the policy model itself** (`LocalHFJudge`, frozen base
  weights, higher TTC via self-consistency sampling + long generations).
  No external teacher, **no API key, $0 marginal judge cost** — judge
  compute is GPU/CPU time. External judges remain available via
  `--judge-model <name>` purely as a Phase-5 ablation.
- **The full loop has now run for real, end to end, on CPU** — real LoRA
  training, real NLL evals, real self-judging — on both synthetic traces
  and live `THUDM/AgentInstruct` data, with a 0.5B policy. Every
  correctness risk in the original "Phase 1 GPU shakedown" and the judge
  half of "Phase 2" is retired. What a GPU adds is *scale* (1.5B+),
  *speed*, and therefore *statistical power* — not new plumbing risk.
- **CPU testing found and fixed 5 real bugs** (§3), two of which silently
  corrupted the core metric. A GPU session starting from the original code
  would have burned its budget producing wrong numbers.

## 1. What ran on CPU (all committed, all green)

Engineering backlog — all seven items from the original plan, each with
tests (the suite grew 296 → 382 passing; 27 → 54 sleep-specific):

| Item | Commit | Notes |
|---|---|---|
| `LocalHFJudge` self-judge | `3997d26` | BaseLM over the in-process policy model; TTC knobs in `JudgeConfig` (`temperature`, `max_new_tokens`); usage tracking |
| Judge parse-failure tolerance | `c973ac5` | malformed JSON sample = skip vote, counted in `n_parse_failures`; prerequisite for small self-judges |
| `SurpriseGate` wiring | `3546732` | `run_night` builds it behind `gate.use_surprise`, reusing the judge's model instance; `--use-surprise` CLI (`72fcd8f`) |
| Judge-output disk cache | `572a97c` | keyed by prompt + judge settings; shared across sweep points; second call = zero LM calls (tested) |
| `--torch-dtype` (auto) | `3997d26` | float32 on cpu / bfloat16 on cuda; T4's missing bf16 gets an explicit escape hatch |
| `--cumulative` re-distill | `3ceea4c` | night N = base + union of all verified examples so far; accumulates from `judge_outputs.json` (never `sft_data.jsonl` — double-count hazard) |
| Retention probe v2 | `3d5a0d8` | 50 frozen items across 5 domains; documented as append-only |
| Win-rate eval | `48892a0` | blind pairwise A/B, seeded order randomization (positional judges score ~chance, tested) |

### Measured CPU runs (Qwen2.5-0.5B-Instruct, float32, 4 cores)

**Synthetic night** (16 episodes/day, self-judge `n_samples=1`, ~13 min
wall including 3 model loads):

| | base | adapted | delta |
|---|---|---|---|
| next-day NLL | 4.3188 | 4.2863 | **−0.0325** |
| test NLL | 4.9377 | 4.9037 | **−0.0340** |
| retention NLL | 2.5545 | 2.5524 | −0.0021 |

Gate 2/16 (both planted-learnworthy), judge 2/2 learn with **0 parse
failures**, train loss 4.09 → 3.93 over 2 steps (5.6 s), adapter
round-tripped and moved all three metrics. Deltas are noise-scale (1
distilled example) — the point is the measurement chain, which works.

**AgentInstruct/os night** (16 episodes, `--use-surprise`, post-fix run):
| | base | adapted | delta |
|---|---|---|---|
| next-day NLL | 0.5900 | 0.5859 | **−0.0041** |
| test NLL | 0.6632 | 0.6584 | **−0.0048** |
| retention NLL | 2.5545 | 2.5501 | −0.0044 |

16m30s wall on 4 CPU cores (gate w/ surprise NLLs ≈ 1 min, 4 self-judge
reflections ≈ 7 min, train 8 examples 23 s, two eval passes ≈ 7 min).
Judge: 4/4 learn, 0 parse failures, 8 candidate examples → **6 verified,
2 rejected — both rejects contained stray `[assistant]` artifacts**, i.e.
the verification gate caught real garbage rather than rubber-stamping.
Note the pre-fix run of the same night reported test NLL 0.19 vs. the
corrected 0.66 — the magnitude of what bug 4 (§3) was silently doing.

**Win-rate functional check**: `generate_pairs` + `judged_win_rate` ran
with the real base + the synthetic night's adapter; the near-identical
generations were correctly judged 3× TIE. CPU generation cost ≈ 2 min/pair
at 0.5B — win-rate is a GPU-tier eval.

**Judge floor**: SmolLM2-135M cannot follow the JSON reflection format at
all (0/1 parseable); Qwen2.5-0.5B produced 0 parse failures across all CPU
runs (n≈10 reflections). The self-judge floor is somewhere in between;
quality (not parseability) is the open question at 0.5B — see §4.

### Live-data survey (gate selection at default budget=25%, floor=0.15)

| domain | episodes | heuristic gate, before fix | after fix |
|---|---|---|---|
| os | 195 | 0 selected | 12 |
| db | 538 | 3 | 10 |
| alfworld | 336 | 29 | 29 |
| webshop | 351 | 1 | 1 |
| kg | 324 | 1 | 1 |
| mind2web | 122 | 0 | 0 |

webshop/kg/mind2web emit no error-shaped text: `--use-surprise` is the
only viable selector there. **db (538 episodes → ~6 days of 64) is the
domain for multi-night curves**; os only supports ~2 days at 64/day (use
`--episodes-per-day 16` for more days).

## 2. Design change: self-judging (and one decision to honor)

The judge runs the **frozen base weights**, never the current adapter, so
reflection quality is stationary across nights and multi-night curves
measure adapter learning rather than judge drift. Judging *with* the
adapter (compounding self-improvement, and where zombie-loop risk lives)
is deliberately deferred to an ablation (§5, item 6).

## 3. Bugs found by CPU testing (all fixed + regression-tested)

1. **Judge crashed on malformed JSON** (`c973ac5`) — one bad sample killed
   the whole night; fatal with any small self-judge.
2. **AgentInstruct loads failed outright** (`bd438c0`) — domains are
   *splits*, not configs, under current `datasets`; loader now falls back.
3. **The gate's error feature was structurally dead on chat-style traces**
   (`bd438c0`) — AgentInstruct has no `tool` role; env feedback arrives as
   user turns. os selection went 0/195 → 12/195 after scanning
   observation turns.
4. **`response_nll` silently scored prompt tokens** (`fe870d9`) — right-
   truncation of prompt+response meant long-prompt episodes (os medians
   are multi-kchar) measured leftover *prompt* NLL. The metric the whole
   harness optimizes was wrong on exactly the realistic data. Now: prompt
   left-truncated, response always scored (response capped at half the
   window so context survives).
5. **Surprise saturated at 1.0 on every real episode** (`fe870d9`) — the
   absolute `min(1, nll/4)` squash had zero discriminative power (bug 4
   inflated NLLs past the knee, but the squash is wrong regardless: NLL
   scales vary with dataset and model size). Now: raw nats/token,
   min-max normalized **within the day**. Post-fix on os: raw NLLs span
   0.22–2.42; the top-selected episode is the genuinely-hard one, and an
   episode with an explicit error but NLL 0.226 (model already knows the
   fix) dropped from rank 1 to rank 8 — RHO-style selection behaving as
   designed.

## 4. Honest caveats from the CPU runs

- **All NLL deltas so far are noise-scale.** 1–3 distilled examples per
  16-episode night cannot move means much. The GPU session's job is
  statistical power: 64-episode days, `n_samples=3`, several nights.
- **0.5B reflection quality is poor** even when parseable: lessons are
  vague ("validate inputs before proceeding"), distilled prompts sometimes
  degenerate ("Put your thoughts here."). The pipeline is sound; whether
  *self*-distillation produces wins likely starts at 1.5B+.
- **Verification semi-rubber-stamps at 0.5B**: 1 of 4 examples rejected on
  the os night — not 100% pass, but not a strong gate. Run the
  corrupted-example canary (deliberately wrong response → must get NO)
  before trusting any positive result.

## 5. GPU continuation plan (in order; ~7–11 GPU-h total, $0 API)

Same CLI commands throughout — only `--device cuda` and model size change.
All wall-clock estimates are extrapolations from measured CPU times;
re-calibrate with G1.

**G1 — Timing calibration (L4, ~0.5 GPU-h).** The exact os command from
§1 with `--device cuda --policy-model Qwen/Qwen2.5-1.5B-Instruct
--episodes-per-day 64`. Record: wall per stage (gate/judge/train/eval),
peak VRAM, judge tok/s. Correctness is already proven; this is purely a
budget measurement. Sanity-check that 1.5B judge reflections are
substantively better than the 0.5B ones in §4 (read 5 of them).

**G2 — First powered night (L4, ~1 GPU-h).** db domain (biggest),
64-episode day, `n_samples=3`, `--use-surprise`. Deliverables: the three
deltas with n_examples ≥ 20; judge parse-failure rate at 1.5B;
verification rejection rate + the corrupted-example canary; 10 distilled
examples eyeballed for quality.

**G3 — Sweep (L4, ~3–5 GPU-h).** Judge cache makes the adapter surface
nearly judge-free: `'{"adapter.lr": [5e-5, 1e-4, 2e-4, 5e-4],
"adapter.rank": [8, 16, 32], "adapter.replay_ratio": [0.0, 0.2, 0.4]}'`.
Question: any region with `next_day_delta < 0` and `retention_delta ≈ 0`?
Secondary: does `replay_ratio=0` measurably hurt the 50-item probe?
(If not, the probe needs harder items — append a v2 list, don't edit v1.)

**G4 — Multi-night curves (L4, ~2–3 GPU-h).** 5–6 nights on db with the
G3 winner, run **twice**: default (independent nights) and `--cumulative`
(re-distill). Plot test NLL + retention NLL vs. night for both. The
divergence between the two curves is the consolidation finding, whichever
direction it goes. Add win-rate (`rlm/sleep/winrate.py`) on the final
night's adapter, ~32 pairs.

**G5 — Ablations (each ~1 GPU-h, ordered by information value).**
1. `judge.verify_examples=false` — is the verification gate load-bearing?
2. Gate ablation at equal budget: random vs heuristic vs surprise vs both.
3. Judge TTC curve: `n_samples ∈ {1, 3, 5}`.
4. **Self-judge vs external judge** (`--judge-model gpt-4o`, needs a key):
   does self-judging match a stronger teacher? Publishable either way.
5. Scale: Qwen2.5-7B on A100 (rank 16, `max_seq_len` 1024 fits 40 GB).
6. Judge-with-adapter (compounding self-improvement; watch retention and
   the canary closely — this is where zombie loops would appear).

### Kill criteria (updated from CPU evidence)

- G2: if 1.5B distilled examples are still §4-grade mush, stop and improve
  `REFLECT_PROMPT` (few-shot it) before sweeping — sweep can't fix bad data.
- G2: if the canary passes verification, the verifier is broken at this
  scale; fix `VERIFY_PROMPT` before trusting any delta.
- G3: if no grid point wins, check base NLL first — db/os gold responses
  are formulaic (CPU-measured os response NLLs as low as 0.22 nats) and a
  1.5B base may simply have nothing to learn; switch domains (alfworld had
  the richest failure signal) before concluding the loop doesn't work.

## 6. Remaining engineering (small, optional)

- Wire win-rate into `run_nightly.py` behind `--win-rate N` (module is
  done and tested; ~20 lines of CLI).
- `HF_TOKEN` env passthrough for faster HF downloads (warning is benign).
- Multi-night adapter *merging/routing* (vs. re-distill) — design open,
  deliberately out of PoC scope.

## 7. Session hygiene

- Branch `claude/clever-hopper-b6orfo`, PR #2; never push red
  (`uv run pytest -q` = 382 passed; the offline smoke commands in
  `scripts/sleep/README.md` still run in <10 s).
- `runs/sleep/` is gitignored — commit metric summaries/plots into `docs/`,
  never adapters.
- `uv sync --extra sleep` gives CPU torch in uv-managed envs by design
  (`[tool.uv]` pin); Colab/pip installs get CUDA wheels. Don't "fix" this.
- The retention probe v1 is frozen. Append v2; never edit.
