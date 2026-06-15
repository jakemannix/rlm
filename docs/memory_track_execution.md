> **Scope:** part of the *parametric-memory* track (frozen base + meta-trained adapter + mutable store) — a SEPARATE line from the active sleep-time / lesson-centric behavioral-memory work (see rlm/sleep/README.md, docs/memory_experiment_status.md).

# Memory-track execution brief (for the Claude Code session)

You are executing the plan in `docs/design_consult_response.md` against the starter
code on this branch. Read that doc first; this brief is about *how* to run, not
*what* to build.

## Scope guard — read this twice

**This repo also implements Recursive Language Modeling (`rlm/core/`, the REPL,
`rlm/environments/`, most of `rlm/clients/`). None of that is in scope.** The
current objective is to verify the Titans-style parametric long-term memory in
isolation — acceptance tests A1–A4 on a frozen Gemma base — *before* any
integration with the RLM machinery. Concretely:

- Work only in `rlm/memory/`, `scripts/memory/`, `tests/test_linear_store.py`,
  `tests/test_episodes.py`, `tests/test_trainer_toy.py`, and run artifacts.
- Do not refactor, "improve", or even reformat `rlm/core/`, the REPL, or clients.
  Do not wire the memory into `rlm/clients/titans_hf.py` yet — that client wraps
  the *old* welded-MLP path and is the A3 baseline arm plus the eventual
  integration point, **after** A1–A4 have verdicts.
- If a memory-track change seems to require touching RLM core, it doesn't; stop
  and leave a note in the run log instead.
- `rlm/memory/titans.py` / `miras.py` / `modeling.py` stay as-is: they are the
  welded-MLP comparison arm for A3, not dead code.

## Boldness calibration

You have an L4 with generous credits and explicit permission for long runs.

- **Do not downscope to save time.** Default sizes are the real ones: d_k = 512,
  ≥ 20k cached episodes, 4k+ training steps, 24-fact acceptance runs. An 8–12 h
  job (A3 grid, base-scale rerun) is fine to launch without asking.
- **Gates are the permission structure.** When a step's gate passes, proceed to
  the next step immediately — do not stop to ask. When a gate fails, consult the
  branch table below, run the named diagnostic, and only stop if the table is
  exhausted.
- **Sweep, don't deliberate.** After the cache exists, training runs are minutes;
  grid `--lambda-kl ∈ {0.1, 0.5, 2.0}`, `--d-k ∈ {256, 512, 1024}`,
  `--fact-only-write-steps ∈ {0, 500}` rather than reasoning about which is best.
- Keep the CPU test suite green (`pytest tests/test_linear_store.py
  tests/test_episodes.py tests/test_trainer_toy.py`) and add a test alongside any
  new math.
- Log everything to `runs/` as JSON/JSONL (the scripts already do); commit run
  artifacts' summaries, not multi-GB caches.

## Branch table on gate failure

| Failed gate | First diagnostic | Branch |
|---|---|---|
| Step 0 (oracle injection) | check `final_logit_softcapping` on the config; retry sweep with raw `W_U[gold]` direction | two-pass soft-token arm (design doc §1/Q2): retrieval-conditioned soft prompt, re-run Step 0 there |
| Step 1 (oracle write/read) | re-run with keys taken from the literal answer position only; check per-fact variance | key-consistency problem: verbatim-prefix keys differ between ingest/query contexts → try post-norm h from the *prompt-final* position only |
| Step 2 (meta-training) | gate stats in the JSONL: read-gate or lr-gate collapsed to ~0? | recall-only warmup (`--lambda-kl 0` for 500 steps), then mix KL back in; raise `--fact-only-write-steps` |
| A2 | `neutral_kl` per λ_KL | raise `--lambda-kl`; if A1 then degrades, gate capacity is the issue → per-token read gate features |
| A3 early saturation | linear-vs-MLP comparison at the same param count | widen d_k before deepening; multi-head store (design doc §1/Q1) |

## Definition of done (this phase)

`runs/acceptance.json` showing A1, A2, A4 = PASS on Gemma-3-1B with the trained
skill, plus an A3 curve for at least two d_k values, plus the oracle JSONs —
committed alongside a short `runs/REPORT.md` summarising numbers against the
design-doc gates. Then, and only then, the RLM-integration question opens.
