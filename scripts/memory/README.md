# Memory-track scripts: the experimental ladder

Implements the ladder from `docs/design_consult_response.md` §4. Run the steps in
order; **each step has a numerical gate — do not proceed past a failed gate, branch
instead** (branch table in `docs/memory_track_execution.md`).

All scripts assume a GPU runtime with `transformers` installed and HF access to the
Gemma checkpoint (`HF_TOKEN`). Approximate wall-clock is for one L4.

| Step | Command | Time | Gate |
|---|---|---|---|
| 0 — oracle injection | `python scripts/memory/oracle_injection.py --out runs/oracle_injection.json` | ~min | ∃λ: mean Δgold ≥ 3 nats, top-5 ≥ 0.9, random-dir flat |
| 1 — oracle write/read | `python scripts/memory/oracle_write_read.py --out runs/oracle_write_read.json` | ~min | ∃scale: lift ≥ 3 nats, top-5 ≥ 0.9, controls flat |
| 2a — build cache | `python scripts/memory/build_cache.py --episodes 20000 --out cache/gemma1b_v1` | ~30–60 min | shards + `head.pt` + `meta.json` written |
| 2b — train skill | `python scripts/memory/train_skill.py --cache cache/gemma1b_v1 --out runs/skill_v1.pt --steps 4000` | ~min–1 h | held-out recall_lift ≥ 3 nats, top5 ≥ 0.8, control flat |
| 2c — live acceptance | `python scripts/memory/eval_acceptance.py --skill runs/skill_v1.pt --out runs/acceptance.json` | ~min | **A1 + A4 PASS** (the pass-or-die moment) |
| 3 — A2 | rerun 2b with `--lambda-kl` sweep if A2 failed at 2c | ~h | A2 PASS without losing A1 |
| 4 — A3 capacity | `eval_acceptance.py --skill ... --a3 1,2,4,8,16,32,64,128` for `--d-k 128,256,512,1024` (retrain per d_k) | ~h–overnight | interference-onset curve per d_k; linear vs welded-MLP comparison |
| 5 — A1b paraphrase | rebuild cache with `--paraphrase-prob 0.5`; retrain with `--encoder-layers 2` | ~h | report verbatim and paraphrase separately |
| 6 — scale base | repeat 0–2c with `--model google/gemma-3-4b-it` | ~h | A1 on the bigger base |

Notes:
- Steps 0–1 need **no training** and de-risk the whole branch in an afternoon. If
  Step 0 fails, residual injection is dead at this splice — switch to the two-pass
  soft-token arm (design doc §1/Q2) before building anything else.
- Sweeps at 2b are cheap (the base is out of the loop). Prefer running grids
  (`--lambda-kl`, `--d-k`, `--fact-only-write-steps`) over deliberating.
- `tests/test_linear_store.py`, `tests/test_episodes.py`, `tests/test_trainer_toy.py`
  run on CPU with no model and must stay green; the toy trainer test is the offline
  replica of the 2b gate (+8.1 nats lift, top-5 = 1.0 in ~6 s).
