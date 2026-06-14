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

## Modal GPU runs

`scripts/memory/modal_gpu.py` runs the same ladder on Modal with persistent
Volumes for cache/run artifacts and the Hugging Face model cache. It is meant
to replace Colab cells when a run should be headless or needs a larger GPU than
an L4.

Local setup:

```bash
uv pip install -e ".[modal]"
modal setup

# Recommended for gated Gemma checkpoints:
modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
export RLM_MODAL_HF_SECRET=huggingface-secret
```

If you do not set `RLM_MODAL_HF_SECRET`, the runner copies local
`HF_TOKEN`, `HF_HUB_TOKEN`, or `HF_API_KEY` into the remote container; `HF_API_KEY`
is aliased to the Hugging Face variable names expected by `transformers`.

Defaults:

| Setting | Default | Override |
|---|---|---|
| Modal app | `rlm-memory-gpu` | `RLM_MODAL_APP` |
| GPU | `L4` | `RLM_MODAL_GPU=H100`, `A100-80GB`, `H100:8`, or `H100,A100-80GB,L40S` |
| Artifacts volume | `rlm-memory-artifacts` mounted at `/vol/rlm` | `RLM_MODAL_ARTIFACT_VOLUME` |
| HF cache volume | `rlm-hf-cache` mounted at `/root/.cache/huggingface` | `RLM_MODAL_HF_CACHE_VOLUME` |
| Timeout | 24 h | `RLM_MODAL_TIMEOUT` seconds |

Smoke-check CUDA:

```bash
modal run scripts/memory/modal_gpu.py --job gpu-report
```

Smoke-check Hugging Face auth and metadata access for `--model` without
printing the token:

```bash
modal run scripts/memory/modal_gpu.py --job hf-report
```

Run the full Step 0 → 2c ladder on one H100:

```bash
RLM_MODAL_GPU=H100 modal run scripts/memory/modal_gpu.py \
  --job acceptance-ladder \
  --episodes 20000 \
  --steps 4000 \
  --d-k 512 \
  --lambda-kl 0.5
```

The combined ladder commits outputs after every step and stops immediately if
Step 0, Step 1, A1, or A4 fails its JSON gate.

Run steps separately:

```bash
modal run scripts/memory/modal_gpu.py --job build-cache --episodes 20000
modal run scripts/memory/modal_gpu.py --job train-skill --steps 4000 --d-k 512
modal run scripts/memory/modal_gpu.py --job eval-acceptance --a3 1,2,4,8,16,32,64,128
```

Run a custom command from the repo root inside the same image/volumes:

```bash
modal run scripts/memory/modal_gpu.py --job command \
  --command "python scripts/memory/train_skill.py --cache /vol/rlm/cache/gemma1b_v1 --out /vol/rlm/runs/skill_lr01.pt --lambda-kl 0.1"
```

Fetch artifacts:

```bash
modal volume ls rlm-memory-artifacts /runs
modal volume get rlm-memory-artifacts /runs/acceptance.json runs/modal_acceptance.json
```

Requesting `H100:8` or another multi-GPU shape reserves that hardware on one
Modal container, but the current memory scripts are still single-process and
use one CUDA device. Use multi-GPU reservations only for jobs whose command
actually launches distributed training (`torchrun`, `accelerate launch`, etc.).
