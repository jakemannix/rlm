#!/usr/bin/env bash
# Fable capacity plan C1+C2 (docs/design_review_capacity.md). C0 already predicted
# whitening takes A3 from ~4 facts to ~128 on the real cache geometry (PR=10, mean
# cos 0.77). This runs the decisive experiments back-to-back (GPU busy = reclaim-safe):
#   C1  base       : d_k=512, no whitening (the A1+A4 baseline, reproduces the collapse)
#   C1  whiten     : + ZCA whitening on cached addressing vectors (the predicted fix)
#   C2  pressure_whiten : data-pressure cache (multifact k≤32, same-relation hard
#                         negatives, paraphrase) + whiten + 2-layer encoder
# A3 reported with 24 facts + 24 a2-facts (denoised) per the stats protocol.
set -uo pipefail
cd /content/rlm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/content/rlm
STEPS=${STEPS:-3000}
say() { echo "=== $1 $(date +%H:%M:%S) ==="; }

train_eval() {
  name=$1; cache=$2; shift 2
  say "train:$name"
  python scripts/memory/train_skill.py --cache "$cache" --out runs/skill_$name.pt \
    --steps "$STEPS" --batch-size 8 --eval-episodes 64 --recall-warmup-steps 300 \
    --kl-warmup-steps 500 --kl-ramp-steps 500 --checkpoint-every 500 "$@"
  say "eval:$name"
  python scripts/memory/eval_acceptance.py --skill runs/skill_$name.pt \
    --facts 24 --a2-facts 24 --a3 1,2,4,8,16,32,64,128 --out runs/acc_$name.json
}

# C1 — decisive whitening A/B on the existing cache
train_eval base   cache/gemma1b_v1
train_eval whiten cache/gemma1b_v1 --whiten

# C2 — data-pressure cache (so the encoder is actually asked to separate >6 facts)
say "build_cache:pressure"
python scripts/memory/build_cache.py --episodes 20000 --out cache/gemma1b_pressure \
  --multifact-k 2,32 --same-relation-control-prob 0.5 --paraphrase-prob 0.25
say "diagnose:pressure(labelled)"
python scripts/memory/diagnose_keys.py --cache cache/gemma1b_pressure --out runs/key_diag_pressure.json || true
train_eval pressure_whiten cache/gemma1b_pressure --whiten --encoder-layers 2
say "WHITEN SWEEP DONE"
