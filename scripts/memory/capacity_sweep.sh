#!/usr/bin/env bash
# Capacity sweep: build cache once, then train+eval four arms back-to-back, so the
# GPU stays busy (idle-reclaim safe) and results land as per-arm JSON. Designed to be
# launched DETACHED on the Colab L4:
#   PYTHONPATH=/content/rlm nohup bash scripts/memory/capacity_sweep.sh >runs/sweep.log 2>&1 &
#
# The science (see docs/handoff_capacity_and_multihead.md): the oracle probe showed the
# ceiling is KEY OVERLAP, not the store, so these arms test decorrelation levers —
#   base      : d_k=512, single head, linear encoder (the A1+A4-PASS baseline)
#   mh8       : 8 INDEPENDENT per-head encoders (the multi-head capacity bet)
#   mh8shared : 8 heads from ONE reshaped encoder (predicted ~no-op A/B control)
#   enc2      : single head, 2-layer GELU encoder (cheapest decorrelation lever)
set -uo pipefail
cd /content/rlm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/content/rlm
STEPS=${STEPS:-3000}
B=${BATCH:-8}
say() { echo "=== $1 $(date +%H:%M:%S) ==="; }

say "build_cache"
python scripts/memory/build_cache.py --episodes 20000 --out cache/gemma1b_v1

say "geometry(diagnose Part B)"
python scripts/memory/diagnose.py --facts 128 | tee runs/diag_geometry.txt

run_arm () {
  name=$1; shift
  say "train:$name"
  python scripts/memory/train_skill.py --cache cache/gemma1b_v1 --out runs/skill_$name.pt \
    --steps "$STEPS" --batch-size "$B" --eval-episodes 64 --recall-warmup-steps 300 \
    --kl-warmup-steps 500 --kl-ramp-steps 500 --checkpoint-every 500 "$@"
  say "eval:$name"
  python scripts/memory/eval_acceptance.py --skill runs/skill_$name.pt \
    --facts 24 --a2-facts 24 --a3 1,2,4,8,16,32,64,128 --out runs/acc_$name.json
}

run_arm base      --d-k 512
run_arm mh8       --d-k 512 --n-heads 8
run_arm mh8shared --d-k 512 --n-heads 8 --shared-encoder
run_arm enc2      --d-k 512 --encoder-layers 2
say "SWEEP DONE"
