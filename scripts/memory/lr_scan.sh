#!/usr/bin/env bash
# Validate probe E on REAL text: does lowering the delta-rule lr (gentler writes,
# less erosion) lift the streamed-deployment capacity ceiling (~8-16 facts)?
# Trains a skill (+whiten) at each max_lr on a real SQuAD cache and runs the
# deployment-regime scale eval (eval_real: ingest one fact/session, save/reload,
# worst-case recall vs N). All arms share the cache; lr is the only variable.
set -uo pipefail
cd /content/rlm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/content/rlm
pip install -q datasets >/dev/null 2>&1 || true
say() { echo "=== $1 $(date +%H:%M:%S) ==="; }

say "build real SQuAD cache (held-out by entity, capacity pressure k<=16)"
python scripts/memory/build_cache.py --out cache/real_train --episodes 20000 \
  --real-corpus --squad --real-split train --multifact-k 4,16 --paraphrase-prob 0.5

for MLR in 2.0 0.6 0.2 0.06; do            # init θ = 1.0, 0.3, 0.1, 0.03
  say "train max_lr=$MLR (+whiten)"
  python scripts/memory/train_skill.py --cache cache/real_train --out runs/skill_mlr$MLR.pt \
    --steps 2500 --batch-size 8 --eval-episodes 64 --whiten --max-lr "$MLR" \
    --recall-warmup-steps 300 --kl-warmup-steps 500 --kl-ramp-steps 500
  say "eval_real max_lr=$MLR (held-out, deployment-regime scale)"
  python scripts/memory/eval_real.py --skill runs/skill_mlr$MLR.pt --squad \
    --baseline-facts 48 --n-seeds 2 --scale 8,16,32,64,128 --out runs/real_mlr$MLR.json
done
say "LR SCAN DONE"
