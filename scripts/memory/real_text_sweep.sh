#!/usr/bin/env bash
# Milestone-3 item #1 (the load-bearing real-world test): does a skill meta-trained
# on REAL text recall HELD-OUT real facts (unseen entities) across a session
# boundary — both verbatim and paraphrased (content-addressable / A1b)?
#
# Held-out protocol: RealCorpusGenerator splits the SQuAD record pool by ENTITY
# (deterministic, seed-independent), so train and eval entities never overlap;
# build_cache uses --real-split train, eval uses --real-split eval on the same pool.
# Whitening is on (C0/C1 showed it is the capacity lever); multifact pressure k≤16.
set -uo pipefail
cd /content/rlm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/content/rlm
pip install -q datasets >/dev/null 2>&1 || true
say() { echo "=== $1 $(date +%H:%M:%S) ==="; }

say "build real train cache (SQuAD train-entities)"
python scripts/memory/build_cache.py --out cache/real_train --episodes 20000 \
  --real-corpus --real-split train --squad --paraphrase-prob 0.5 --multifact-k 2,16

say "diagnose real-train key geometry (labelled)"
python scripts/memory/diagnose_keys.py --cache cache/real_train --out runs/key_diag_real.json || true

say "train skill on real text (+whiten)"
python scripts/memory/train_skill.py --cache cache/real_train --out runs/skill_real.pt \
  --steps 3000 --batch-size 8 --eval-episodes 64 --whiten \
  --recall-warmup-steps 300 --kl-warmup-steps 500 --kl-ramp-steps 500 --checkpoint-every 500

say "eval HELD-OUT real entities — VERBATIM query"
python scripts/memory/eval_acceptance.py --skill runs/skill_real.pt \
  --real-corpus --real-split eval --squad --facts 64 --a2-facts 32 \
  --a3 1,2,4,8,16,32 --out runs/acc_real_verbatim.json

say "eval HELD-OUT real entities — PARAPHRASE query (A1b / content-addressable)"
python scripts/memory/eval_acceptance.py --skill runs/skill_real.pt \
  --real-corpus --real-split eval --squad --paraphrase --facts 64 --a2-facts 32 \
  --a3 1,2,4,8,16,32 --out runs/acc_real_paraphrase.json
say "MILESTONE-3 #1 DONE"
