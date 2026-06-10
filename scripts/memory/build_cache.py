"""Step 2a — build the cached-activation episode dataset (design doc §3).

Runs the frozen base once over synthetic episodes and stores post-norm hiddens
+ next-token ids; after this the base is out of the training loop entirely.
~20k episodes ≈ 4M tokens of Gemma-3-1B forwards ≈ tens of minutes on an L4,
~10 GB on disk at fp16.

    python scripts/memory/build_cache.py --model google/gemma-3-1b-it \
        --episodes 20000 --out cache/gemma1b_v1
"""

from __future__ import annotations

import argparse
import json

import torch

from rlm.memory.cache import build_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--paraphrase-prob",
        type=float,
        default=0.0,
        help="0.0 for the A1-verbatim ladder; raise for the A1b arm",
    )
    ap.add_argument(
        "--mix", default=None, help='JSON, e.g. \'{"recall":0.6,"abstain":0.2,"control":0.2}\''
    )
    ap.add_argument("--multifact-k", default="2,6",
                    help="comma lo,hi for #facts per multifact episode — capacity pressure (design review C2)")
    ap.add_argument("--same-relation-control-prob", type=float, default=0.0,
                    help="fraction of control episodes that are same-relation hard negatives (C2)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=1_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild even if a (mismatched) cache exists at --out; deletes its stale artifacts",
    )
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        .to(args.device)
        .eval()
    )
    from rlm.memory.episodes import EpisodeGenerator

    lo, hi = (int(v) for v in args.multifact_k.split(","))
    gen = EpisodeGenerator(
        seed=args.seed,
        paraphrase_prob=args.paraphrase_prob,
        multifact_k=(lo, hi),
        same_relation_control_prob=args.same_relation_control_prob,
    )
    build_cache(
        model,
        tok,
        args.out,
        n_episodes=args.episodes,
        mix=json.loads(args.mix) if args.mix else None,
        generator=gen,
        seed=args.seed,
        paraphrase_prob=args.paraphrase_prob,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
