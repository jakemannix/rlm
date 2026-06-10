"""Step 2b — meta-train the memory skill from the cached activations.

The base never loads: each step is GEMMs over cached tensors, so a 3k-step run
on an L4 is minutes-to-tens-of-minutes.  Sweeps are cheap — prefer running the
full grid over agonising about one configuration.

    python scripts/memory/train_skill.py --cache cache/gemma1b_v1 \
        --out runs/skill_v1.pt --steps 4000 --d-k 512

Gate (design doc §4, Step 2 — evaluated live by eval_acceptance.py, but the
trainer's held-out metrics should already show it): recall_lift ≥ 3 nats,
top5 ≥ 0.8, control probe |lift| ≤ 0.5.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from rlm.memory.linear_store import LinearStoreConfig
from rlm.memory.skill import MemorySkill, SkillConfig
from rlm.memory.trainer import EpisodeTrainer, TrainerConfig, collate


def materialize(ep: dict, W: torch.Tensor) -> dict:
    """Cache stores next-token *ids*; the trainer wants embedding rows."""
    return {
        "sessions": [
            {"hn": s["hn"], "e_next": W[s["e_next_ids"]], "fact_mask": s["fact_mask"]}
            for s in ep["sessions"]
        ],
        "query": ep["query"],
    }


class ShardPool:
    """Keep up to ``ram_shards`` shards in RAM, rotating through the rest."""

    def __init__(self, cache_dir: Path, ram_shards: int, eval_episodes: int, rng: random.Random):
        self.paths = sorted(cache_dir.glob("shard_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no shards in {cache_dir}")
        self.rng = rng
        first = torch.load(self.paths[0], weights_only=False)
        self.eval_episodes = first[:eval_episodes]
        self.pool: list[list[dict]] = [first[eval_episodes:]]
        for p in self.paths[1:ram_shards]:
            self.pool.append(torch.load(p, weights_only=False))
        self.next_path = ram_shards
        print(f"pool: {sum(len(s) for s in self.pool)} episodes in RAM "
              f"({len(self.pool)}/{len(self.paths)} shards), {len(self.eval_episodes)} held out")

    def rotate(self) -> None:
        if len(self.paths) <= len(self.pool):
            return
        slot = self.rng.randrange(1, len(self.pool)) if len(self.pool) > 1 else 0
        self.pool[slot] = torch.load(self.paths[self.next_path], weights_only=False)
        self.next_path = (self.next_path + 1) % len(self.paths) or 1  # skip shard 0 (eval lives there)

    def sample(self, n: int) -> list[dict]:
        shard = self.rng.choice(self.pool)
        return [self.rng.choice(shard) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=0.5)
    ap.add_argument("--d-k", type=int, default=512)
    ap.add_argument("--encoder-layers", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--fact-only-write-steps", type=int, default=0,
                    help="curriculum: write only fact spans for the first N steps")
    ap.add_argument("--ram-shards", type=int, default=8)
    ap.add_argument("--rotate-every", type=int, default=200)
    ap.add_argument("--eval-episodes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cache = Path(args.cache)
    head = torch.load(cache / "head.pt", weights_only=False)
    W = head["W"].to(torch.float32)
    print(f"head: d={head['d_model']} vocab={W.shape[0]} softcap={head['softcap']} from {head['model_name']}")

    rng = random.Random(args.seed)
    pool = ShardPool(cache, args.ram_shards, args.eval_episodes, rng)

    skill = MemorySkill(SkillConfig(
        d_model=head["d_model"], d_k=args.d_k, encoder_layers=args.encoder_layers,
        store=LinearStoreConfig(chunk_size=args.chunk_size),
    ))
    cfg = TrainerConfig(
        lr=args.lr, steps=args.steps, batch_size=args.batch_size, lambda_kl=args.lambda_kl,
        fact_only_write_steps=args.fact_only_write_steps, device=args.device, seed=args.seed,
    )
    trainer = EpisodeTrainer(skill, W, cfg, logit_softcap=head["softcap"])

    step = {"n": 0}

    def sample_batch() -> dict:
        step["n"] += 1
        if step["n"] % args.rotate_every == 0:
            pool.rotate()
        eps = [materialize(e, W) for e in pool.sample(args.batch_size)]
        return collate(eps, args.device)

    eval_batch = collate([materialize(e, W) for e in pool.eval_episodes], args.device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.train(sample_batch, eval_batch=eval_batch, log_path=str(out.with_suffix(".log.jsonl")))

    final = trainer.evaluate(eval_batch)
    skill.save(str(out))
    out.with_suffix(".eval.json").write_text(json.dumps(final, indent=2))
    print(f"\nfinal held-out eval: {json.dumps(final, indent=2)}")
    print(f"saved skill → {out}")


if __name__ == "__main__":
    main()
