"""Hyperparameter sweep over one nightly cycle.

Examples:
    # Offline smoke test of the sweep machinery:
    uv run python scripts/sleep/run_sweep.py --synthetic --mock-judge --dry-run

    # Real sweep on a Colab GPU:
    OPENAI_API_KEY=... uv run python scripts/sleep/run_sweep.py \
        --dataset THUDM/AgentInstruct --config os \
        --grid '{"adapter.lr": [1e-4, 2e-4, 5e-4], "adapter.rank": [8, 16, 32]}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rlm.sleep.config import SleepConfig
from rlm.sleep.local_judge import resolve_torch_dtype
from rlm.sleep.sweep import run_sweep
from rlm.sleep.traces import partition_days, split_episodes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID = '{"adapter.lr": [1e-4, 2e-4], "adapter.rank": [8, 16]}'


def main() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.sleep.common import (
        add_common_args,
        build_judge,
        fake_eval_fn,
        fake_train_fn,
        load_episodes,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--grid", default=DEFAULT_GRID, help="JSON dict of dotted keys -> values")
    parser.add_argument("--out", default="runs/sleep/sweep")
    args = parser.parse_args()

    episodes = load_episodes(args)
    train, _, test = split_episodes(episodes)
    days = partition_days(train, args.episodes_per_day)
    if len(days) < 2:
        raise ValueError("Need at least 2 days of episodes for next-day eval")

    config = SleepConfig(
        policy_model=args.policy_model,
        device=args.device,
        torch_dtype=resolve_torch_dtype(args.torch_dtype, args.device),
        episodes_per_day=args.episodes_per_day,
    )
    config.gate.use_surprise = args.use_surprise
    kwargs = {"train_fn": fake_train_fn, "eval_fn": fake_eval_fn} if args.dry_run else {}
    csv_path = run_sweep(
        grid=json.loads(args.grid),
        base_config=config,
        day_episodes=days[0],
        next_day_episodes=days[1],
        test_episodes=test,
        judge_lm=build_judge(args, config.judge),
        out_dir=args.out,
        **kwargs,
    )
    print(f"\nsweep results: {csv_path}")


if __name__ == "__main__":
    main()
