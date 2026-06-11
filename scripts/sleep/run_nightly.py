"""Run one (or several) nightly consolidation cycles.

Examples:
    # Offline smoke test (synthetic traces, mock judge, no GPU needed):
    uv run python scripts/sleep/run_nightly.py --synthetic --mock-judge --dry-run

    # Real run on AgentInstruct/os with an OpenAI judge (GPU):
    OPENAI_API_KEY=... uv run python scripts/sleep/run_nightly.py \
        --dataset THUDM/AgentInstruct --config os \
        --policy-model Qwen/Qwen2.5-1.5B-Instruct \
        --judge-model gpt-4o --days 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlm.sleep.config import SleepConfig
from rlm.sleep.local_judge import resolve_torch_dtype
from rlm.sleep.loop import run_night
from rlm.sleep.traces import partition_days, split_episodes

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument("--days", type=int, default=1, help="number of nightly cycles to run")
    parser.add_argument("--out", default="runs/sleep/nightly")
    args = parser.parse_args()

    episodes = load_episodes(args)
    train, _, test = split_episodes(episodes)
    days = partition_days(train, args.episodes_per_day)
    if len(days) < args.days + 1:
        raise ValueError(
            f"Need at least {args.days + 1} days of episodes, got {len(days)} "
            f"(use a smaller --episodes-per-day or more data)"
        )
    print(f"{len(episodes)} episodes -> {len(train)} train / {len(test)} test, {len(days)} days")

    config = SleepConfig(
        policy_model=args.policy_model,
        device=args.device,
        torch_dtype=resolve_torch_dtype(args.torch_dtype, args.device),
        episodes_per_day=args.episodes_per_day,
    )
    judge_lm = build_judge(args, config.judge)
    kwargs = {"train_fn": fake_train_fn, "eval_fn": fake_eval_fn} if args.dry_run else {}

    for day in range(args.days):
        result = run_night(
            day_index=day,
            day_episodes=days[day],
            next_day_episodes=days[day + 1],
            test_episodes=test,
            config=config,
            judge_lm=judge_lm,
            out_dir=Path(args.out) / f"day_{day:03d}",
            **kwargs,
        )
        print(
            f"[day {day}] episodes={result.n_episodes} gated={result.n_selected} "
            f"learn={result.n_learn_verdicts} examples={result.n_examples}"
        )
        if result.adapted_report is not None:
            base, adapted = result.base_report, result.adapted_report
            print(
                f"        next-day NLL {base.next_day_nll:.4f} -> {adapted.next_day_nll:.4f} | "
                f"test NLL {base.test_nll:.4f} -> {adapted.test_nll:.4f} | "
                f"retention NLL {base.retention_nll:.4f} -> {adapted.retention_nll:.4f}"
            )


if __name__ == "__main__":
    main()
