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
import json
import sys
from pathlib import Path

from rlm.sleep.config import SleepConfig
from rlm.sleep.dataset import collect_examples
from rlm.sleep.judge import output_from_dict
from rlm.sleep.local_judge import resolve_torch_dtype
from rlm.sleep.loop import run_night
from rlm.sleep.traces import partition_days, split_episodes

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.sleep.common import (
        add_common_args,
        apply_judge_overrides,
        build_judge,
        dump_judge_usage,
        fake_eval_fn,
        fake_train_fn,
        load_episodes,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--days", type=int, default=1, help="number of nightly cycles to run")
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="re-distill baseline: night N trains from base on the union of "
        "all nights' verified examples so far (vs. independent nightly adapters)",
    )
    parser.add_argument(
        "--win-rate",
        type=int,
        default=0,
        metavar="N",
        help="after the last night, blind pairwise win-rate (adapted vs base) "
        "on N next-day prompts (0 = off; ignored under --dry-run)",
    )
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
    config.gate.use_surprise = args.use_surprise
    apply_judge_overrides(config, args)
    judge_lm = build_judge(args, config.judge)
    kwargs = {"train_fn": fake_train_fn, "eval_fn": fake_eval_fn} if args.dry_run else {}

    accumulated: list = []
    last_result = None
    for day in range(args.days):
        day_dir = Path(args.out) / f"day_{day:03d}"
        result = run_night(
            day_index=day,
            day_episodes=days[day],
            next_day_episodes=days[day + 1],
            test_episodes=test,
            config=config,
            judge_lm=judge_lm,
            out_dir=day_dir,
            judge_cache_path=Path(args.out) / "judge_cache.json",
            extra_examples=list(accumulated) if args.cumulative else None,
            **kwargs,
        )
        if args.cumulative:
            # Tonight's *new* verified examples (pre-replay, pre-extras) come
            # from the persisted judge outputs, never from sft_data.jsonl —
            # the latter would double-count prior nights.
            outputs = json.loads((day_dir / "judge_outputs.json").read_text())
            accumulated.extend(collect_examples([output_from_dict(o) for o in outputs]))
        last_result = result
        print(
            f"[day {day}] episodes={result.n_episodes} gated={result.n_selected} "
            f"learn={result.n_learn_verdicts} examples={result.n_examples}"
        )
        if result.stage_seconds:
            print(
                "        stage seconds: "
                + " | ".join(f"{k} {v:.1f}" for k, v in result.stage_seconds.items())
            )
        if result.adapted_report is not None:
            base, adapted = result.base_report, result.adapted_report
            print(
                f"        next-day NLL {base.next_day_nll:.4f} -> {adapted.next_day_nll:.4f} | "
                f"test NLL {base.test_nll:.4f} -> {adapted.test_nll:.4f} | "
                f"retention NLL {base.retention_nll:.4f} -> {adapted.retention_nll:.4f}"
            )

    if args.win_rate and not args.dry_run:
        if last_result is None or last_result.adapter_dir is None:
            print("win-rate: skipped (no adapter trained on the final night)")
        else:
            from rlm.sleep.lora import load_adapter, load_policy
            from rlm.sleep.winrate import generate_pairs, judged_win_rate

            prompts = days[args.days][: args.win_rate]
            base_model, tokenizer = load_policy(
                config.policy_model, device=config.device, torch_dtype=config.torch_dtype
            )
            base_model.eval()
            adapted_model, _ = load_adapter(
                config.policy_model,
                last_result.adapter_dir,
                device=config.device,
                torch_dtype=config.torch_dtype,
            )
            pairs = generate_pairs(base_model, adapted_model, tokenizer, prompts, config.device)
            report = judged_win_rate(judge_lm, pairs, seed=config.adapter.seed)
            (Path(args.out) / "win_rate.json").write_text(json.dumps(report.to_dict(), indent=2))
            print(
                f"win-rate (adapted vs base, n={report.n}): {report.win_rate:.3f} "
                f"[{report.wins}W/{report.losses}L/{report.ties}T]"
            )

    usage = dump_judge_usage(judge_lm, Path(args.out) / "judge_usage.json")
    for model_name, model_usage in usage.items():
        print(
            f"judge usage [{model_name}]: {model_usage['total_calls']} calls, "
            f"{model_usage['total_input_tokens']} in / {model_usage['total_output_tokens']} out tokens"
        )


if __name__ == "__main__":
    main()
