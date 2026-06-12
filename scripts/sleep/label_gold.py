"""Generate gold labels for memory candidates from Claude Code traces.

Loads episodes from a transcript archive (see ``extract_cc_sessions.py``),
prioritizes the ones carrying error/correction signal, and runs the full
offline judging stack (reflect -> verify -> rubric -> refute) from
``rlm.sleep.gold``. Outputs, under ``--out`` (gitignored):

- ``gold_labels.jsonl`` — every candidate with scores, votes, and label
- ``sft_gold.jsonl``    — gold(+silver with --include-silver) examples in
  the same chat format the LoRA trainer consumes
- ``summary.json``      — aggregate stats

Example (offline smoke):
    python scripts/sleep/label_gold.py --mock-judge --limit-episodes 5

Real run (GPU, self-judge):
    python scripts/sleep/label_gold.py --device cuda \
        --policy-model Qwen/Qwen2.5-1.5B-Instruct --limit-episodes 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rlm.sleep.cc_traces import load_cc_episodes
from rlm.sleep.config import JudgeConfig
from rlm.sleep.dataset import write_jsonl
from rlm.sleep.gold import GoldConfig, label_episodes, summarize
from rlm.sleep.types import Episode

REPO_ROOT = Path(__file__).resolve().parents[2]

MOCK_RUBRIC = json.dumps(
    {
        "scores": {
            "correctness": 4,
            "reusability": 4,
            "grounding": 5,
            "specificity": 4,
            "self_contained": 4,
        },
        "fatal_flaw": "",
    }
)


def error_signal(episode: Episode) -> int:
    meta = episode.meta
    return (
        meta.get("n_tool_errors", 0)
        + meta.get("n_soft_errors", 0)
        + 2 * meta.get("n_user_corrections", 0)
    )


def build_mock_judge():
    """Offline judge covering all four prompt families (smoke tests only)."""
    from scripts.sleep.common import MOCK_LEARN_RESPONSE
    from tests.mock_lm import MockLM

    def respond(prompt) -> str:
        text = str(prompt)
        if "scoring one candidate" in text:
            return MOCK_RUBRIC
        if "candidate training example" in text or "last gate" in text:
            return "No real flaw found.\nVERDICT: YES"
        return MOCK_LEARN_RESPONSE

    return MockLM(response_fn=respond)


def main() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="~/Documents/cc-session-archive")
    parser.add_argument("--include-subagents", action="store_true")
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="label everything, not just error/correction-bearing episodes",
    )
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--policy-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--judge-model",
        default="self",
        help='"self" = policy model judges; else an OpenAI-compatible name',
    )
    parser.add_argument("--mock-judge", action="store_true", help="canned judge (offline smoke)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--n-samples", type=int, default=3, help="reflection self-consistency")
    parser.add_argument("--rubric-samples", type=int, default=3)
    parser.add_argument("--refuter-votes", type=int, default=3)
    parser.add_argument(
        "--include-silver",
        action="store_true",
        help="include silver-labeled examples in sft_gold.jsonl",
    )
    parser.add_argument(
        "--allow-external-judge",
        action="store_true",
        help="required to use a non-'self' judge: ships transcript-derived text to an external API",
    )
    parser.add_argument("--out", default="runs/sleep/gold")
    args = parser.parse_args()

    episodes = load_cc_episodes(args.source, include_subagents=args.include_subagents)
    if not args.all_episodes:
        episodes = [ep for ep in episodes if error_signal(ep) > 0]
    episodes.sort(key=error_signal, reverse=True)
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]
    print(f"labeling {len(episodes)} episodes from {args.source}")

    judge_config = JudgeConfig(n_samples=args.n_samples)
    gold_config = GoldConfig(rubric_samples=args.rubric_samples, refuter_votes=args.refuter_votes)
    if args.mock_judge:
        judge_lm = build_mock_judge()
    else:
        if args.judge_model != "self" and not args.allow_external_judge:
            raise SystemExit(
                "Refusing to send personal transcripts to an external judge "
                f"({args.judge_model!r}). Redaction is best-effort, not a guarantee. "
                "Pass --allow-external-judge if you have reviewed the episodes."
            )
        from scripts.sleep.common import build_judge

        judge_lm = build_judge(args, judge_config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = label_episodes(
        episodes,
        judge_lm,
        judge_config=judge_config,
        gold_config=gold_config,
        cache_path=out_dir / "judge_cache.json",
        gold_cache_path=out_dir / "gold_cache.json",
    )

    with (out_dir / "gold_labels.jsonl").open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label.to_dict(), ensure_ascii=False) + "\n")

    keep = {"gold", "silver"} if args.include_silver else {"gold"}
    kept = [label.example for label in labels if label.label in keep]
    write_jsonl(kept, out_dir / "sft_gold.jsonl")

    stats = summarize(labels)
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"{len(kept)} examples -> {out_dir / 'sft_gold.jsonl'}")


if __name__ == "__main__":
    main()
