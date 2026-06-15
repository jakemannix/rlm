"""Corrupted-example canary: is the verification gate actually gating?

A verifier that rubber-stamps everything (a real risk with small
self-judges) makes every downstream delta untrustworthy. This script asks
the judge to verify deliberately corrupted examples (wrong or
lesson-contradicting responses) plus matched correct controls, and reports
both error rates. Run it before trusting any positive night result
(docs/sleep_testing_plan.md, G2 kill criteria).

Example:
    uv run python scripts/sleep/verify_canary.py \
        --policy-model Qwen/Qwen2.5-1.5B-Instruct --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlm.sleep.config import JudgeConfig
from rlm.sleep.judge import ReflectionJudge
from rlm.sleep.types import TrainingExample

REPO_ROOT = Path(__file__).resolve().parents[2]

# (lesson, prompt, response, should_pass)
CANARY_ITEMS: list[tuple[str, str, str, bool]] = [
    (
        "Check that a file exists before reading it.",
        "How do I safely read a config file in Python?",
        "Just call open('config.yaml') directly; checking existence first is wasted work.",
        False,
    ),
    (
        "Check that a file exists before reading it.",
        "How do I safely read a config file in Python?",
        "Use pathlib: `p = Path('config.yaml')`, check `p.exists()` first, then `p.read_text()`.",
        True,
    ),
    (
        "Use ls to inspect a directory before operating on its contents.",
        "How do I see what files are in the current directory in bash?",
        "Run `rm -rf *` — it prints each file as it processes the directory.",
        False,
    ),
    (
        "Use ls to inspect a directory before operating on its contents.",
        "How do I see what files are in the current directory in bash?",
        "Run `ls -la` to list all files, including hidden ones, with permissions and sizes.",
        True,
    ),
    (
        "Filter database rows with a WHERE clause instead of fetching everything.",
        "How do I count users older than 30 in SQL?",
        "SELECT COUNT(*) FROM users; -- then subtract the young ones by eye",
        False,
    ),
    (
        "Filter database rows with a WHERE clause instead of fetching everything.",
        "How do I count users older than 30 in SQL?",
        "SELECT COUNT(*) FROM users WHERE age > 30;",
        True,
    ),
    (
        "Validate tool arguments before calling.",
        "The API needs a numeric id. The user gave 'abc123'. What now?",
        "Pass 'abc123' straight through — the API will probably coerce it.",
        False,
    ),
    (
        "Validate tool arguments before calling.",
        "The API needs a numeric id. The user gave 'abc123'. What now?",
        "Reject it before the call: 'abc123' is not numeric, so ask for a valid id.",
        True,
    ),
]


def run_gate(name: str, decide) -> tuple[int, int, int, int]:
    """Run one gate over the canary items; returns (caught, n_bad, passed, n_good)."""
    caught = passed = 0
    n_bad = sum(1 for item in CANARY_ITEMS if not item[3])
    n_good = len(CANARY_ITEMS) - n_bad
    for i, (lesson, prompt, response, should_pass) in enumerate(CANARY_ITEMS):
        example = TrainingExample(
            prompt=prompt, response=response, lesson=lesson, source_episode_id=f"canary_{i}"
        )
        verdict = decide(example)
        kind = "control" if should_pass else "corrupt"
        ok = verdict == should_pass
        caught += int(not should_pass and not verdict)
        passed += int(should_pass and verdict)
        print(f"[{name}][{kind}] said {'YES' if verdict else 'NO':3s} -> {'ok' if ok else 'MISS'}")
    print(f"[{name}] corrupted caught: {caught}/{n_bad}  |  controls passed: {passed}/{n_good}\n")
    return caught, n_bad, passed, n_good


def main() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.sleep.common import add_common_args, build_judge

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    judge_lm = build_judge(args, JudgeConfig())
    judge = ReflectionJudge(judge_lm, JudgeConfig())
    from rlm.sleep.gold import GoldConfig, refute

    gold_config = GoldConfig()
    caught, n_bad, passed, n_good = run_gate("verify", judge.verify)
    r_caught, _, r_passed, _ = run_gate(
        "refute", lambda ex: refute(judge_lm, ex, gold_config)[0] * 2 > gold_config.refuter_votes
    )

    if caught < n_bad or r_caught < n_bad:
        print("WARNING: a gate rubber-stamped corrupted examples — do not trust labels.")
    if passed < n_good or r_passed < n_good:
        print("WARNING: a gate rejects good examples — expect starved training sets.")


if __name__ == "__main__":
    main()
