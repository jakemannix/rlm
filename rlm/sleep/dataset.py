"""Build the nightly SFT dataset from judge outputs.

Only verified examples from "learn" verdicts are kept.  A configurable
fraction of general replay examples is mixed in to guard against
forgetting, and a leakage guard asserts no source episode belongs to the
held-out splits.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rlm.sleep.types import Episode, JudgeOutput, TrainingExample

# A tiny general-instruction replay pool.  Deliberately bland: its job is
# to anchor general chat behavior during the (small) nightly update, the
# same role replay plays in continual pretraining recipes.
REPLAY_POOL: list[tuple[str, str]] = [
    ("What is the capital of France?", "The capital of France is Paris."),
    (
        "Explain the difference between a list and a tuple in Python.",
        "A list is mutable — you can add, remove, or change elements. A tuple is "
        "immutable: once created, its contents cannot change. Tuples are hashable "
        "when their elements are, so they can serve as dictionary keys.",
    ),
    (
        "Write a one-line shell command to count the lines in every .txt file "
        "in the current directory.",
        "wc -l *.txt",
    ),
    (
        "Summarize in one sentence why unit tests are useful.",
        "Unit tests catch regressions early by verifying each component's expected "
        "behavior automatically.",
    ),
    (
        "What does HTTP status code 404 mean?",
        "404 means Not Found: the server could not locate the requested resource.",
    ),
    (
        "Convert 2.5 hours to minutes.",
        "2.5 hours is 150 minutes.",
    ),
]


def collect_examples(outputs: list[JudgeOutput]) -> list[TrainingExample]:
    """Keep only verified examples from confident "learn" verdicts."""
    return [ex for out in outputs if out.verdict == "learn" for ex in out.examples if ex.verified]


def assert_no_leakage(examples: list[TrainingExample], held_out: list[Episode]) -> None:
    """Fail loudly if any training example came from a held-out episode."""
    held_out_ids = {ep.episode_id for ep in held_out}
    leaked = sorted({ex.source_episode_id for ex in examples} & held_out_ids)
    if leaked:
        raise ValueError(f"SFT examples leak from held-out episodes: {leaked}")


def with_replay(
    examples: list[TrainingExample],
    replay_ratio: float,
    seed: int = 0,
) -> list[TrainingExample]:
    """Mix general replay examples into the nightly batch."""
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("replay_ratio must lie in [0, 1)")
    if not examples or replay_ratio == 0.0:
        return list(examples)
    n_replay = max(1, round(len(examples) * replay_ratio / (1.0 - replay_ratio)))
    rng = random.Random(seed)
    replay = [
        TrainingExample(
            prompt=p,
            response=r,
            lesson="replay",
            source_episode_id="replay",
            verified=True,
        )
        for p, r in (rng.choice(REPLAY_POOL) for _ in range(n_replay))
    ]
    mixed = list(examples) + replay
    rng.shuffle(mixed)
    return mixed


def write_jsonl(
    examples: list[TrainingExample],
    path: str | Path,
    system_prompt: str | None = None,
) -> Path:
    """Persist examples as chat-format JSONL ({"messages": [...]})."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            record = {
                "messages": ex.to_messages(system_prompt=system_prompt),
                "lesson": ex.lesson,
                "source_episode_id": ex.source_episode_id,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[TrainingExample]:
    """Load examples back from chat-format JSONL."""
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        messages = record["messages"]
        user = next(m for m in messages if m["role"] == "user")
        assistant = next(m for m in messages if m["role"] == "assistant")
        examples.append(
            TrainingExample(
                prompt=user["content"],
                response=assistant["content"],
                lesson=record.get("lesson", ""),
                source_episode_id=record.get("source_episode_id", "unknown"),
                verified=True,
            )
        )
    return examples
