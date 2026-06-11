"""Tests for SFT dataset assembly: filtering, replay, leakage guard, round-trip."""

import pytest

from rlm.sleep.dataset import (
    assert_no_leakage,
    collect_examples,
    read_jsonl,
    with_replay,
    write_jsonl,
)
from rlm.sleep.traces import synthetic_traces
from rlm.sleep.types import JudgeOutput, TrainingExample


def example(ep_id: str = "syn-0-0", verified: bool = True) -> TrainingExample:
    return TrainingExample(
        prompt="p",
        response="r",
        lesson="l",
        source_episode_id=ep_id,
        verified=verified,
    )


def test_collect_keeps_only_verified_learn_examples():
    outputs = [
        JudgeOutput(
            episode_id="a",
            verdict="learn",
            confidence=0.9,
            lesson="l",
            examples=[example("a", verified=True), example("a", verified=False)],
        ),
        JudgeOutput(
            episode_id="b",
            verdict="skip",
            confidence=0.9,
            lesson="",
            examples=[example("b", verified=True)],
        ),
    ]
    kept = collect_examples(outputs)
    assert len(kept) == 1
    assert kept[0].source_episode_id == "a"


def test_leakage_guard_raises():
    held_out = synthetic_traces(5, seed=6)
    leaky = [example(ep_id=held_out[0].episode_id)]
    with pytest.raises(ValueError, match="leak"):
        assert_no_leakage(leaky, held_out)
    assert_no_leakage([example(ep_id="not-held-out")], held_out)


def test_with_replay_ratio():
    examples = [example() for _ in range(8)]
    mixed = with_replay(examples, replay_ratio=0.2, seed=0)
    n_replay = sum(1 for ex in mixed if ex.source_episode_id == "replay")
    assert n_replay == 2  # 8 * 0.2 / 0.8 = 2
    assert len(mixed) == 10
    # Deterministic under the same seed.
    again = with_replay(examples, replay_ratio=0.2, seed=0)
    assert [ex.prompt for ex in again] == [ex.prompt for ex in mixed]


def test_with_replay_zero_is_identity():
    examples = [example() for _ in range(3)]
    assert with_replay(examples, replay_ratio=0.0) == examples


def test_jsonl_round_trip(tmp_path):
    examples = [
        TrainingExample(
            prompt="What is 2+2?",
            response="4",
            lesson="arithmetic",
            source_episode_id="e1",
            verified=True,
        )
    ]
    path = write_jsonl(examples, tmp_path / "sft.jsonl", system_prompt="be brief")
    loaded = read_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].prompt == "What is 2+2?"
    assert loaded[0].response == "4"
    assert loaded[0].lesson == "arithmetic"
