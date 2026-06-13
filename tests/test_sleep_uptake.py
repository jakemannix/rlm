"""Tests for the uptake-experiment pure helpers (no GPU, no API)."""

import argparse
import json

from scripts.sleep.run_uptake_experiment import build_train_examples, compute_split


def ns(**kw) -> argparse.Namespace:
    base = dict(split="random", heldout_fraction=0.3, split_seed=0, cutoff="2025-06-01")
    base.update(kw)
    return argparse.Namespace(**base)


def prep():
    return [
        {
            "memory_id": "fm_000",
            "memory_type": "workflow_rule",
            "tier": "gold",
            "evidence": [{"created_at": "2024-07-01"}],
        },
        {
            "memory_id": "fm_001",
            "memory_type": "workflow_rule",
            "tier": "gold",
            "evidence": [{"created_at": "2026-01-15"}],
        },
        {
            "memory_id": "fm_002",
            "memory_type": "technical_principle",
            "tier": "silver",
            "evidence": [{"created_at": "2025-03-01"}],
        },
        {
            "memory_id": "fm_003",
            "memory_type": "technical_principle",
            "tier": "silver",
            "evidence": [{"created_at": "2026-05-01"}],
        },
    ]


def test_random_split_partitions():
    s = compute_split(prep(), ns(split="random"))
    assert set(s["train"]).isdisjoint(s["heldout"])
    assert set(s["train"]) | set(s["heldout"]) == {"fm_000", "fm_001", "fm_002", "fm_003"}


def test_temporal_split_by_cutoff():
    s = compute_split(prep(), ns(split="temporal", cutoff="2025-06-01"))
    assert set(s["train"]) == {"fm_000", "fm_002"}  # dated <= cutoff
    assert set(s["heldout"]) == {"fm_001", "fm_003"}  # dated after


def test_build_train_examples_filters_to_train_ids(tmp_path):
    days = tmp_path / "days"
    days.mkdir()
    (days / "2024-07-01.jsonl").write_text(
        json.dumps(
            {
                "memory_id": "fm_000",
                "memory": "lesson A",
                "messages": [
                    {"role": "user", "content": "ctx"},
                    {"role": "assistant", "content": "mem A"},
                ],
            }
        )
        + "\n"
        + json.dumps(
            {
                "memory_id": "fm_001",
                "memory": "lesson B",
                "messages": [
                    {"role": "user", "content": "ctx"},
                    {"role": "assistant", "content": "mem B"},
                ],
            }
        )
    )
    examples = build_train_examples(tmp_path, {"fm_000"})
    assert len(examples) == 1
    assert examples[0].source_episode_id == "fm_000"
    assert examples[0].response == "mem A"
