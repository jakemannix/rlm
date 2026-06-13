"""Tests for the train/heldout memory split."""

from rlm.sleep.frontier_memories import split_memories

MEMS = [
    {"memory_id": f"fm_{i:03d}", "memory_type": t, "tier": tier}
    for i, (t, tier) in enumerate(
        [("workflow_rule", "gold")] * 10
        + [("technical_principle", "gold")] * 10
        + [("style_preference", "silver")] * 4
    )
]


def test_split_is_deterministic():
    a = split_memories(MEMS, heldout_fraction=0.3, seed=0)
    b = split_memories(MEMS, heldout_fraction=0.3, seed=0)
    assert a == b


def test_split_partitions_without_overlap():
    s = split_memories(MEMS, heldout_fraction=0.3, seed=0)
    assert set(s["train"]).isdisjoint(s["heldout"])
    assert set(s["train"]) | set(s["heldout"]) == {m["memory_id"] for m in MEMS}


def test_split_is_stratified():
    s = split_memories(MEMS, heldout_fraction=0.3, seed=0)
    heldout = set(s["heldout"])
    by_type = {"workflow_rule": 0, "technical_principle": 0, "style_preference": 0}
    for m in MEMS:
        if m["memory_id"] in heldout:
            by_type[m["memory_type"]] += 1
    # 30% of each stratum: 3 / 3 / 1
    assert by_type == {"workflow_rule": 3, "technical_principle": 3, "style_preference": 1}


def test_seed_changes_membership_not_proportions():
    s0 = split_memories(MEMS, heldout_fraction=0.3, seed=0)
    s1 = split_memories(MEMS, heldout_fraction=0.3, seed=1)
    assert len(s0["heldout"]) == len(s1["heldout"])
    assert s0["heldout"] != s1["heldout"]
