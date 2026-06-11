"""Tests for trace normalization, splits, and day partitioning."""

from rlm.sleep.traces import normalize_record, partition_days, split_episodes, synthetic_traces


def test_synthetic_traces_deterministic():
    a = synthetic_traces(20, seed=7)
    b = synthetic_traces(20, seed=7)
    assert [ep.episode_id for ep in a] == [ep.episode_id for ep in b]
    assert [ep.steps[-1].content for ep in a] == [ep.steps[-1].content for ep in b]


def test_synthetic_traces_plant_learnworthy_episodes():
    episodes = synthetic_traces(100, seed=0)
    planted = [ep for ep in episodes if ep.meta["planted_learnworthy"]]
    assert 10 < len(planted) < 60
    # Planted episodes carry the signals the gate looks for.
    assert any("Error" in s.content for ep in planted for s in ep.steps)


def test_normalize_agentinstruct_format():
    record = {
        "id": "os_42",
        "conversations": [
            {"from": "human", "value": "List the files in /tmp"},
            {"from": "gpt", "value": "I'll run ls /tmp"},
            {"from": "observation", "value": "a.txt b.txt"},
            {"from": "gpt", "value": "The files are a.txt and b.txt."},
        ],
    }
    ep = normalize_record(record, source="THUDM/AgentInstruct", index=0)
    assert ep.episode_id == "os_42"
    assert [s.role for s in ep.steps] == ["user", "assistant", "tool", "assistant"]
    assert ep.final_response() == "The files are a.txt and b.txt."
    assert ep.task.startswith("List the files")


def test_normalize_messages_format():
    record = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    ep = normalize_record(record, source="x", index=3)
    assert ep.episode_id == "x-3"
    assert ep.final_response() == "hello"


def test_split_is_deterministic_and_disjoint():
    episodes = synthetic_traces(300, seed=1)
    train1, val1, test1 = split_episodes(episodes)
    train2, val2, test2 = split_episodes(list(reversed(episodes)))
    ids = lambda eps: {ep.episode_id for ep in eps}  # noqa: E731
    assert ids(train1) == ids(train2)
    assert ids(val1) == ids(val2)
    assert ids(test1) == ids(test2)
    assert not (ids(train1) & ids(val1) & ids(test1))
    assert len(train1) + len(val1) + len(test1) == 300
    # Roughly the requested proportions.
    assert 0.7 < len(train1) / 300 < 0.9


def test_partition_days():
    episodes = synthetic_traces(25, seed=2)
    days = partition_days(episodes, episodes_per_day=10)
    assert [len(d) for d in days] == [10, 10, 5]
    assert days[0][0].episode_id == episodes[0].episode_id
