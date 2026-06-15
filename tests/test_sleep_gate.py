"""Tests for the cheap gate: scoring, budget, and signal quality."""

from rlm.sleep.config import GateConfig
from rlm.sleep.gate import HeuristicGate, heuristic_signals, select_for_reflection
from rlm.sleep.traces import synthetic_traces
from rlm.sleep.types import Episode, Step


def make_episode(steps: list[Step], outcome: str = "unknown") -> Episode:
    return Episode(episode_id="e1", source="test", task="t", steps=steps, outcome=outcome)


def test_clean_episode_scores_low():
    ep = make_episode(
        [
            Step("user", "TASK: do x"),
            Step("assistant", "doing x"),
            Step("tool", "ok"),
            Step("assistant", "done"),
        ],
        outcome="success",
    )
    score, signals = HeuristicGate(GateConfig()).score(ep)
    assert score < 0.15
    assert signals["error"] == 0.0
    assert signals["failure"] == 0.0


def test_error_and_failure_episode_scores_high():
    ep = make_episode(
        [
            Step("user", "TASK: do x"),
            Step("assistant", "try a"),
            Step("tool", "Error: boom\nTraceback"),
            Step("assistant", "try a"),
            Step("tool", "Error: boom again"),
            Step("assistant", "giving up"),
        ],
        outcome="failure",
    )
    score, signals = HeuristicGate(GateConfig()).score(ep)
    assert score > 0.6
    assert signals["error"] == 1.0
    assert signals["retry"] == 1.0
    assert signals["failure"] == 1.0


def test_budget_caps_selection():
    episodes = synthetic_traces(100, seed=3)
    config = GateConfig(budget_fraction=0.2)
    decisions = select_for_reflection(episodes, config)
    n_selected = sum(1 for d in decisions if d.selected)
    assert 1 <= n_selected <= 20


def test_gate_recovers_planted_episodes():
    """Selected episodes should be heavily enriched in planted-learnworthy ones."""
    episodes = synthetic_traces(200, seed=4)
    planted = {ep.episode_id for ep in episodes if ep.meta["planted_learnworthy"]}
    decisions = select_for_reflection(episodes, GateConfig(budget_fraction=0.25))
    selected = {d.episode_id for d in decisions if d.selected}
    assert selected, "gate selected nothing"
    precision = len(selected & planted) / len(selected)
    assert precision > 0.9, f"gate precision too low: {precision:.2f}"


def test_min_score_floor():
    """A day of clean episodes should promote (almost) nothing."""
    episodes = [
        Episode(
            episode_id=f"clean-{i}",
            source="test",
            task="t",
            steps=[
                Step("user", "TASK: trivial"),
                Step("assistant", "plan"),
                Step("tool", "ok"),
                Step("assistant", "done"),
            ],
            outcome="success",
        )
        for i in range(20)
    ]
    decisions = select_for_reflection(episodes, GateConfig(budget_fraction=0.5, min_score=0.15))
    assert sum(1 for d in decisions if d.selected) == 0


def test_errors_in_user_observation_turns_are_counted():
    """Chat-style agent datasets (AgentInstruct) deliver env feedback as user
    turns; errors there must fire the gate just like tool-step errors."""
    ep = Episode(
        episode_id="obs-1",
        source="test",
        task="list files",
        steps=[
            Step("user", "TASK: list files"),
            Step("assistant", "Act: bash ls /nope"),
            Step("user", "bash: ls: cannot access '/nope': Error: no such file"),
            Step("assistant", "Act: bash ls /tmp"),
            Step("user", "ok"),
            Step("assistant", "Done."),
        ],
    )
    signals = heuristic_signals(ep)
    assert signals["error"] > 0


def test_task_instruction_mentioning_errors_does_not_fire():
    """The opening user turn is the instruction, not feedback — the word
    'error' there must not count."""
    ep = Episode(
        episode_id="obs-2",
        source="test",
        task="x",
        steps=[
            Step("user", "TASK: if you see an error or timeout, report it as failed"),
            Step("assistant", "Done, no issues."),
        ],
    )
    signals = heuristic_signals(ep)
    assert signals["error"] == 0
