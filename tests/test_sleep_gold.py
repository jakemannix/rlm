"""Tests for the offline gold-labeling stack (mock LM, no models)."""

import json

from rlm.sleep.config import JudgeConfig
from rlm.sleep.gold import (
    GoldConfig,
    label_episodes,
    label_example,
    refute,
    score_rubric,
    summarize,
)
from rlm.sleep.types import Episode, Step, TrainingExample
from tests.mock_lm import MockLM


def episode() -> Episode:
    return Episode(
        episode_id="cc_test_000",
        source="claude_code/test",
        task="Fix the failing import",
        steps=[
            Step(role="assistant", content="[tool:Bash] pytest -q"),
            Step(role="tool", content="ModuleNotFoundError: no module named foo"),
            Step(role="assistant", content="Installed foo; tests pass."),
        ],
        outcome="success",
    )


def example(verified: bool = True) -> TrainingExample:
    return TrainingExample(
        prompt="Tests fail with ModuleNotFoundError: foo. What now?",
        response="Install the missing package (uv pip install foo), then rerun pytest.",
        lesson="A ModuleNotFoundError means a missing dependency, not broken code.",
        source_episode_id="cc_test_000",
        verified=verified,
    )


def rubric_json(score: float, flaw: str = "") -> str:
    scores = {
        d: score
        for d in ("correctness", "reusability", "grounding", "specificity", "self_contained")
    }
    return json.dumps({"scores": scores, "fatal_flaw": flaw})


def test_rubric_scores_average_across_samples():
    lm = MockLM(responses=[rubric_json(5), rubric_json(3), rubric_json(4)])
    scores, flaws, failures = score_rubric(lm, episode(), example(), GoldConfig())
    assert scores["correctness"] == 4.0
    assert failures == 0 and flaws == []


def test_rubric_parse_failures_counted_and_survivable():
    lm = MockLM(responses=["not json at all", rubric_json(4), rubric_json(4)])
    scores, _, failures = score_rubric(lm, episode(), example(), GoldConfig())
    assert failures == 1
    assert scores["grounding"] == 4.0


def test_unverified_example_rejected_without_judge_calls():
    lm = MockLM(responses=[])  # any call would raise
    result = label_example(lm, episode(), example(verified=False), GoldConfig())
    assert result.label == "reject"
    assert lm._call_count == 0


def test_gold_label_requires_score_and_refute_survival():
    lm = MockLM(
        responses=[
            rubric_json(5),
            rubric_json(4),
            rubric_json(4.5),
            "VERDICT: YES",
            "weak objection\nVERDICT: YES",
            "VERDICT: NO",
        ]
    )
    result = label_example(lm, episode(), example(), GoldConfig())
    assert result.label == "gold"
    assert result.refute_keep_votes == 2 and result.refute_votes == 3
    assert result.mean_score == 4.5


def test_refuter_majority_rejects_despite_high_scores():
    lm = MockLM(
        responses=[
            rubric_json(5),
            rubric_json(5),
            rubric_json(5),
            "VERDICT: NO",
            "VERDICT: NO",
            "VERDICT: YES",
        ]
    )
    result = label_example(lm, episode(), example(), GoldConfig())
    assert result.label == "reject"
    assert result.mean_score == 5.0  # scores preserved for re-thresholding


def test_fatal_flaw_majority_short_circuits_refuters():
    lm = MockLM(
        responses=[
            rubric_json(5, "teaches a destructive command"),
            rubric_json(5, "harmful"),
            rubric_json(5),
        ]
    )
    result = label_example(lm, episode(), example(), GoldConfig())
    assert result.label == "reject"
    assert result.refute_votes == 0  # never reached the refuters
    assert len(result.fatal_flaws) == 2


def test_silver_band():
    lm = MockLM(
        responses=[
            rubric_json(3.5),
            rubric_json(3.5),
            rubric_json(3.5),
            "VERDICT: YES",
            "VERDICT: YES",
            "VERDICT: YES",
        ]
    )
    result = label_example(lm, episode(), example(), GoldConfig())
    assert result.label == "silver"


def test_all_rubric_samples_unparseable_rejects():
    lm = MockLM(responses=["nope", "still nope", "garbage"])
    result = label_example(lm, episode(), example(), GoldConfig())
    assert result.label == "reject"
    assert result.n_rubric_failures == 3


def test_refute_parses_fail_closed():
    lm = MockLM(responses=["I have thoughts but no verdict", "VERDICT: YES", "VERDICT: YES"])
    keep, total = refute(lm, example(), GoldConfig())
    assert (keep, total) == (2, 3)


def test_label_episodes_end_to_end_and_summary():
    learn = json.dumps(
        {
            "verdict": "learn",
            "confidence": 0.9,
            "lesson": "A ModuleNotFoundError means a missing dependency.",
            "examples": [
                {
                    "prompt": "Tests fail with ModuleNotFoundError. Fix?",
                    "response": "Install the missing package and rerun.",
                }
            ],
        }
    )

    def respond(prompt) -> str:
        text = str(prompt)
        if "scoring one candidate" in text:
            return rubric_json(4.5)
        if "candidate training example" in text or "last gate" in text:
            return "VERDICT: YES"
        return learn

    labels = label_episodes(
        [episode()],
        MockLM(response_fn=respond),
        judge_config=JudgeConfig(n_samples=1),
        gold_config=GoldConfig(),
    )
    assert len(labels) == 1
    assert labels[0].label == "gold"
    stats = summarize(labels)
    assert stats["n_gold"] == 1 and stats["n_candidates"] == 1
    assert stats["dimension_means"]["correctness"] == 4.5
