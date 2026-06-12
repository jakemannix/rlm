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


def test_refute_parses_fail_closed_and_counts_failures():
    lm = MockLM(responses=["I have thoughts but no verdict", "VERDICT: YES", "VERDICT: YES"])
    keep, total, failures = refute(lm, example(), GoldConfig())
    assert (keep, total, failures) == (2, 3, 1)


def test_refute_template_echo_does_not_count_as_reject():
    """A judge that echoes the prompt's format spec (which ends 'VERDICT: NO')
    before answering must be judged on its OWN final line."""
    from rlm.sleep.gold import parse_refute_verdict

    echo_then_answer = "VERDICT: YES\nor\nVERDICT: NO\n\nNo real flaw.\nVERDICT: YES"
    assert parse_refute_verdict(echo_then_answer) is True
    pure_echo = "VERDICT: YES\nor\nVERDICT: NO"
    assert parse_refute_verdict(pure_echo) is False  # template's last line, fails closed
    assert parse_refute_verdict("rambling with no verdict") is None
    assert parse_refute_verdict("") is None


def test_fatal_flaw_placeholders_not_counted():
    """Small judges write 'None'/'N/A' instead of an empty string."""
    lm = MockLM(
        responses=[
            rubric_json(5, "None"),
            rubric_json(5, "no fatal flaw."),
            rubric_json(5, "N/A"),
            "VERDICT: YES",
            "VERDICT: YES",
            "VERDICT: YES",
        ]
    )
    result = label_example(lm, episode(), example(), GoldConfig())
    assert result.fatal_flaws == []
    assert result.label == "gold"


def test_single_parsed_rubric_sample_cannot_mint_gold():
    lm = MockLM(
        responses=[
            "garbage",
            "also garbage",
            rubric_json(5),
            "VERDICT: YES",
            "VERDICT: YES",
            "VERDICT: YES",
        ]
    )
    result = label_example(lm, episode(), example(), GoldConfig())
    assert result.n_rubric_failures == 2
    assert result.label == "silver"  # high score, but uncorroborated


def test_gold_cache_second_run_makes_zero_calls(tmp_path):
    from rlm.sleep.gold import _CallCache

    cache_file = tmp_path / "gold_cache.json"
    lm1 = MockLM(
        responses=[
            rubric_json(5),
            rubric_json(5),
            rubric_json(5),
            "VERDICT: YES",
            "VERDICT: YES",
            "VERDICT: YES",
        ]
    )
    first = label_example(lm1, episode(), example(), GoldConfig(), cache=_CallCache(cache_file))
    assert first.label == "gold" and lm1._call_count == 6

    lm2 = MockLM(responses=[])  # any call would raise
    second = label_example(lm2, episode(), example(), GoldConfig(), cache=_CallCache(cache_file))
    assert second.label == "gold"
    assert lm2._call_count == 0


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
