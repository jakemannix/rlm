"""Tests for the reflection judge (with a mock LM)."""

import json

import pytest

from rlm.sleep.config import JudgeConfig
from rlm.sleep.judge import ReflectionJudge, extract_json
from rlm.sleep.traces import synthetic_traces
from tests.mock_lm import MockLM

LEARN_RESPONSE = json.dumps(
    {
        "verdict": "learn",
        "confidence": 0.9,
        "lesson": "Validate tool arguments before the first call.",
        "examples": [
            {
                "prompt": "Run a tool that takes a numeric id safely.",
                "response": "First validate the id is numeric, then call the tool once.",
            }
        ],
    }
)

SKIP_RESPONSE = json.dumps({"verdict": "skip", "confidence": 0.8, "lesson": "", "examples": []})


def episode():
    return synthetic_traces(1, seed=5)[0]


def test_extract_json_from_chatty_response():
    text = f"Sure! Here is my analysis:\n```json\n{LEARN_RESPONSE}\n```\nHope this helps."
    parsed = extract_json(text)
    assert parsed["verdict"] == "learn"


def test_extract_json_raises_without_json():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_learn_verdict_produces_verified_examples():
    # reflect (1 sample) then verify (1 per example)
    lm = MockLM(responses=[LEARN_RESPONSE, "YES"])
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=1, verify_examples=True))
    out = judge.reflect(episode())
    assert out.verdict == "learn"
    assert len(out.examples) == 1
    assert out.examples[0].verified
    assert out.examples[0].source_episode_id == out.episode_id


def test_failed_verification_marks_example_unverified():
    lm = MockLM(responses=[LEARN_RESPONSE, "NO, the response is wrong."])
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=1, verify_examples=True))
    out = judge.reflect(episode())
    assert out.verdict == "learn"
    assert not out.examples[0].verified


def test_skip_verdict_short_circuits():
    lm = MockLM(responses=[SKIP_RESPONSE])
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=1))
    out = judge.reflect(episode())
    assert out.verdict == "skip"
    assert out.examples == []


def test_self_consistency_majority_vote():
    # 1 learn vs 2 skips -> skip wins.
    lm = MockLM(responses=[LEARN_RESPONSE, SKIP_RESPONSE, SKIP_RESPONSE])
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=3, verify_examples=False))
    out = judge.reflect(episode())
    assert out.verdict == "skip"


def test_low_confidence_learn_is_demoted():
    low_conf = json.dumps({"verdict": "learn", "confidence": 0.2, "lesson": "meh", "examples": []})
    lm = MockLM(responses=[low_conf])
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=1, min_confidence=0.5))
    out = judge.reflect(episode())
    assert out.verdict == "skip"
