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


def test_extract_json_salvages_sql_quote_escapes():
    """LLMs writing SQL inside JSON emit \\' (invalid JSON); we recover it."""
    text = (
        '{"verdict": "learn", "confidence": 0.9,'
        ' "lesson": "Quote LIKE patterns.",'
        ' "examples": [{"prompt": "Find penicillin allergies.",'
        ' "response": "SELECT * FROM allergies WHERE allergen LIKE \\\'%penicillin%\\\'"}]}'
    )
    parsed = extract_json(text)
    assert parsed["verdict"] == "learn"
    assert "%penicillin%" in parsed["examples"][0]["response"]


def test_learn_verdict_produces_verified_examples():
    # reflect (1 sample) then verify (1 per example)
    lm = MockLM(responses=[LEARN_RESPONSE, "No problems found.\nVERDICT: YES"])
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=1, verify_examples=True))
    out = judge.reflect(episode())
    assert out.verdict == "learn"
    assert len(out.examples) == 1
    assert out.examples[0].verified
    assert out.examples[0].source_episode_id == out.episode_id


def test_failed_verification_marks_example_unverified():
    lm = MockLM(responses=[LEARN_RESPONSE, "The response is wrong.\nVERDICT: NO"])
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


def test_unparseable_sample_counts_as_skip_vote():
    # 1 learn + 2 garbage samples: learn loses the majority -> skip.
    lm = MockLM(responses=[LEARN_RESPONSE, "I think we should learn!", "{broken json"])
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=3, verify_examples=False))
    out = judge.reflect(episode())
    assert out.verdict == "skip"
    assert out.n_parse_failures == 2


def test_all_samples_unparseable_yields_skip_not_crash():
    lm = MockLM(responses=["nope", "still nope"])
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=2))
    out = judge.reflect(episode())
    assert out.verdict == "skip"
    assert out.n_parse_failures == 2
    assert out.confidence == 1.0


def test_cache_second_reflect_makes_zero_lm_calls(tmp_path):
    cache = tmp_path / "judge_cache.json"
    ep = episode()

    lm1 = MockLM(responses=[LEARN_RESPONSE, "VERDICT: YES"])
    judge1 = ReflectionJudge(lm1, JudgeConfig(n_samples=1), cache_path=cache)
    first = judge1.reflect(ep)
    assert lm1._call_count == 2  # reflect + verify

    # Fresh judge instance, fresh LM with NO responses: any call would raise.
    lm2 = MockLM(responses=[])
    judge2 = ReflectionJudge(lm2, JudgeConfig(n_samples=1), cache_path=cache)
    second = judge2.reflect(ep)
    assert lm2._call_count == 0
    assert judge2.cache_hits == 1
    assert second.verdict == first.verdict
    assert second.examples[0].prompt == first.examples[0].prompt
    assert second.examples[0].verified == first.examples[0].verified


def test_cache_misses_on_different_judge_settings(tmp_path):
    cache = tmp_path / "judge_cache.json"
    ep = episode()
    judge1 = ReflectionJudge(
        MockLM(responses=[LEARN_RESPONSE, "VERDICT: YES"]),
        JudgeConfig(n_samples=1),
        cache_path=cache,
    )
    judge1.reflect(ep)

    # Same episode, different n_samples -> different key -> real calls again.
    lm = MockLM(responses=[LEARN_RESPONSE, LEARN_RESPONSE, "VERDICT: YES"])
    judge2 = ReflectionJudge(lm, JudgeConfig(n_samples=2), cache_path=cache)
    judge2.reflect(ep)
    assert lm._call_count == 3
    assert judge2.cache_hits == 0


def test_no_cache_path_means_no_caching():
    ep = episode()
    lm = MockLM(
        response_fn=lambda p: "VERDICT: YES"
        if "candidate training example" in str(p)
        else LEARN_RESPONSE
    )
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=1))
    judge.reflect(ep)
    calls_after_first = lm._call_count
    judge.reflect(ep)
    assert lm._call_count == 2 * calls_after_first


def test_parse_verify_verdict_variants():
    from rlm.sleep.judge import parse_verify_verdict

    assert parse_verify_verdict("VERDICT: YES")
    assert parse_verify_verdict("The SQL is correct.\nVERDICT: yes")
    assert not parse_verify_verdict("Wrong command.\nVERDICT: NO")
    assert not parse_verify_verdict("Looks good to me!")  # no verdict -> fail closed
    assert not parse_verify_verdict("")
    assert not parse_verify_verdict("VERDICT: YES\nWait, actually:\nVERDICT: NO")  # last wins


def test_verify_greedy_decodes_then_restores_temperature():
    """Sampling-capable judges are forced greedy for the verify call only."""
    from rlm.sleep.types import TrainingExample

    seen: list[float] = []

    def respond(prompt) -> str:
        seen.append(lm.temperature)
        return "VERDICT: YES"

    lm = MockLM(response_fn=respond)
    lm.temperature = 0.7
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=1, verify_examples=True))
    example = TrainingExample(prompt="p", response="r", lesson="l", source_episode_id="e")
    assert judge.verify(example)
    assert seen == [0.0]
    assert lm.temperature == 0.7


def test_cache_misses_on_different_max_new_tokens(tmp_path):
    """A longer generation budget changes reflection truncation -> new key."""
    cache = tmp_path / "judge_cache.json"
    ep = episode()
    lm1 = MockLM(responses=[LEARN_RESPONSE, "VERDICT: YES"])
    lm1.max_new_tokens = 768
    ReflectionJudge(lm1, JudgeConfig(n_samples=1), cache_path=cache).reflect(ep)

    lm2 = MockLM(responses=[LEARN_RESPONSE, "VERDICT: YES"])
    lm2.max_new_tokens = 1024
    judge2 = ReflectionJudge(lm2, JudgeConfig(n_samples=1), cache_path=cache)
    judge2.reflect(ep)
    assert judge2.cache_hits == 0
    assert lm2._call_count == 2


def test_cache_invalidated_by_verify_prompt_change(tmp_path, monkeypatch):
    """verified flags live inside cached outputs; a prompt edit must not reuse them."""
    import rlm.sleep.judge as judge_mod

    cache = tmp_path / "judge_cache.json"
    ep = episode()
    judge1 = ReflectionJudge(
        MockLM(responses=[LEARN_RESPONSE, "VERDICT: YES"]),
        JudgeConfig(n_samples=1),
        cache_path=cache,
    )
    judge1.reflect(ep)

    monkeypatch.setattr(judge_mod, "VERIFY_PROMPT", judge_mod.VERIFY_PROMPT + "\nBe stricter.")
    lm = MockLM(responses=[LEARN_RESPONSE, "VERDICT: NO"])
    judge2 = ReflectionJudge(lm, JudgeConfig(n_samples=1), cache_path=cache)
    out = judge2.reflect(ep)
    assert judge2.cache_hits == 0
    assert lm._call_count == 2
    assert not out.examples[0].verified


def test_malformed_examples_are_dropped_not_fatal():
    response = json.dumps(
        {
            "verdict": "learn",
            "confidence": 0.9,
            "lesson": "Check arguments.",
            "examples": [
                {"prompt": "Good prompt.", "response": "Good response."},
                {"prompt": "Missing response key."},
                "not even a dict",
            ],
        }
    )
    lm = MockLM(responses=[response])
    judge = ReflectionJudge(lm, JudgeConfig(n_samples=1, verify_examples=False))
    out = judge.reflect(episode())
    assert out.verdict == "learn"
    assert len(out.examples) == 1
    assert out.examples[0].prompt == "Good prompt."
