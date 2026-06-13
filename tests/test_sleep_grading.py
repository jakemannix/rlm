"""Tests for judge-based probe grading (mock LM, no models)."""

from rlm.sleep.grading import grade_behavior, grade_cot, grade_over_application
from tests.mock_lm import MockLM


def test_grade_behavior_parses_last_verdict():
    yes = MockLM(response_fn=lambda _: "It asks the user first.\nVERDICT: YES")
    assert grade_behavior(yes, "scenario", "response", "ask before sampling")
    no = MockLM(response_fn=lambda _: "It just proceeds.\n**VERDICT:** NO")
    assert not grade_behavior(no, "s", "r", "ask first")


def test_grade_unparseable_is_false():
    lm = MockLM(response_fn=lambda _: "I am not sure, hard to say.")
    assert not grade_behavior(lm, "s", "r", "m")
    assert not grade_cot(lm, "r", "principle")
    assert not grade_over_application(lm, "s", "r", "refuse")


def test_grade_cot_and_over_application_route_to_right_prompt():
    seen = {}

    def respond(prompt) -> str:
        text = str(prompt)
        seen["cot"] = seen.get("cot") or "visibly REASONS" in text
        seen["neg"] = seen.get("neg") or "OVER-APPLIED" in text
        return "VERDICT: YES"

    lm = MockLM(response_fn=respond)
    assert grade_cot(lm, "resp", "check sources first")
    assert grade_over_application(lm, "routine scenario", "resp", "refuse")
    assert seen["cot"] and seen["neg"]
