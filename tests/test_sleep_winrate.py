"""Tests for the blind pairwise win-rate eval (mock judge; no models)."""

from rlm.sleep.winrate import PAIRWISE_PROMPT, judged_win_rate, parse_verdict
from tests.mock_lm import MockLM


def test_parse_verdict_variants():
    assert parse_verdict("A") == "A"
    assert parse_verdict("  b.\nbecause...") == "B"
    assert parse_verdict("TIE") == "tie"
    assert parse_verdict("Both are fine") == "tie"  # unparseable -> tie
    assert parse_verdict("") == "tie"


def test_blind_order_maps_verdicts_back_correctly():
    """A judge that always prefers the adapted text should win every item,
    regardless of which presentation slot the adapted response landed in."""

    def prefer_adapted(prompt) -> str:
        text = str(prompt)
        a_section = text.split("RESPONSE A:")[1].split("RESPONSE B:")[0]
        return "A" if "ADAPTED" in a_section else "B"

    items = [(f"task {i}", "base answer", "ADAPTED answer") for i in range(20)]
    report = judged_win_rate(MockLM(response_fn=prefer_adapted), items, seed=3)
    assert report.wins == 20
    assert report.losses == 0
    assert report.win_rate == 1.0


def test_position_biased_judge_scores_near_chance():
    """A judge that always says "A" should not systematically favor either
    model, because presentation order is randomized."""
    items = [(f"task {i}", "base", "adapted") for i in range(100)]
    report = judged_win_rate(MockLM(response_fn=lambda _: "A"), items, seed=0)
    assert report.n == 100
    assert 0.35 <= report.win_rate <= 0.65


def test_ties_are_excluded_from_win_rate():
    responses = iter(["TIE", "TIE", "A", "TIE"])
    items = [(f"t{i}", "b", "a") for i in range(4)]
    report = judged_win_rate(MockLM(response_fn=lambda _: next(responses)), items, seed=1)
    assert report.ties == 3
    assert report.wins + report.losses == 1
    assert report.win_rate in (0.0, 1.0)


def test_prompt_contains_both_responses():
    assert "RESPONSE A" in PAIRWISE_PROMPT and "RESPONSE B" in PAIRWISE_PROMPT
