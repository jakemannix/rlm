"""Tests for the memory-worthiness judging task + judge eval metrics."""

from rlm.sleep.judge_worthiness import parse_worthiness, worthiness_messages
from scripts.sleep.train_judge import metrics


def test_parse_worthiness_tiers():
    assert parse_worthiness('{"keep": true, "tier": "gold", "reason": "x"}') == {
        "keep": True,
        "tier": "gold",
        "reason": "x",
    }
    # keep is derived from tier, not trusted from the model
    assert parse_worthiness('{"keep": true, "tier": "reject"}')["keep"] is False
    assert parse_worthiness('{"tier": "silver"}')["keep"] is True
    assert parse_worthiness("no json") is None
    assert parse_worthiness('{"tier": "maybe"}') is None  # invalid tier


def test_worthiness_messages_roundtrip():
    msgs = worthiness_messages("mem", "evid", {"keep": True, "tier": "gold"})
    assert msgs[0]["role"] == "user" and "mem" in msgs[0]["content"]
    assert msgs[1]["role"] == "assistant" and "gold" in msgs[1]["content"]
    assert len(worthiness_messages("m", "e")) == 1  # no label -> inference prompt only


def test_metrics_balanced_accuracy():
    rows = [
        {"label": {"keep": True}, "p": {"keep": True}},  # tp
        {"label": {"keep": True}, "p": {"keep": False}},  # fn
        {"label": {"keep": False}, "p": {"keep": False}},  # tn
        {"label": {"keep": False}, "p": {"keep": True}},  # fp
        {"label": {"keep": False}, "p": None},  # unparsed -> skipped
    ]
    m = metrics(rows, "p")
    assert m["n_parsed"] == 4
    assert m["keep_recall"] == 0.5 and m["reject_recall"] == 0.5
    assert m["balanced_accuracy"] == 0.5
