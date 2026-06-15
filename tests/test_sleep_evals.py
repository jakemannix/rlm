"""Tests for the NLL eval plumbing (stub model/tokenizer; no downloads)."""

import pytest

from rlm.sleep.config import GateConfig
from rlm.sleep.gate import select_for_reflection
from rlm.sleep.traces import synthetic_traces

torch = pytest.importorskip("torch")

from rlm.sleep.evals import response_nll  # noqa: E402


class CharTokenizer:
    """One token per character; chat template is the raw concatenation."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        return {"input_ids": [ord(c) for c in text]}


class LabelCapturingModel:
    def __init__(self):
        self.last_input_ids = None
        self.last_labels = None

    def __call__(self, input_ids, labels=None):
        self.last_input_ids = input_ids
        self.last_labels = labels

        class Out:
            loss = torch.tensor(1.23)

        return Out()


def supervised_text(model) -> str:
    ids = model.last_input_ids[0]
    labels = model.last_labels[0]
    return "".join(chr(int(i)) for i, lab in zip(ids, labels, strict=True) if int(lab) != -100)


def test_short_prompt_scores_exactly_the_response():
    model = LabelCapturingModel()
    nll = response_nll(
        model, CharTokenizer(), [{"role": "user", "content": "hi "}], "yes", "cpu", max_seq_len=64
    )
    assert nll == pytest.approx(1.23)
    assert supervised_text(model) == "yes"


def test_long_prompt_is_left_truncated_response_survives():
    """A prompt longer than the window must not displace the response —
    the regression here silently scored prompt tokens instead."""
    model = LabelCapturingModel()
    long_prompt = "p" * 500
    response_nll(
        model,
        CharTokenizer(),
        [{"role": "user", "content": long_prompt}],
        "RESPONSE",
        "cpu",
        max_seq_len=32,
    )
    assert supervised_text(model) == "RESPONSE"
    assert model.last_input_ids.shape[1] == 32  # window fully used: prompt tail + response


def test_huge_response_is_capped_at_half_window():
    model = LabelCapturingModel()
    response_nll(
        model,
        CharTokenizer(),
        [{"role": "user", "content": "p" * 100}],
        "r" * 100,
        "cpu",
        max_seq_len=20,
    )
    assert supervised_text(model) == "r" * 10
    assert model.last_input_ids.shape[1] == 20


class RawNLLGate:
    """Fake surprise gate emitting absolute NLLs far above the old /4 squash."""

    def __init__(self, nll_by_id):
        self.nll_by_id = nll_by_id

    def score(self, episode) -> float:
        return self.nll_by_id(episode.episode_id)


def test_surprise_is_normalized_within_day_not_absolute():
    """All NLLs > 4 nats would have saturated the old absolute squash at 1.0;
    min-max normalization keeps them discriminative."""
    episodes = synthetic_traces(10, seed=2)
    gate = RawNLLGate(lambda eid: 5.0 + 3.0 * (int(eid.rsplit("-", 1)[1]) % 3))
    decisions = select_for_reflection(episodes, GateConfig(), surprise_gate=gate)
    normalized = {d.signals["surprise"] for d in decisions}
    assert normalized == {0.0, 0.5, 1.0}
    for d in decisions:
        assert d.signals["surprise_nll"] >= 5.0  # raw NLL preserved for inspection


def test_identical_surprise_collapses_to_neutral():
    episodes = synthetic_traces(5, seed=2)
    decisions = select_for_reflection(
        episodes, GateConfig(), surprise_gate=RawNLLGate(lambda _: 7.7)
    )
    assert all(d.signals["surprise"] == 0.5 for d in decisions)
