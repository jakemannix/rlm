"""Regression tests for masked-SFT encoding (needs the real tokenizer)."""

import pytest

pytest.importorskip("transformers")

from rlm.sleep.lora import encode_example  # noqa: E402
from rlm.sleep.types import TrainingExample  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL)
    except OSError:  # pragma: no cover - offline machine without the cache
        pytest.skip("tokenizer not cached locally")


def example(prompt: str, response: str) -> TrainingExample:
    return TrainingExample(
        prompt=prompt, response=response, lesson="", source_episode_id="x", verified=True
    )


def test_long_prompt_never_yields_all_masked_labels(tokenizer):
    """Right-truncation used to drop the response entirely -> NaN loss."""
    long_prompt = "word " * 5000  # far beyond the window
    ids, labels = encode_example(tokenizer, example(long_prompt, "REMEMBER THIS."), None, 2048)
    assert len(ids) <= 2048
    supervised = [t for t in labels if t != -100]
    assert supervised, "response tokens must survive truncation"
    assert "REMEMBER THIS." in tokenizer.decode(supervised)


def test_short_example_supervises_exactly_the_response(tokenizer):
    ids, labels = encode_example(tokenizer, example("Quick question?", "Short answer."), None, 2048)
    assert len(ids) == len(labels)
    supervised = [t for t in labels if t != -100]
    decoded = tokenizer.decode(supervised)
    assert "Short answer." in decoded
    assert "Quick question?" not in decoded


def test_long_response_capped_at_half_window(tokenizer):
    ids, labels = encode_example(tokenizer, example("p", "answer " * 4000), None, 1024)
    assert len(ids) <= 1024
    supervised = [t for t in labels if t != -100]
    assert 0 < len(supervised) <= 512
