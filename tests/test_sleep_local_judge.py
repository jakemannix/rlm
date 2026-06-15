"""Tests for the local self-judge (stubbed model/tokenizer; no downloads)."""

import pytest

from rlm.sleep.local_judge import LocalHFJudge, resolve_torch_dtype

torch = pytest.importorskip("torch")


class StubTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        assert add_generation_prompt
        return f"<chat>{messages[-1]['content']}</chat>"

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        # 5 prompt tokens regardless of text content.
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(t)}" for t in ids)


class StubModel:
    def __init__(self):
        self.last_gen_kwargs = None

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.last_gen_kwargs = kwargs
        # Echo the prompt then "generate" three new tokens.
        new = torch.tensor([[7, 8, 9]])
        return torch.cat([input_ids, new], dim=1)


def make_judge(**kwargs) -> LocalHFJudge:
    return LocalHFJudge(StubModel(), StubTokenizer(), model_name="stub", device="cpu", **kwargs)


def test_completion_returns_only_new_tokens():
    judge = make_judge()
    out = judge.completion("reflect on this episode")
    assert out == "tok7 tok8 tok9"


def test_sampling_knobs_reach_generate():
    judge = make_judge(temperature=0.9, max_new_tokens=123)
    judge.completion("x")
    kwargs = judge.model.last_gen_kwargs
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.9
    assert kwargs["max_new_tokens"] == 123


def test_zero_temperature_is_greedy():
    judge = make_judge(temperature=0.0)
    judge.completion("x")
    kwargs = judge.model.last_gen_kwargs
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs


def test_usage_is_tracked():
    judge = make_judge()
    judge.completion("a")
    judge.completion("b")
    usage = judge.get_usage_summary().model_usage_summaries["stub"]
    assert usage.total_calls == 2
    assert usage.total_input_tokens == 10  # 5 prompt tokens per call
    assert usage.total_output_tokens == 6  # 3 new tokens per call


def test_resolve_torch_dtype():
    assert resolve_torch_dtype("auto", "cpu") == "float32"
    assert resolve_torch_dtype("auto", "cuda") == "bfloat16"
    assert resolve_torch_dtype("float16", "cpu") == "float16"
