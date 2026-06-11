"""Self-judging: the policy model as its own LLM-as-a-Judge.

The consolidation premise (``docs/learning_signal.md``) is the
generation-verification gap *within one model* — no smarter teacher.
:class:`LocalHFJudge` wraps a local HuggingFace causal LM behind the
:class:`~rlm.clients.base_lm.BaseLM` interface so :class:`ReflectionJudge`
can spend test-time compute (self-consistency samples, long generations)
on the very model that will be trained.

The judge always runs the *frozen base* weights — never tonight's adapter —
so reflection quality is stationary across nights and multi-night curves
measure adapter learning rather than judge drift.
"""

from __future__ import annotations

from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


def resolve_torch_dtype(torch_dtype: str, device: str) -> str:
    """Resolve the "auto" dtype: float32 on CPU, bfloat16 on accelerators."""
    if torch_dtype != "auto":
        return torch_dtype
    return "float32" if device == "cpu" else "bfloat16"


class LocalHFJudge(BaseLM):
    """A ``BaseLM`` over an in-process transformers model.

    ``temperature`` and ``max_new_tokens`` are the judge-side TTC knobs:
    sampling diversity feeds the self-consistency vote, and reflection
    gets a longer leash than ordinary policy decoding.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        device: str = "cpu",
        temperature: float = 0.7,
        max_new_tokens: int = 768,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0

    @classmethod
    def from_policy(
        cls,
        model_name: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        temperature: float = 0.7,
        max_new_tokens: int = 768,
    ) -> LocalHFJudge:
        """Load the policy model once and judge with it (frozen, eval mode)."""
        from rlm.sleep.lora import load_policy

        model, tokenizer = load_policy(
            model_name, device=device, torch_dtype=resolve_torch_dtype(torch_dtype, device)
        )
        model.eval()
        return cls(
            model,
            tokenizer,
            model_name=model_name,
            device=device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def completion(self, prompt: str | dict[str, Any]) -> str:
        import torch

        text = prompt if isinstance(prompt, str) else str(prompt)
        chat_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs |= {"do_sample": True, "temperature": self.temperature}
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            output = self.model.generate(
                input_ids, attention_mask=encoded["attention_mask"].to(self.device), **gen_kwargs
            )
        new_tokens = output[0, input_ids.shape[1] :]
        self._calls += 1
        self._input_tokens += int(input_ids.shape[1])
        self._output_tokens += int(new_tokens.shape[0])
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self._calls,
                    total_input_tokens=self._input_tokens,
                    total_output_tokens=self._output_tokens,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=self._calls, total_input_tokens=0, total_output_tokens=0
        )
