"""
Hugging Face client augmented with a Titans / Miras parametric memory.

This is a :class:`BaseLM` subclass so it plugs into the rest of the RLM
machinery (sub-LM calls inside the REPL, usage tracking, etc.).

By default it loads ``google/gemma-3-1b-it`` and wraps it with a Titans
memory module configured via :class:`rlm.memory.MemoryConfig`.  Override
``model_name`` / ``model_kwargs`` / ``memory_config`` to use a different
backbone.

Heavy dependencies (``torch``, ``transformers``) are loaded lazily so the
rest of the RLM library does not require them.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.memory.config import MemoryConfig


class TitansHFClient(BaseLM):
    """
    Local Hugging Face causal LM wrapped with a Titans/Miras memory module.

    Parameters
    ----------
    model_name
        HF model id (defaults to ``google/gemma-3-1b-it``).
    memory_config
        :class:`MemoryConfig`.  If ``None``, a small default is used.
    flavor
        ``"titans"`` (default) or ``"miras"``.
    device
        Where to place the model.  Defaults to ``"cuda"`` if available
        else ``"cpu"``.
    torch_dtype
        Optional dtype override (e.g. ``"bfloat16"``).
    insert_layer
        Index of the transformer block whose output is augmented (passed
        through to :class:`rlm.memory.TitansAugmentedLM`).
    freeze_base
        If True, the backbone weights are frozen (default — the memory is
        the only trainable part).
    max_new_tokens
        Cap on generation length per call.
    persistent_memory
        If True (default), the memory state persists across calls on this
        client instance.  Call :meth:`reset_memory` to wipe it.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        memory_config: MemoryConfig | None = None,
        flavor: str = "titans",
        device: str | None = None,
        torch_dtype: str | None = None,
        insert_layer: int = -1,
        freeze_base: bool = True,
        max_new_tokens: int = 512,
        persistent_memory: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.memory_config = memory_config or MemoryConfig(
            key_dim=0, value_dim=0, hidden_dim=256, n_layers=2
        )
        self.flavor = flavor
        self.max_new_tokens = max_new_tokens
        self.persistent_memory = persistent_memory

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from rlm.memory.modeling import TitansAugmentedLM

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dtype = None
        if torch_dtype is not None:
            dtype = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device if device != "cpu" else None
        )
        if device == "cpu":
            base = base.to(device)
        self.model = TitansAugmentedLM(
            base,
            self.memory_config,
            flavor=flavor,
            insert_layer=insert_layer,
            freeze_base=freeze_base,
        ).to(device)
        self.model.eval()
        self._torch = torch

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    # ------------------------------------------------------------------
    # Memory controls
    # ------------------------------------------------------------------
    def reset_memory(self) -> None:
        """Wipe the per-conversation memory state."""
        self.model.reset_memory(batch_size=1, device=self.device)

    # ------------------------------------------------------------------
    # BaseLM API
    # ------------------------------------------------------------------
    def _render_prompt(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        text = self._render_prompt(prompt)
        if not self.persistent_memory:
            self.reset_memory()

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs.input_ids.shape[1])
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion_tokens = int(out.shape[1]) - prompt_tokens
        response_ids = out[0, inputs.input_ids.shape[1] :]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self._track(prompt_tokens, completion_tokens, model or self.model_name)
        return response

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        # No real async inference for local HF models; run sync.
        return self.completion(prompt, model)

    def _track(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += prompt_tokens
        self.model_output_tokens[model] += completion_tokens
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                m: ModelUsageSummary(
                    total_calls=self.model_call_counts[m],
                    total_input_tokens=self.model_input_tokens[m],
                    total_output_tokens=self.model_output_tokens[m],
                    total_cost=None,
                )
                for m in self.model_call_counts
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            total_cost=None,
        )
