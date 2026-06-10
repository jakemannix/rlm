"""
Hugging Face causal-LM wrapper that splices a Titans memory module into
the hidden-state stream.

The wrapper supports a "memory-as-context" (MAC) integration: the model's
own hidden states are projected into ``(key, value, query)`` triples; the
Titans memory module is updated on the keys/values and queried with the
queries; the resulting memory read is added back into the hidden stream.

The wrapper is intentionally minimal — its goal is to make it easy to
benchmark a parametric memory module on top of a small open-weights model
(Gemma 2-2B, Gemma 3-1B-IT, ...) without having to fork the HF
implementation.  For research integration with a specific architecture,
prefer hooking the memory in at a chosen transformer block.

Example
-------
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from rlm.memory import MemoryConfig, TitansAugmentedLM
    >>> tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    >>> base = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
    >>> lm = TitansAugmentedLM(base, MemoryConfig(key_dim=256, value_dim=256))
    >>> out = lm.generate(tok("Hello", return_tensors="pt").input_ids,
    ...                    max_new_tokens=32)
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover - torch is an optional dep
    raise ImportError(
        "rlm.memory.modeling requires torch.  Install with `pip install 'rlms[titans]'`."
    ) from e

from rlm.memory.config import MemoryConfig
from rlm.memory.miras import MirasMemory
from rlm.memory.titans import TitansMemory


def _resolve_hidden_size(base_model: nn.Module) -> int:
    cfg = getattr(base_model, "config", None)
    if cfg is None:
        raise ValueError("base_model must have a HF ``config`` attribute.")
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError("Could not infer hidden size from the base model config.")


class TitansAugmentedLM(nn.Module):
    """
    Wrap a HF causal LM with a Titans/Miras memory module.

    Parameters
    ----------
    base_model
        A Hugging Face ``AutoModelForCausalLM`` instance.
    cfg
        :class:`MemoryConfig`.  Its ``key_dim``/``value_dim`` may be ``0``
        in which case they default to the base model hidden size.
    flavor
        ``"titans"`` (default) or ``"miras"``.  Selects the underlying
        memory class.
    insert_layer
        Index of the transformer block whose output we augment.  ``-1``
        (default) means "after the last block, before the LM head".
    freeze_base
        If True, the base model parameters are frozen.  Useful when
        meta-training only the memory module on top of a fixed LM.
    """

    def __init__(
        self,
        base_model: nn.Module,
        cfg: MemoryConfig,
        flavor: str = "titans",
        insert_layer: int = -1,
        freeze_base: bool = True,
    ):
        super().__init__()
        hidden = _resolve_hidden_size(base_model)
        if cfg.key_dim == 0:
            cfg.key_dim = hidden
        if cfg.value_dim == 0:
            cfg.value_dim = hidden

        self.base_model = base_model
        self.cfg = cfg
        self.hidden_size = hidden
        self.insert_layer = insert_layer

        memory_cls = {"titans": TitansMemory, "miras": MirasMemory}[flavor]
        self.memory = memory_cls(cfg)

        # Projections between the model's hidden space and the memory's
        # key/value/query space.
        self.k_proj = nn.Linear(hidden, cfg.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden, cfg.value_dim, bias=False)
        self.q_proj = nn.Linear(hidden, cfg.key_dim, bias=False)
        self.out_proj = nn.Linear(cfg.value_dim, hidden, bias=False)
        for p in (self.k_proj, self.v_proj, self.q_proj, self.out_proj):
            nn.init.normal_(p.weight, std=cfg.init_scale)

        # Per-conversation memory state.  ``None`` until ``reset_memory`` is
        # called or the first forward pass.
        self._mem_state: dict[str, Any] | None = None
        self._hook_handle = None
        self._captured: torch.Tensor | None = None

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # Place the new memory + projection submodules on the same device and
        # dtype as the (already-loaded) base model, so the wrapper is usable on
        # whatever device the base sits on (CPU / CUDA / XLA) without the caller
        # remembering a trailing ``.to(device)``.
        ref = next(self.base_model.parameters(), None)
        if ref is not None:
            for sub in (self.k_proj, self.v_proj, self.q_proj, self.out_proj, self.memory):
                sub.to(device=ref.device, dtype=ref.dtype)

        self._install_hook()

    # ------------------------------------------------------------------
    # Memory state management
    # ------------------------------------------------------------------
    def reset_memory(self, batch_size: int = 1, device=None, dtype=None) -> None:
        """Wipe any per-sequence memory state."""
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        self._mem_state = self.memory.init_state(batch_size, device=device, dtype=dtype)

    @property
    def memory_state(self) -> dict[str, Any] | None:
        return self._mem_state

    # ------------------------------------------------------------------
    # Hook plumbing
    # ------------------------------------------------------------------
    def _resolve_target_module(self) -> nn.Module:
        """Find the block whose output we should augment."""
        # Try common HF naming conventions.
        m = self.base_model
        layers = None
        for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
            obj = m
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok:
                layers = obj
                break
        if layers is None:
            raise ValueError(
                "Could not locate transformer blocks on the base model. "
                "Pass a model that exposes one of: model.layers, "
                "transformer.h, gpt_neox.layers, model.decoder.layers."
            )
        idx = self.insert_layer if self.insert_layer >= 0 else len(layers) - 1
        return layers[idx]

    def _install_hook(self) -> None:
        target = self._resolve_target_module()

        def hook(_module, _inputs, output):
            # HF blocks return either a tensor or a tuple (hidden, ...).
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            augmented = self._apply_memory(hidden)
            if rest:
                return (augmented, *rest)
            return augmented

        self._hook_handle = target.register_forward_hook(hook)

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ------------------------------------------------------------------
    # Memory application
    # ------------------------------------------------------------------
    def _apply_memory(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3:
            return hidden
        bsz = hidden.shape[0]

        keys = self.k_proj(hidden)
        values = self.v_proj(hidden)
        queries = self.q_proj(hidden)

        # Source state.  In TRAIN mode start each forward from a fresh,
        # differentiable init state: the outer-loop gradient then reaches the
        # initial memory parameters (not just the projections/gates) and each
        # sequence is independent.  Persisting + detaching the state across
        # calls — as the eval path does for memory-as-context across tokens /
        # turns — would silently keep the memory MLP out of the graph, which is
        # the meta-training bug this avoids.
        if self.training:
            src = self.memory.init_state(bsz, hidden.device, hidden.dtype, detach=False)
        else:
            if (
                self._mem_state is None
                or self._mem_state["params"][next(iter(self._mem_state["params"]))].shape[0] != bsz
            ):
                self.reset_memory(bsz, hidden.device, hidden.dtype)
            src = self._mem_state

        # Upcast to float32 for the online update's numerical stability; the
        # memory module RMS-normalizes keys/values internally, so no boundary
        # normalization is needed here.
        keys_f = keys.to(torch.float32)
        values_f = values.to(torch.float32)
        queries_f = queries.to(torch.float32)
        state_f = {
            "params": {k: v.to(torch.float32) for k, v in src["params"].items()},
            "momentum": {k: v.to(torch.float32) for k, v in src["momentum"].items()},
        }
        read, new_state = self.memory.read_write(keys_f, values_f, queries_f, state_f)
        if not self.training:
            # Persist across calls only in eval; detach so the graph can't grow.
            self._mem_state = {
                "params": {k: v.to(hidden.dtype).detach() for k, v in new_state["params"].items()},
                "momentum": {
                    k: v.to(hidden.dtype).detach() for k, v in new_state["momentum"].items()
                },
            }
        out = self.out_proj(read.to(hidden.dtype))
        return hidden + out

    # ------------------------------------------------------------------
    # Standard nn.Module interface — just delegate to the base model.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)
