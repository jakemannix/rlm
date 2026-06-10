"""
Titans neural long-term memory (Behrouz et al., 2025).

Implements the parametric memory MLP M with the online update rule::

    grad_t = nabla_theta  L( M(k_t) , v_t )        # "surprise"
    S_t    = eta_t * S_{t-1} - lr_t * grad_t       # momentum surprise
    M_t    = (1 - alpha_t) * M_{t-1} + S_t         # forget + write

The whole update is differentiable through ``torch.func.functional_call`` so
that the *initial* parameters of M (and the gate networks) can be trained
end-to-end with the rest of the model while the *online* updates happen
during the forward pass at test time.

Notes
-----
* The retention surrogate ``L`` is the L2 loss in the original Titans paper;
  the Miras generalisation (l1/huber/kl) is implemented in
  :mod:`rlm.memory.miras` by subclassing :class:`TitansMemory`.
* The memory is "Memory-as-Context" style: callers project tokens into
  ``(keys, values, queries)`` triples, write ``(keys, values)``, then
  read ``M(queries)``.  See :mod:`rlm.memory.modeling` for the HF wrapper
  that does the projection.
* Chunked updates (``MemoryConfig.chunk_size > 1``) accumulate the gradient
  across the chunk before applying it — the chunk-wise variant from the
  Titans paper, which is much faster on accelerators.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
    from torch.func import functional_call, grad
except ImportError as e:  # pragma: no cover - torch is an optional dep
    raise ImportError(
        "rlm.memory requires torch.  Install with `pip install 'rlms[titans]'` "
        "or `pip install torch`."
    ) from e

from rlm.memory.config import MemoryConfig


def _activation(name: str) -> nn.Module:
    return {
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
    }[name]


class MemoryMLP(nn.Module):
    """The small MLP whose weights serve as the long-term memory state."""

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg = cfg
        dims = [cfg.key_dim] + [cfg.hidden_dim] * (cfg.n_layers - 1) + [cfg.value_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_activation(cfg.activation))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_scale)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TitansMemory(nn.Module):
    """
    Titans parametric memory module.

    Parameters
    ----------
    cfg
        :class:`MemoryConfig` describing geometry and update rule.

    Forward
    -------
    ``read_write(keys, values, queries)`` returns ``M(queries)`` *after*
    writing ``(keys, values)`` into the memory.  All three tensors are
    shaped ``(batch, seq, dim)``.
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg = cfg
        self.memory = MemoryMLP(cfg)

        # Persistent (data-independent) token concatenated with reads.
        if cfg.persistent_dim > 0:
            self.persistent = nn.Parameter(
                torch.zeros(cfg.persistent_dim).normal_(std=cfg.init_scale)
            )
        else:
            self.persistent = None

        # Learnable gates for theta_t (inner_lr), eta_t (momentum) and
        # alpha_t (forget rate).  Each gate maps a key into a scalar in
        # (0, 1) (after a sigmoid), then is scaled by the static config
        # value.  This matches the data-dependent gates in the paper.
        if cfg.learnable_gates:
            self.lr_gate = nn.Linear(cfg.key_dim, 1)
            self.momentum_gate = nn.Linear(cfg.key_dim, 1)
            self.forget_gate = nn.Linear(cfg.key_dim, 1)
            for g in (self.lr_gate, self.momentum_gate, self.forget_gate):
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)
        else:
            self.lr_gate = None
            self.momentum_gate = None
            self.forget_gate = None

    # ------------------------------------------------------------------
    # Retention surrogate.  Overridden in MirasMemory.
    # ------------------------------------------------------------------
    def retention_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L2 retention surrogate (Titans).

        Mean is taken over the value dim and summed over the tokens in the
        chunk; that way the per-element gradient scale is independent of
        ``value_dim`` but additional tokens in the chunk contribute
        additively (as in the original Titans chunk-wise gradient).
        """
        return 0.5 * (pred - target).pow(2).mean(dim=-1).sum()

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _rms_norm(x: torch.Tensor) -> torch.Tensor:
        """RMS-normalize over the feature dim (unit mean-square).

        Keeps the online update bounded and well-conditioned regardless of the
        caller's input scale.  Real LM activations can be O(1e4-1e5), which
        otherwise makes the surprise gradient explode to NaN within a couple of
        chunks.
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------
    def _gates(self, keys_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute (lr, momentum, forget) for the chunk.

        Each comes back as a scalar tensor (mean across the chunk and
        batch).  Keeping the gates scalar lets us reuse ``functional_call``
        without per-token parameter explosions.
        """
        cfg = self.cfg
        if not cfg.learnable_gates:
            device, dtype = keys_chunk.device, keys_chunk.dtype
            return (
                torch.tensor(cfg.inner_lr, device=device, dtype=dtype),
                torch.tensor(cfg.momentum, device=device, dtype=dtype),
                torch.tensor(cfg.forget_rate, device=device, dtype=dtype),
            )
        # Mean-pool over the chunk so the gate output is a scalar.
        pooled = keys_chunk.mean(dim=(0, 1)) if keys_chunk.dim() == 3 else keys_chunk.mean(0)
        # The gate Linears live in the module's dtype, but callers (e.g. the HF
        # wrapper) may upcast keys to float32 for numerical stability.  Run the
        # gate in its own dtype, then cast the scalar back, so mixed-precision
        # pipelines don't hit a `mat1 and mat2 must have the same dtype` error.
        in_dtype = keys_chunk.dtype
        pooled = pooled.to(self.lr_gate.weight.dtype)
        lr = (torch.sigmoid(self.lr_gate(pooled)).squeeze() * cfg.inner_lr * 2).to(in_dtype)
        mom = (torch.sigmoid(self.momentum_gate(pooled)).squeeze() * cfg.momentum).to(in_dtype)
        fgt = (torch.sigmoid(self.forget_gate(pooled)).squeeze() * cfg.forget_rate * 2).to(in_dtype)
        return lr, mom, fgt

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------
    def init_state(
        self, batch_size: int, device=None, dtype=None, detach: bool | None = None
    ) -> dict[str, Any]:
        """Initialise per-sequence memory + momentum state.

        ``detach`` controls whether the initial parameters are detached from
        ``self.memory``.  When ``None`` (default) it resolves to
        ``not self.training`` so that **meta-training** (``module.train()``)
        lets the outer-loop gradient flow into the initial memory parameters,
        while **inference** (``module.eval()``) detaches for a clean,
        graph-free per-sequence reset.  Detaching unconditionally — as the
        original code did — silently zeroed the gradient to the memory MLP, so
        only the gate heads ever trained.
        """
        if detach is None:
            detach = not self.training
        params = {}
        for name, p in self.memory.named_parameters():
            base = p.detach() if detach else p
            params[name] = base.unsqueeze(0).expand(batch_size, *base.shape).contiguous()
        momentum = {name: torch.zeros_like(p) for name, p in params.items()}
        return {"params": params, "momentum": momentum}

    def _memory_apply(self, params_b: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Apply the memory MLP using per-batch parameters via functional_call."""
        return functional_call(self.memory, params_b, (x,))

    def _chunk_loss(
        self,
        params_b: dict[str, torch.Tensor],
        keys_chunk: torch.Tensor,
        values_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self._memory_apply(params_b, keys_chunk)
        return self.retention_loss(pred, values_chunk)

    def _update_one(
        self,
        params_b: dict[str, torch.Tensor],
        momentum_b: dict[str, torch.Tensor],
        keys_chunk: torch.Tensor,
        values_chunk: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """One online update for one element of the batch."""
        lr, mom, fgt = self._gates(keys_chunk)
        grad_fn = grad(self._chunk_loss, argnums=0)
        grads = grad_fn(params_b, keys_chunk, values_chunk)
        new_momentum = {name: mom * momentum_b[name] - lr * grads[name] for name in params_b}
        new_params = {name: (1.0 - fgt) * params_b[name] + new_momentum[name] for name in params_b}
        return new_params, new_momentum

    # ------------------------------------------------------------------
    # Forward variants
    # ------------------------------------------------------------------
    def read(self, queries: torch.Tensor, state: dict[str, Any] | None = None) -> torch.Tensor:
        """Read from the memory at the current state without updating it."""
        if self.cfg.normalize_qk:
            queries = self._rms_norm(queries)
        if state is None:
            state = self.init_state(queries.shape[0], queries.device, queries.dtype)
        outs = [
            self._memory_apply({k: v[b] for k, v in state["params"].items()}, queries[b])
            for b in range(queries.shape[0])
        ]
        return torch.stack(outs, dim=0)

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stream ``(keys, values)`` into the memory and return the new state."""
        if self.cfg.normalize_qk:
            keys = self._rms_norm(keys)
        if self.cfg.normalize_v:
            values = self._rms_norm(values)
        if state is None:
            state = self.init_state(keys.shape[0], keys.device, keys.dtype)
        batch_size, seq_len, _ = keys.shape
        chunk = self.cfg.chunk_size
        new_params = state["params"]
        new_momentum = state["momentum"]
        for b in range(batch_size):
            pb = {name: new_params[name][b] for name in new_params}
            mb = {name: new_momentum[name][b] for name in new_momentum}
            for start in range(0, seq_len, chunk):
                end = min(start + chunk, seq_len)
                pb, mb = self._update_one(pb, mb, keys[b, start:end], values[b, start:end])
            for name in new_params:
                new_params[name] = new_params[name].clone()
                new_params[name][b] = pb[name]
                new_momentum[name] = new_momentum[name].clone()
                new_momentum[name][b] = mb[name]
        return {"params": new_params, "momentum": new_momentum}

    def read_write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        state: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Write ``(keys, values)`` then read at ``queries``."""
        new_state = self.write(keys, values, state)
        return self.read(queries, new_state), new_state

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor | None = None,
        state: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Default forward = read-then-write-then-read.

        Returns ``(memory_output, new_state)`` where ``memory_output`` is
        ``M_new(queries)``.  If ``queries`` is None, queries default to keys
        so the returned tensor is the memory's reconstruction of the values.
        """
        if queries is None:
            queries = keys
        return self.read_write(keys, values, queries, state)
