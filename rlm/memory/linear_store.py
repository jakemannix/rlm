"""
Zero-init linear fast-weight store with an analytic delta-rule update.

This is the "decoupled store" arm from ``docs/design_consult_response.md`` §1/Q1:
all *skill* lives outside (key/query encoders, gates, readout); the *content* is a
single mutable matrix ``M ∈ R^{d_v × d_k}`` initialised to **zero**, so there is no
init-vs-content competition at all (θ₀ ≡ 0) and an empty store reads back exactly 0.

The Titans surprise update with an L2 retention loss on a linear map *is* the delta
rule (Widrow–Hoff / DeltaNet with momentum and decay)::

    err_t = M_{c} k_t − v_t                       # at start-of-chunk M (chunk-wise)
    G_c   = Σ_t θ_t · err_t k_tᵀ / d_k            # per-token gated, analytic gradient
    S_c   = η_c · S_{c−1} − G_c                   # momentum
    M_c   = (1 − α_c) · M_{c−1} + S_c             # forget + write

so the inner loop needs no autograd at all: it is a few GEMMs, fully batched, and
differentiable w.r.t. the gates / encoders by ordinary BPTT through the chunk
sequence.

Scaling convention (deliberate deviation from ``titans.py``, see design doc §5.1):
keys/queries are RMS-normalised (⇒ ‖k‖² = d_k) and the gradient carries a 1/d_k, so
**lr θ = 1 writes a fresh association exactly in one shot** (for a key orthogonal to
existing content):  M ← M + θ·v kᵀ/d_k  ⇒  M k = θ·v.  The default gate range is
(0, 2) with init exactly 1.0, because one-shot episodic writes are the regime this
store is for; ``titans.py``'s 1e-2 default cannot write a single-occurrence fact.

Gates (all start near the one-shot-exact regime; training must *earn* drift):
- ``lr``      — per *token* (this is the selectivity mechanism: write the fact,
                skip the filler), sigmoid(W k_t + b) · max_lr, zero-init ⇒ θ₀ = 1.
- ``momentum``— per chunk scalar, pooled over the chunk's keys.  NB heavy-ball
                momentum deposits each gradient with total weight 1/(1−η) over
                subsequent chunks; init is ≈0 so writes are one-shot exact.
- ``forget``  — per chunk scalar.  Forgetting **compounds across chunks**: α per
                chunk over C chunks retains (1−α)^C, so the gate is biased strongly
                negative at init (α₀ ≈ 1e-3 · max_forget-ish) and training must
                *earn* forgetting.  Cross-session persistence dies here otherwise.

Delta-rule physics worth knowing (encoded in tests/test_linear_store.py):
error-correcting writes *erode* existing content along overlapping key
directions — m unselective writes at lr θ retain ≈ exp(−m·θ/d_k) of an earlier
association.  Read-side interference is what the delta rule buys down; the price
is write-side erosion, and the per-token lr gate's selectivity (θ→0 on filler) is
therefore load-bearing for cross-session persistence, not a nicety.  The
``multisession`` episode type exists to train exactly this.

Serialization (acceptance test A4): ``state_dict_for_persistence`` /
``load_persisted_state`` round-trip **M only** — momentum is zeroed on load, since a
stale momentum direction from session N would contaminate the first writes of
session N+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """RMS-normalise the last dim to unit mean-square (so ‖x‖² = dim)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


@dataclass
class LinearStoreConfig:
    """Geometry + update-rule hyperparameters for :class:`DeltaRuleStore`.

    Kept separate from :class:`rlm.memory.config.MemoryConfig` on purpose: that
    config's ``init_scale`` is shared with the HF wrapper's projection init, and a
    zero-init store must not zero-init the projections (design doc §5.6).
    """

    d_k: int = 512
    d_v: int = 1152  # default: Gemma-3-1B embedding dim (values live in e-space)

    # Gate ceilings.  lr range is (0, max_lr) with init exactly max_lr/2; the
    # default ceiling of 2.0 puts the one-shot-write point θ=1 at the init.
    max_lr: float = 2.0
    max_momentum: float = 0.95
    # sigmoid(momentum_bias_init) · max_momentum = η at init.  Heavy-ball momentum
    # re-deposits each gradient into M with total weight 1/(1−η) across later
    # chunks (geometric series), so η must start ≈0 for one-shot writes to be
    # *exact* — training can raise it and will see the deposit factor in the loss.
    momentum_bias_init: float = -4.0
    max_forget: float = 0.1
    # sigmoid(forget_bias_init) · max_forget = α at init.  −5 ⇒ α₀ ≈ 6.7e-4.
    forget_bias_init: float = -5.0

    # Tokens per chunk-wise update during *writes*.  Small for one-shot facts.
    chunk_size: int = 4

    normalize_v: bool = True


class StoreState:
    """Plain-tensor state: ``M`` [B, d_v, d_k] and momentum ``S`` (same shape).

    Not an ``nn.Module`` — this is content, not skill.  ``detach()`` returns a
    graph-free copy (deployment / across-episode boundaries); training keeps the
    graph so BPTT reaches the gates and encoders.
    """

    def __init__(self, M: Tensor, S: Tensor):
        if M.shape != S.shape or M.dim() != 3:
            raise ValueError(f"M/S must be [B, d_v, d_k] and match, got {M.shape} / {S.shape}")
        self.M = M
        self.S = S

    @property
    def batch_size(self) -> int:
        return self.M.shape[0]

    def detach(self) -> StoreState:
        return StoreState(self.M.detach(), self.S.detach())

    def clone(self) -> StoreState:
        return StoreState(self.M.clone(), self.S.clone())

    def zeroed(self) -> StoreState:
        """Empty store of the same shape — the A1 'empty memory' control."""
        return StoreState(torch.zeros_like(self.M), torch.zeros_like(self.S))


class DeltaRuleStore(nn.Module):
    """Linear fast-weight store with gated delta-rule writes.

    The module's *parameters* are only the three gate networks (skill, meta-trained
    then frozen).  The *content* is a :class:`StoreState` owned by the caller.
    """

    def __init__(self, cfg: LinearStoreConfig):
        super().__init__()
        self.cfg = cfg

        # Per-token lr gate on the (normalised) key.  Zero init ⇒ sigmoid(0)=0.5
        # ⇒ θ₀ = max_lr/2 = 1.0 with the default ceiling.
        self.lr_gate = nn.Linear(cfg.d_k, 1)
        nn.init.zeros_(self.lr_gate.weight)
        nn.init.zeros_(self.lr_gate.bias)

        # Per-chunk scalar gates (pooled over the chunk's keys).
        self.momentum_gate = nn.Linear(cfg.d_k, 1)
        nn.init.zeros_(self.momentum_gate.weight)
        nn.init.constant_(self.momentum_gate.bias, cfg.momentum_bias_init)

        self.forget_gate = nn.Linear(cfg.d_k, 1)
        nn.init.zeros_(self.forget_gate.weight)
        nn.init.constant_(self.forget_gate.bias, cfg.forget_bias_init)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def init_state(self, batch_size: int, device=None, dtype=torch.float32) -> StoreState:
        device = device if device is not None else self.lr_gate.weight.device
        shape = (batch_size, self.cfg.d_v, self.cfg.d_k)
        return StoreState(
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    # ------------------------------------------------------------------
    # Write (analytic delta rule, chunk-wise, batched)
    # ------------------------------------------------------------------
    def write(
        self,
        keys: Tensor,  # [B, T, d_k]  (un-normalised; normalised here)
        values: Tensor,  # [B, T, d_v]
        state: StoreState,
        write_mask: Tensor | None = None,  # [B, T] in [0,1]; multiplies the lr gate
    ) -> StoreState:
        """Stream (keys, values) into the store; returns the new state.

        Differentiable w.r.t. gate parameters and w.r.t. ``keys``/``values`` (and
        hence the upstream encoders) via BPTT over chunks.  The inner loop itself
        is closed-form — no ``torch.func``.
        """
        cfg = self.cfg
        B, T, _ = keys.shape
        if state.batch_size != B:
            raise ValueError(f"state batch {state.batch_size} != keys batch {B}")

        k = rms_norm(keys)
        v = rms_norm(values) if cfg.normalize_v else values

        # Per-token learning rate (selectivity), optionally masked (padding /
        # fact-span-only writes).
        theta = torch.sigmoid(self.lr_gate(k)).squeeze(-1) * cfg.max_lr  # [B, T]
        if write_mask is not None:
            theta = theta * write_mask

        M, S = state.M, state.S
        for start in range(0, T, cfg.chunk_size):
            end = min(start + cfg.chunk_size, T)
            kc = k[:, start:end]  # [B, c, d_k]
            vc = v[:, start:end]  # [B, c, d_v]
            tc = theta[:, start:end]  # [B, c]

            # Chunk-wise: every token in the chunk sees start-of-chunk M.
            pred = torch.einsum("bvk,bck->bcv", M, kc)
            err = pred - vc  # [B, c, d_v]
            # Gated, 1/d_k-scaled gradient: θ=1 ⇒ exact one-shot write.
            G = torch.einsum("bcv,bck->bvk", err * tc.unsqueeze(-1), kc) / cfg.d_k

            pooled = kc.mean(dim=1)  # [B, d_k]
            eta = torch.sigmoid(self.momentum_gate(pooled)) * cfg.max_momentum  # [B,1]
            alpha = torch.sigmoid(self.forget_gate(pooled)) * cfg.max_forget  # [B,1]

            S = eta.unsqueeze(-1) * S - G
            M = (1.0 - alpha).unsqueeze(-1) * M + S

        return StoreState(M, S)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read(self, queries: Tensor, state: StoreState) -> Tensor:
        """``M q`` for RMS-normalised queries.  [B, T, d_k] → [B, T, d_v].

        An empty store returns exactly zero — the natural 'abstain' (design doc
        §2): downstream injection of out_proj(0)·g is a no-op, unlike an MLP
        store, which returns *something* for every query.
        """
        q = rms_norm(queries)
        return torch.einsum("bvk,btk->btv", state.M, q)

    # ------------------------------------------------------------------
    # Persistence (acceptance test A4)
    # ------------------------------------------------------------------
    @staticmethod
    def state_dict_for_persistence(state: StoreState) -> dict:
        """M only, fp32.  Momentum is deliberately not persisted (§5.5)."""
        return {"format": 1, "M": state.M.detach().to(torch.float32).cpu()}

    def load_persisted_state(self, payload: dict, device=None) -> StoreState:
        if payload.get("format") != 1:
            raise ValueError(f"unknown store payload format: {payload.get('format')!r}")
        M = payload["M"].to(device=device or self.lr_gate.weight.device, dtype=torch.float32)
        if M.dim() != 3 or M.shape[1] != self.cfg.d_v or M.shape[2] != self.cfg.d_k:
            raise ValueError(f"persisted M shape {tuple(M.shape)} != cfg ({self.cfg.d_v},{self.cfg.d_k})")
        return StoreState(M, torch.zeros_like(M))

    def save_state(self, state: StoreState, path: str) -> None:
        torch.save(self.state_dict_for_persistence(state), path)

    def load_state(self, path: str, device=None) -> StoreState:
        return self.load_persisted_state(torch.load(path, map_location="cpu", weights_only=True), device)
