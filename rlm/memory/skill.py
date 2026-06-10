"""
The memory *skill*: everything meta-trained and then frozen at deployment.

Design doc §2.  The skill wraps a :class:`DeltaRuleStore` with:

- ``key_enc`` / ``query_enc`` — (small) encoders from the base's **post-final-norm**
  hidden states into key space.  Orthogonal init so keys are well-spread before any
  training.
- value pathway — values are the base's **embedding rows of the next token**
  (answer-shifted, design doc §0).  The skill never owns the embedding table; the
  caller passes ``e_next`` (trainer: from the cache; deployment: from the model).
- ``out_proj`` — value-space → model-space, identity init when square (with tied
  embeddings the read is already logit-shaped, so identity is the near-closed-form
  solution the meta-learner only needs to refine).
- ``read_gate`` — per-token scalar g ∈ (0,1) on [q; read]: the explicit *abstain*
  mechanism (A2).  Trained down by the KL-to-base term on irrelevant positions,
  up by CE on recall positions.

The same module is used by the cached-activation trainer **and** the live
deployment session, which is the structural fix for failure mode §5.8
(train/deploy skew in the inner loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from rlm.memory.linear_store import DeltaRuleStore, LinearStoreConfig, StoreState


@dataclass
class SkillConfig:
    d_model: int = 1152  # Gemma-3-1B hidden/embedding dim (verify on the config)
    d_k: int = 512
    # 1 ⇒ Linear encoders; ≥2 ⇒ MLP with GELU (the paraphrase-robustness ladder,
    # design doc §1/Q3 — depth lives *outside* the frozen base).
    encoder_layers: int = 1
    encoder_hidden: int = 1024
    tie_kq: bool = False
    # Key whitening (ZCA fitted on the cached addressing population, see
    # MemorySkill.set_whitening): equalises the key-input spectrum so the small
    # entity-discriminative components carry addressing instead of the dominant
    # shared template/position directions.  The capacity lever from the design
    # review (docs/design_review_capacity.md §2): raw template-clustered keys
    # collapse by m≈16–32 regardless of d_k or head count; whitened keys hold 128.
    whiten: bool = False
    read_gate_bias_init: float = -2.0  # g₀ ≈ 0.12: small but alive
    # Multi-head addressing (design doc §1/Q1).  n_heads>1 + shared_encoder=False ⇒
    # H *independent* encoders (d_model→d_k/H) feeding H per-head-normed stores: the
    # capacity bet (more independent key directions).  shared_encoder=True is the A/B
    # control — one encoder reshaped into heads, predicted ≈ no-op for a linear store.
    n_heads: int = 1
    shared_encoder: bool = False
    store: LinearStoreConfig = field(default_factory=LinearStoreConfig)

    def __post_init__(self) -> None:
        # Values live in embedding space: d_v is pinned to d_model.
        self.store.d_k = self.d_k
        self.store.d_v = self.d_model
        self.store.n_heads = self.n_heads
        if self.d_k % self.n_heads != 0:
            raise ValueError(f"d_k={self.d_k} must be divisible by n_heads={self.n_heads}")


def _make_one(cfg: SkillConfig, out_dim: int) -> nn.Module:
    """One encoder d_model→out_dim: Linear (encoder_layers=1) or GELU-MLP (≥2)."""
    if cfg.encoder_layers == 1:
        enc = nn.Linear(cfg.d_model, out_dim, bias=False)
        nn.init.orthogonal_(enc.weight)
        return enc
    layers: list[nn.Module] = []
    dims = [cfg.d_model] + [cfg.encoder_hidden] * (cfg.encoder_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1], bias=True)
        nn.init.orthogonal_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class _MultiHeadEncoder(nn.Module):
    """``n_heads`` independent encoders d_model→d_k/n_heads, concatenated to d_k, so
    each head addresses its own subspace (the design doc §1/Q1 capacity bet)."""

    def __init__(self, cfg: SkillConfig):
        super().__init__()
        d_h = cfg.d_k // cfg.n_heads
        self.heads = nn.ModuleList([_make_one(cfg, d_h) for _ in range(cfg.n_heads)])

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([h(x) for h in self.heads], dim=-1)


def make_encoder(cfg: SkillConfig) -> nn.Module:
    # n_heads=1 or shared_encoder ⇒ a single encoder → d_k (byte-identical to the
    # single-head path).  Independent heads are what can raise effective key dim.
    if cfg.n_heads == 1 or cfg.shared_encoder:
        return _make_one(cfg, cfg.d_k)
    return _MultiHeadEncoder(cfg)


class MemorySkill(nn.Module):
    """Bind (post-norm hidden @ t) → (embedding of token t+1); retrieve as a
    post-norm injection delta."""

    def __init__(self, cfg: SkillConfig):
        super().__init__()
        self.cfg = cfg
        self.store = DeltaRuleStore(cfg.store)

        self.key_enc = make_encoder(cfg)
        self.query_enc = self.key_enc if cfg.tie_kq else make_encoder(cfg)

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        with torch.no_grad():
            self.out_proj.weight.copy_(torch.eye(cfg.d_model))

        # Gate features: [hn, read, ‖read‖_rms].  The read-magnitude feature lets
        # the gate abstain when the store returns a weak/spurious read (key
        # interference on an unrelated query) vs a strong matched recall — the
        # branch table's "per-token read gate features" fix for A2 (raising
        # lambda_kl alone over-suppresses and breaks A1's wrong-fact control).
        # Whitening buffers (identity until set_whitening): applied to encoder
        # inputs only — the read gate still sees the raw post-norm hidden.
        self.register_buffer("whiten_mean", torch.zeros(cfg.d_model))
        self.register_buffer("whiten_T", torch.eye(cfg.d_model))

        self.read_gate = nn.Linear(2 * cfg.d_model + 1, 1)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, cfg.read_gate_bias_init)

    # ------------------------------------------------------------------
    def set_whitening(self, mean: Tensor, transform: Tensor) -> None:
        """Install a fitted ZCA (see :func:`fit_zca`); persists via state_dict."""
        if mean.shape != (self.cfg.d_model,) or transform.shape != (self.cfg.d_model, self.cfg.d_model):
            raise ValueError(f"whitening shapes {mean.shape}/{transform.shape} != d_model {self.cfg.d_model}")
        self.whiten_mean.copy_(mean.to(self.whiten_mean))
        self.whiten_T.copy_(transform.to(self.whiten_T))
        self.cfg.whiten = True

    def whiten(self, hn: Tensor) -> Tensor:
        if not self.cfg.whiten:
            return hn
        return (hn - self.whiten_mean) @ self.whiten_T.T

    def init_state(self, batch_size: int, device=None) -> StoreState:
        return self.store.init_state(batch_size, device=device)

    def write(
        self,
        hn: Tensor,  # [B, T, d_model] post-final-norm hidden states (ingest)
        e_next: Tensor,  # [B, T, d_model] embedding rows of token t+1
        state: StoreState,
        write_mask: Tensor | None = None,  # [B, T]
        stats: dict | None = None,  # optional out-dict: write-side gate means
    ) -> StoreState:
        keys = self.key_enc(self.whiten(hn))
        return self.store.write(keys, e_next, state, write_mask=write_mask, stats=stats)

    def read_delta(
        self,
        hn: Tensor,  # [B, T, d_model] post-final-norm hidden states (recall)
        state: StoreState,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the additive post-norm injection ``delta`` and diagnostics.

        head_input = hn + delta ;  delta = g · out_proj(M q).
        Empty store ⇒ read = 0 ⇒ delta ≡ 0 (exact no-op).
        """
        q = self.query_enc(self.whiten(hn))
        read = self.store.read(q, state)  # [B, T, d_model]
        # ‖read‖ via norm() (overflow-safe — pow(2) overflows fp32 on the large
        # reads an untrained store produces on long multisession ingests, and the
        # zero-init gate weight would then do 0·inf = NaN), compressed with log1p
        # so an occasional huge read can't saturate the gate during training.
        read_rms = read.norm(dim=-1, keepdim=True) * (read.shape[-1] ** -0.5)  # [B, T, 1]
        g = torch.sigmoid(self.read_gate(torch.cat([hn, read, torch.log1p(read_rms)], dim=-1)))
        delta = g * self.out_proj(read)
        aux = {
            "read_gate": g.squeeze(-1).detach(),
            "read_rms": read.detach().pow(2).mean(dim=-1).sqrt(),
        }
        return delta, aux

    # ------------------------------------------------------------------
    # Persistence of the *skill* (the store state has its own A4 path).
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"cfg": self.cfg, "state_dict": self.state_dict()}, path)

    @staticmethod
    def load(path: str, device=None) -> MemorySkill:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        skill = MemorySkill(payload["cfg"])
        skill.load_state_dict(payload["state_dict"])
        if device is not None:
            skill.to(device)
        return skill


def fit_zca(x: Tensor, eps_frac: float = 1e-3) -> tuple[Tensor, Tensor]:
    """Fit a shrinkage-regularised ZCA whitening transform on rows of ``x``.

    Returns ``(mean, T)`` with ``T = U diag((S + ε)^{-1/2}) Uᵀ`` for the
    eigendecomposition of the (shrunk) covariance, ε = eps_frac · mean(S).
    Apply as ``(x − mean) @ T.T``.  ``eps_frac`` trades how aggressively the
    small (entity-carrying) directions are amplified against amplifying noise —
    sweep {1e-2, 1e-3, 1e-4} if queries are noisy (paraphrase regime).
    """
    if x.dim() != 2 or x.shape[0] < 2:
        raise ValueError(f"fit_zca expects [N>=2, d], got {tuple(x.shape)}")
    x = x.to(torch.float32)
    mean = x.mean(dim=0)
    xc = x - mean
    cov = xc.T @ xc / (x.shape[0] - 1)
    S, U = torch.linalg.eigh(cov)
    S = S.clamp(min=0.0)
    eps = eps_frac * S.mean()
    T = U @ torch.diag((S + eps).rsqrt()) @ U.T
    return mean, T
