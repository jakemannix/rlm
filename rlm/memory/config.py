"""Hyper-parameter config shared by Titans and Miras memory modules."""

from dataclasses import dataclass, field
from typing import Literal

RetentionLoss = Literal["l2", "l1", "huber", "kl"]


@dataclass
class MemoryConfig:
    """
    Configuration for a parametric (Titans / Miras) memory module.

    Notation follows the Titans paper (Behrouz et al., 2025):
        S_t = eta_t * S_{t-1} - theta_t * grad_M  L(M; k_t, v_t)
        M_t = (1 - alpha_t) * M_{t-1} + S_t

    where ``alpha_t`` is the input-dependent forget gate, ``eta_t`` the
    momentum coefficient, and ``theta_t`` the inner learning rate.
    """

    # --- Geometry --------------------------------------------------------
    # Dimensionality of keys / values the memory operates on.
    key_dim: int = 64
    value_dim: int = 64
    # Hidden width of the memory MLP itself.
    hidden_dim: int = 128
    # Number of hidden layers in the memory MLP (>=1).
    n_layers: int = 2
    # Width of the carry / persistent token (concatenated with reads).
    persistent_dim: int = 0

    # --- Online update -------------------------------------------------
    # Inner learning rate (theta_t) when no learned gate is used.
    inner_lr: float = 1.0e-2
    # Momentum coefficient (eta_t) when no learned gate is used.
    momentum: float = 0.9
    # Forget rate (alpha_t in [0, 1]) when no learned gate is used.
    forget_rate: float = 1.0e-3
    # If True, theta_t / eta_t / alpha_t are produced by small linear gates
    # from the current input — matching the input-dependent variant in the
    # Titans paper.
    learnable_gates: bool = True
    # Number of tokens accumulated per memory update.  ``chunk_size=1``
    # gives the pure online recurrence; larger chunks are faster and use
    # the chunk-wise gradient from the paper.
    chunk_size: int = 1
    # RMS-normalize keys/queries (``normalize_qk``) and values (``normalize_v``)
    # before the online update.  Essential on real LMs: their residual-stream
    # activations are O(1e4-1e5) ("massive activations"), which makes the
    # unnormalized surprise gradient diverge to NaN.  Empirically, QK-norm is
    # what conditions recall, and V-norm is a small additional gain (values in
    # the normalized residual space already have ~uniform magnitude).  Turn
    # ``normalize_v`` off only if your values carry information in their norm.
    normalize_qk: bool = True
    normalize_v: bool = True

    # --- Retention surrogate (Miras) -----------------------------------
    # Surrogate loss used to compute the "surprise" signal.  "l2" recovers
    # vanilla Titans; "l1" / "huber" / "kl" instantiate the Miras
    # generalisation.
    retention: RetentionLoss = "l2"
    # Threshold used by the Huber retention loss.
    huber_delta: float = 1.0

    # --- Activations -----------------------------------------------------
    activation: Literal["gelu", "silu", "relu", "tanh"] = "gelu"

    # --- Initialisation -------------------------------------------------
    init_scale: float = 0.02

    # --- Misc -----------------------------------------------------------
    # When integrating with an HF model, how many "memory tokens" worth of
    # information to splice into the embedding stream per forward pass.
    n_memory_tokens: int = 8
    # Names of additional fields downstream code may stash on the config.
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("MemoryConfig.n_layers must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("MemoryConfig.chunk_size must be >= 1")
        if not (0.0 <= self.forget_rate <= 1.0):
            raise ValueError("MemoryConfig.forget_rate must lie in [0, 1]")
        if self.retention not in ("l2", "l1", "huber", "kl"):
            raise ValueError(f"Unknown retention loss: {self.retention!r}")
