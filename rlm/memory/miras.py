"""
Miras (Behrouz et al., 2025) generalisation of Titans.

The Miras paper observes that many modern attention / state-space layers
can be viewed as approximate solutions to the inner-loop problem::

    min_M  L_retention( M(K), V )

where ``L_retention`` is some convex surrogate.  Different surrogates give
different update rules:

* ``l2``     - vanilla Titans (associative L2 loss + SGD-momentum).
* ``l1``     - sign-SGD style updates, similar to gated SSMs.
* ``huber``  - smooth interpolation between l1 and l2.
* ``kl``     - Bregman / softmax-style retention; the values are treated
  as a categorical distribution and the memory predicts log-probs.

This module reuses :class:`TitansMemory` and only swaps the retention
surrogate.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover - torch is an optional dep
    raise ImportError(
        "rlm.memory.miras requires torch.  Install with `pip install 'rlms[titans]'`."
    ) from e

from rlm.memory.config import MemoryConfig
from rlm.memory.titans import TitansMemory


class MirasMemory(TitansMemory):
    """Pluggable-retention parametric memory.  See module docstring."""

    def __init__(self, cfg: MemoryConfig):
        super().__init__(cfg)
        self.retention = cfg.retention
        self.huber_delta = cfg.huber_delta

    def retention_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Retention surrogate, value-dim-normalised and chunk-summed."""
        if self.retention == "l2":
            return 0.5 * (pred - target).pow(2).mean(dim=-1).sum()
        if self.retention == "l1":
            return (pred - target).abs().mean(dim=-1).sum()
        if self.retention == "huber":
            per = F.huber_loss(pred, target, reduction="none", delta=self.huber_delta)
            return per.mean(dim=-1).sum()
        if self.retention == "kl":
            # Treat ``target`` as a categorical distribution along the last
            # dim; the memory predicts unnormalised log-probabilities.
            log_pred = F.log_softmax(pred, dim=-1)
            tgt = F.softmax(target, dim=-1)
            per = (tgt * (tgt.clamp_min(1e-12).log() - log_pred)).sum(dim=-1)
            return per.sum()
        raise ValueError(f"Unknown retention loss: {self.retention!r}")
