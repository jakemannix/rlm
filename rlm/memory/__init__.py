"""
Titans / Miras parametric memory modules for RLM.

The classes in this package implement neural long-term memory modules that
learn at test time via online gradient updates on a small "memory MLP".

- ``TitansMemory``: the Titans formulation (Behrouz et al., 2025) with
  momentum surprise and adaptive forgetting.
- ``MirasMemory``: the Miras (Behrouz et al., 2025) generalisation, which
  exposes the retention/attention bias as a pluggable surrogate loss
  (``"l2"``, ``"l1"``, ``"huber"``, ``"kl"`` ...).
- ``TitansAugmentedLM``: a thin wrapper that splices a Titans memory module
  into the embedding stream of a Hugging Face causal LM
  (designed for Gemma but works with any HF causal model).
- ``MemoryConfig``: hyper-parameter dataclass shared by the modules.

Torch and transformers are optional dependencies; importing this package
without ``torch`` installed only fails when the offending class is used.
"""

from rlm.memory.config import MemoryConfig

__all__ = [
    "MemoryConfig",
    "TitansMemory",
    "MirasMemory",
    "TitansAugmentedLM",
]


def __getattr__(name: str):
    """Lazy attribute access so optional torch deps only load on demand."""
    if name == "TitansMemory":
        from rlm.memory.titans import TitansMemory

        return TitansMemory
    if name == "MirasMemory":
        from rlm.memory.miras import MirasMemory

        return MirasMemory
    if name == "TitansAugmentedLM":
        from rlm.memory.modeling import TitansAugmentedLM

        return TitansAugmentedLM
    raise AttributeError(f"module 'rlm.memory' has no attribute {name!r}")
