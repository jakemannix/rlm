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

The decoupled v1 design (``docs/design_consult_response.md``) lives alongside:

- ``DeltaRuleStore`` / ``LinearStoreConfig`` / ``StoreState``: zero-init linear
  fast-weight store with the analytic delta-rule update (the *content*).
- ``MemorySkill`` / ``SkillConfig``: encoders, gates and readout (the *skill*),
  shared by the cached-activation trainer and live deployment.
- ``EpisodeTrainer`` / ``TrainerConfig``: meta-training from cached post-norm
  activations — the base model never enters the training loop.
- ``MemorySession``: live cross-session ingest/recall/persist on a frozen base.
- ``EpisodeGenerator``: synthetic fact/recall/abstain/control episodes.

Torch and transformers are optional dependencies; importing this package
without ``torch`` installed only fails when the offending class is used.
"""

from rlm.memory.config import MemoryConfig

__all__ = [
    "MemoryConfig",
    "TitansMemory",
    "MirasMemory",
    "TitansAugmentedLM",
    "DeltaRuleStore",
    "LinearStoreConfig",
    "StoreState",
    "MemorySkill",
    "SkillConfig",
    "EpisodeTrainer",
    "TrainerConfig",
    "MemorySession",
    "EpisodeGenerator",
]

_LAZY = {
    "DeltaRuleStore": ("rlm.memory.linear_store", "DeltaRuleStore"),
    "LinearStoreConfig": ("rlm.memory.linear_store", "LinearStoreConfig"),
    "StoreState": ("rlm.memory.linear_store", "StoreState"),
    "MemorySkill": ("rlm.memory.skill", "MemorySkill"),
    "SkillConfig": ("rlm.memory.skill", "SkillConfig"),
    "EpisodeTrainer": ("rlm.memory.trainer", "EpisodeTrainer"),
    "TrainerConfig": ("rlm.memory.trainer", "TrainerConfig"),
    "MemorySession": ("rlm.memory.session", "MemorySession"),
    "EpisodeGenerator": ("rlm.memory.episodes", "EpisodeGenerator"),
}


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
    if name in _LAZY:
        import importlib

        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'rlm.memory' has no attribute {name!r}")
