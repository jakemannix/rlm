"""Configuration for the sleep-time consolidation pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class GateConfig:
    """Stage 1 — the cheap "worth reflecting on?" filter.

    ``budget_fraction`` is the economic knob: at most this fraction of a
    day's episodes is forwarded to the (expensive) judge.  ``min_score``
    is a floor so quiet days don't promote noise just to fill the budget.
    """

    budget_fraction: float = 0.25
    min_score: float = 0.15
    # Feature weights for the heuristic gate.
    weight_error: float = 0.5
    weight_retry: float = 0.3
    weight_failure: float = 0.4
    weight_length: float = 0.1
    # Optional NLL surprise scorer (requires the ``sleep`` extra + a GPU to
    # be fast).  When enabled, the surprise score is averaged with the
    # heuristic score.
    use_surprise: bool = False
    surprise_max_tokens: int = 512

    def __post_init__(self) -> None:
        if not 0.0 < self.budget_fraction <= 1.0:
            raise ValueError("GateConfig.budget_fraction must lie in (0, 1]")


@dataclass
class JudgeConfig:
    """Stage 2 — high-test-time-compute reflection via LLM-as-a-Judge."""

    # Independent reflection samples per episode (self-consistency).
    n_samples: int = 1
    # Majority "learn" verdicts required across samples to keep an episode.
    min_confidence: float = 0.5
    # Generation knobs for a local self-judge (LocalHFJudge): sampling
    # diversity feeds the self-consistency vote; reflection gets a longer
    # leash than ordinary policy decoding.
    temperature: float = 0.7
    max_new_tokens: int = 768
    # Run the cheap verification pass over each distilled example
    # (generation-verification gap: only verified examples earn a gradient).
    verify_examples: bool = True
    max_examples_per_episode: int = 3
    max_transcript_chars: int = 6000


@dataclass
class AdapterConfig:
    """Stage 3 — LoRA SFT hyperparameters (the sweep surface)."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lr: float = 2.0e-4
    epochs: int = 2
    batch_size: int = 4
    grad_accum: int = 1
    max_seq_len: int = 1024
    warmup_steps: int = 5
    # Fraction of the SFT batch drawn from a general replay pool to guard
    # against forgetting (see docs/learning_signal.md, "verification gate").
    replay_ratio: float = 0.2
    seed: int = 0


@dataclass
class SleepConfig:
    """Top-level config for one nightly cycle (and the sweep grid)."""

    policy_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    episodes_per_day: int = 64
    system_prompt: str = "You are a careful agent. Reason step by step before acting."
    gate: GateConfig = field(default_factory=GateConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    def with_overrides(self, overrides: dict[str, Any]) -> SleepConfig:
        """Return a deep copy with dotted-key overrides applied.

        e.g. ``{"adapter.lr": 1e-4, "gate.budget_fraction": 0.5}``
        """
        cfg = copy.deepcopy(self)
        for key, value in overrides.items():
            parts = key.split(".")
            target: Any = cfg
            for part in parts[:-1]:
                target = getattr(target, part)
            leaf = parts[-1]
            if not any(f.name == leaf for f in fields(target)):
                raise KeyError(f"Unknown config field: {key}")
            setattr(target, leaf, value)
        return cfg
