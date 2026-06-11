"""Shared dataclasses for the sleep-time consolidation pipeline.

The pipeline (see ``docs/learning_signal.md``) is:

    day of episodes -> gate (cheap filter) -> judge (high-TTC reflection)
        -> verified SFT examples -> LoRA train -> next-day / held-out eval
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Step:
    """One turn of an agentic episode."""

    role: Role
    content: str


@dataclass
class Episode:
    """A normalized agentic trajectory.

    ``outcome`` is "success" / "failure" when the source dataset records it,
    else "unknown".  ``meta`` carries source-specific extras (and, for
    synthetic traces, the planted ground truth used by gate tests).
    """

    episode_id: str
    source: str
    task: str
    steps: list[Step]
    outcome: Literal["success", "failure", "unknown"] = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def transcript(self, max_chars: int = 6000) -> str:
        """Render the episode as a plain-text transcript (tail-truncated)."""
        lines = [f"TASK: {self.task}"]
        lines += [f"[{s.role}] {s.content}" for s in self.steps]
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... transcript truncated ...]"
        return text

    def final_response(self) -> str:
        """The last assistant turn (gold target for NLL evals)."""
        for step in reversed(self.steps):
            if step.role == "assistant":
                return step.content
        raise ValueError(f"Episode {self.episode_id} has no assistant step")

    def prompt_messages(self) -> list[dict[str, str]]:
        """Chat messages up to (excluding) the final assistant turn."""
        last_assistant = max(i for i, s in enumerate(self.steps) if s.role == "assistant")
        messages = [{"role": "user", "content": f"TASK: {self.task}"}]
        for step in self.steps[:last_assistant]:
            role = "user" if step.role == "tool" else step.role
            messages.append({"role": role, "content": step.content})
        return messages


@dataclass
class GateDecision:
    """Stage-1 output: should this episode get expensive reflection?"""

    episode_id: str
    score: float
    selected: bool
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingExample:
    """One verified SFT example distilled by the judge."""

    prompt: str
    response: str
    lesson: str
    source_episode_id: str
    verified: bool = False

    def to_messages(self, system_prompt: str | None = None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.prompt})
        messages.append({"role": "assistant", "content": self.response})
        return messages


@dataclass
class JudgeOutput:
    """Stage-2 output: the judge's reflection over one flagged episode."""

    episode_id: str
    verdict: Literal["learn", "skip"]
    confidence: float
    lesson: str
    examples: list[TrainingExample] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)
    # Reflection samples whose output contained no parseable JSON (each
    # counts as a skip vote; tracked so judge quality is observable).
    n_parse_failures: int = 0


@dataclass
class EvalReport:
    """Eval-harness output for one adapter (or the base model)."""

    name: str
    next_day_nll: float
    test_nll: float
    retention_nll: float
    n_next_day: int
    n_test: int
    extras: dict[str, float] = field(default_factory=dict)


@dataclass
class DayResult:
    """Everything produced by one nightly consolidation cycle."""

    day_index: int
    n_episodes: int
    n_selected: int
    n_learn_verdicts: int
    n_examples: int
    adapter_dir: str | None
    base_report: EvalReport | None
    adapted_report: EvalReport | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
