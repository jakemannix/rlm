"""Stage 1 — the cheap "is there anything to learn here?" gate.

Two scorers:

* :class:`HeuristicGate` — pure-Python features over the trace itself
  (tool errors, retries, failure, unusual length).  Free; runs anywhere.
* :class:`SurpriseGate` — mean NLL of the episode's assistant tokens under
  the current policy model: "hard to predict in advance" (RHO-style
  surprise, lifted from token to episode level).  Requires the ``sleep``
  extra and benefits from a GPU.

``select_for_reflection`` applies the budget: at most ``budget_fraction``
of the day's episodes — the highest-scoring ones above ``min_score`` —
are forwarded to the expensive judge.
"""

from __future__ import annotations

import re

from rlm.sleep.config import GateConfig
from rlm.sleep.types import Episode, GateDecision

ERROR_PATTERN = re.compile(r"\b(error|traceback|exception|failed|denied|timeout)\b", re.IGNORECASE)


def heuristic_signals(episode: Episode) -> dict[str, float]:
    """Raw per-episode features, each roughly in [0, 1]."""
    tool_steps = [s for s in episode.steps if s.role == "tool"]
    assistant_steps = [s for s in episode.steps if s.role == "assistant"]

    n_errors = sum(1 for s in tool_steps if ERROR_PATTERN.search(s.content))
    error_signal = min(1.0, n_errors / 2.0)

    n_retries = sum(
        1
        for a, b in zip(assistant_steps, assistant_steps[1:], strict=False)
        if a.content.strip() == b.content.strip()
    )
    retry_signal = min(1.0, float(n_retries))

    failure_signal = 1.0 if episode.outcome == "failure" else 0.0

    # Episodes much longer than a "clean" trajectory suggest struggle.
    length_signal = min(1.0, max(0.0, (len(episode.steps) - 6) / 12.0))

    return {
        "error": error_signal,
        "retry": retry_signal,
        "failure": failure_signal,
        "length": length_signal,
    }


class HeuristicGate:
    def __init__(self, config: GateConfig):
        self.config = config

    def score(self, episode: Episode) -> tuple[float, dict[str, float]]:
        signals = heuristic_signals(episode)
        cfg = self.config
        weighted = (
            cfg.weight_error * signals["error"]
            + cfg.weight_retry * signals["retry"]
            + cfg.weight_failure * signals["failure"]
            + cfg.weight_length * signals["length"]
        )
        return min(1.0, weighted), signals


class SurpriseGate:
    """Mean NLL of assistant tokens under the current policy model."""

    def __init__(self, model, tokenizer, device: str = "cuda", max_tokens: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens

    def score(self, episode: Episode) -> float:
        import torch

        from rlm.sleep.evals import response_nll

        with torch.no_grad():
            nll = response_nll(
                self.model,
                self.tokenizer,
                prompt_messages=episode.prompt_messages(),
                response=episode.final_response(),
                device=self.device,
                max_seq_len=self.max_tokens,
            )
        # Squash to [0, 1]: NLL of ~4 nats/token is already very surprising.
        return min(1.0, nll / 4.0)


def select_for_reflection(
    episodes: list[Episode],
    config: GateConfig,
    surprise_gate: SurpriseGate | None = None,
) -> list[GateDecision]:
    """Score every episode; select the top ``budget_fraction`` above ``min_score``."""
    heuristic = HeuristicGate(config)
    decisions = []
    for ep in episodes:
        score, signals = heuristic.score(ep)
        if surprise_gate is not None:
            surprise = surprise_gate.score(ep)
            signals["surprise"] = surprise
            score = (score + surprise) / 2.0
        decisions.append(
            GateDecision(episode_id=ep.episode_id, score=score, selected=False, signals=signals)
        )

    budget = max(1, int(len(episodes) * config.budget_fraction))
    ranked = sorted(decisions, key=lambda d: d.score, reverse=True)
    for decision in ranked[:budget]:
        if decision.score >= config.min_score:
            decision.selected = True
    return decisions
