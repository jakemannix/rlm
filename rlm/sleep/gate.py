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
    # Environment feedback is "tool" steps, plus user turns *after* the
    # first: chat-style agent datasets (e.g. AgentInstruct's human/gpt
    # alternation) deliver observations as user turns, and the opening
    # user turn is the task instruction, not feedback.
    first_user = next((i for i, s in enumerate(episode.steps) if s.role == "user"), -1)
    observation_steps = [
        s
        for i, s in enumerate(episode.steps)
        if s.role == "tool" or (s.role == "user" and i > first_user)
    ]
    assistant_steps = [s for s in episode.steps if s.role == "assistant"]

    n_errors = sum(1 for s in observation_steps if ERROR_PATTERN.search(s.content))
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
    """Raw mean NLL of the gold response under the current policy model.

    Returns nats/token, *not* a squashed score: absolute NLL scales differ
    wildly across datasets and model sizes (measured: an absolute /4.0
    squash saturated at 1.0 on every AgentInstruct/os episode under a 0.5B
    policy). ``select_for_reflection`` min-max normalizes within the day,
    so "surprising" always means *relative to today's batch*.
    """

    def __init__(self, model, tokenizer, device: str = "cuda", max_tokens: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens

    def score(self, episode: Episode) -> float:
        import torch

        from rlm.sleep.evals import response_nll

        with torch.no_grad():
            return response_nll(
                self.model,
                self.tokenizer,
                prompt_messages=episode.prompt_messages(),
                response=episode.final_response(),
                device=self.device,
                max_seq_len=self.max_tokens,
            )


def select_for_reflection(
    episodes: list[Episode],
    config: GateConfig,
    surprise_gate: SurpriseGate | None = None,
) -> list[GateDecision]:
    """Score every episode; select the top ``budget_fraction`` above ``min_score``.

    When a surprise gate is provided, its raw NLLs are min-max normalized
    *within the day* to [0, 1] (ties stay tied, ordering is preserved) and
    averaged with the heuristic score.
    """
    heuristic = HeuristicGate(config)
    scored = []
    for ep in episodes:
        score, signals = heuristic.score(ep)
        if surprise_gate is not None:
            signals["surprise_nll"] = surprise_gate.score(ep)
        scored.append((ep, score, signals))

    if surprise_gate is not None and scored:
        nlls = [signals["surprise_nll"] for _, _, signals in scored]
        lo, hi = min(nlls), max(nlls)
        for _, _, signals in scored:
            signals["surprise"] = (signals["surprise_nll"] - lo) / (hi - lo) if hi > lo else 0.5

    decisions = []
    for ep, score, signals in scored:
        if surprise_gate is not None:
            score = (score + signals["surprise"]) / 2.0
        decisions.append(
            GateDecision(episode_id=ep.episode_id, score=score, selected=False, signals=signals)
        )

    budget = max(1, int(len(episodes) * config.budget_fraction))
    ranked = sorted(decisions, key=lambda d: d.score, reverse=True)
    for decision in ranked[:budget]:
        if decision.score >= config.min_score:
            decision.selected = True
    return decisions
