"""Judged win-rate eval: base vs. adapted, blind pairwise A/B.

NLL is the cheap deterministic metric, but the question people actually
ask is "does the adapted model *answer better*?". This module generates
responses from both models on held-out prompts and has a judge pick a
winner without knowing which is which (presentation order is randomized
per item, seeded).

Like reflection, judging should use the frozen base (or a fixed external
judge) so the metric is stationary across nights.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from rlm.clients.base_lm import BaseLM
from rlm.sleep.types import Episode

PAIRWISE_PROMPT = """\
You are judging two assistant responses to the same task. Pick the response \
that is more correct, more complete, and more direct.

TASK:
{task}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Which response is better? Answer with exactly one word: A, B, or TIE.
"""


@dataclass
class WinRateReport:
    """Outcome counts from the adapted model's perspective."""

    wins: int = 0
    losses: int = 0
    ties: int = 0
    verdicts: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.5

    def to_dict(self) -> dict:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "n": self.n,
            "win_rate": round(self.win_rate, 4),
        }


def parse_verdict(text: str) -> str:
    """Map a judge response to "A", "B", or "tie" (unparseable counts as tie)."""
    word = text.strip().upper().split()[0] if text.strip() else ""
    word = word.strip(".,:;!\"'")
    if word in ("A", "B"):
        return word
    return "tie"


def judged_win_rate(
    judge_lm: BaseLM,
    items: list[tuple[str, str, str]],
    seed: int = 0,
) -> WinRateReport:
    """Blind pairwise judging over ``(task, base_response, adapted_response)`` items.

    Presentation order is randomized per item so the judge can't learn a
    positional shortcut; verdicts are mapped back to base/adapted before
    counting. ``win_rate`` is from the adapted model's perspective,
    excluding ties.
    """
    rng = random.Random(seed)
    report = WinRateReport()
    for task, base_response, adapted_response in items:
        adapted_is_a = rng.random() < 0.5
        a, b = (
            (adapted_response, base_response) if adapted_is_a else (base_response, adapted_response)
        )
        verdict = parse_verdict(
            judge_lm.completion(PAIRWISE_PROMPT.format(task=task, response_a=a, response_b=b))
        )
        if verdict == "tie":
            report.ties += 1
            report.verdicts.append("tie")
        elif (verdict == "A") == adapted_is_a:
            report.wins += 1
            report.verdicts.append("adapted")
        else:
            report.losses += 1
            report.verdicts.append("base")
    return report


def generate_response(
    model,
    tokenizer,
    prompt_messages: list[dict[str, str]],
    device: str,
    max_new_tokens: int = 512,
) -> str:
    """Greedy generation for one chat prompt (deterministic for evals)."""
    import torch

    chat_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)


def generate_pairs(
    base_model,
    adapted_model,
    tokenizer,
    episodes: list[Episode],
    device: str,
    max_new_tokens: int = 512,
) -> list[tuple[str, str, str]]:
    """Generate (task, base_response, adapted_response) items for judging."""
    items = []
    for ep in episodes:
        messages = ep.prompt_messages()
        base_out = generate_response(base_model, tokenizer, messages, device, max_new_tokens)
        adapted_out = generate_response(adapted_model, tokenizer, messages, device, max_new_tokens)
        items.append((ep.task, base_out, adapted_out))
    return items
