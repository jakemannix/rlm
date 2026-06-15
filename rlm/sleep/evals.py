"""Eval harness for nightly consolidation.

Three measurements per model (base, and base+adapter):

* **next-day NLL** — teacher-forced NLL of gold assistant responses on the
  *following* day's episodes (forward transfer: did tonight's lessons help
  tomorrow?).
* **test NLL** — same metric on the held-out test split (generalization,
  fixed across all days/sweep points).
* **retention NLL** — NLL on a small fixed general-text probe (forgetting
  guard: the verification gate for the whole nightly update).

Lower is better for all three; the report records deltas vs. the base.
"""

from __future__ import annotations

from rlm.sleep.retention_probe import RETENTION_PROBE_V1
from rlm.sleep.types import Episode, EvalReport

RETENTION_PROBE: list[str] = RETENTION_PROBE_V1


def response_nll(
    model,
    tokenizer,
    prompt_messages: list[dict[str, str]],
    response: str,
    device: str,
    max_seq_len: int = 1024,
) -> float:
    """Mean per-token NLL (nats) of ``response`` given the chat prompt.

    When prompt + response exceed ``max_seq_len``, the *prompt* is
    left-truncated so the response is always what gets scored — otherwise
    long-prompt episodes (e.g. AgentInstruct os instructions) would
    silently measure leftover prompt tokens instead.
    """
    import torch

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    # Keep the full response (capped at half the window so some prompt
    # context always remains), then fill the rest with the prompt's tail.
    n_response = min(len(response_ids), max(1, max_seq_len // 2))
    n_prompt = min(len(prompt_ids), max_seq_len - n_response)
    full_ids = prompt_ids[len(prompt_ids) - n_prompt :] + response_ids[:n_response]

    input_ids = torch.tensor([full_ids], device=device)
    labels = torch.full_like(input_ids, -100)
    labels[0, n_prompt:] = input_ids[0, n_prompt:]
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    return float(out.loss.item())


def text_nll(model, tokenizer, text: str, device: str, max_seq_len: int = 512) -> float:
    """Mean per-token NLL of plain text (for the retention probe)."""
    import torch

    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_seq_len]
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return float(out.loss.item())


def episodes_nll(
    model,
    tokenizer,
    episodes: list[Episode],
    device: str,
    max_seq_len: int = 1024,
) -> float:
    """Mean teacher-forced NLL of final assistant responses over episodes."""
    if not episodes:
        raise ValueError("episodes_nll called with no episodes")
    total = 0.0
    for ep in episodes:
        total += response_nll(
            model,
            tokenizer,
            prompt_messages=ep.prompt_messages(),
            response=ep.final_response(),
            device=device,
            max_seq_len=max_seq_len,
        )
    return total / len(episodes)


def evaluate_model(
    model,
    tokenizer,
    name: str,
    next_day: list[Episode],
    test: list[Episode],
    device: str,
    max_seq_len: int = 1024,
) -> EvalReport:
    """Run the full harness for one model."""
    model.eval()
    retention = sum(text_nll(model, tokenizer, t, device=device) for t in RETENTION_PROBE) / len(
        RETENTION_PROBE
    )
    return EvalReport(
        name=name,
        next_day_nll=episodes_nll(model, tokenizer, next_day, device, max_seq_len),
        test_nll=episodes_nll(model, tokenizer, test, device, max_seq_len),
        retention_nll=retention,
        n_next_day=len(next_day),
        n_test=len(test),
    )
