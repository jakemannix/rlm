"""The memory-worthiness judging task: is a candidate memory worth keeping?

This is curation stage 1 — given a candidate memory and the conversation
evidence it came from, decide keep/reject and a quality tier. Shared by the
frontier labeler (builds the trainset), the fine-tuned 8B judge (trained to
imitate), and the two-stage curator (uses it for cheap bulk filtering), so
all three speak the same protocol.
"""

from __future__ import annotations

import json

from rlm.sleep.judge import extract_json

WORTHINESS_PROMPT = """\
You are curating a personal agent's long-term memory. Decide whether this \
candidate memory — distilled from the conversation evidence below — is worth \
keeping and training a future agent on.

CANDIDATE MEMORY:
{memory}

EVIDENCE (excerpt of the conversation it came from):
{evidence}

Keep it only if it is durable (useful beyond this one conversation), \
grounded in the evidence, specific enough to act on, and not a vague \
platitude. Assign a tier: "gold" = clearly worth it, broadly reusable; \
"silver" = useful but narrow or weakly grounded; "reject" = not worth \
training on.

Respond with ONLY a JSON object:
{{"keep": true or false, "tier": "gold" or "silver" or "reject",
  "reason": "<one short sentence>"}}
"""


def worthiness_messages(memory: str, evidence: str, label: dict | None = None) -> list[dict]:
    """Chat messages for labeling/training/inference; label set on the
    assistant turn only when building training data."""
    user = WORTHINESS_PROMPT.format(memory=memory, evidence=evidence)
    messages = [{"role": "user", "content": user}]
    if label is not None:
        messages.append({"role": "assistant", "content": json.dumps(label)})
    return messages


def parse_worthiness(text: str) -> dict | None:
    """Parse a worthiness verdict; None if unparseable."""
    try:
        parsed = extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return None
    tier = str(parsed.get("tier", "")).lower()
    if tier not in ("gold", "silver", "reject"):
        return None
    return {"keep": tier != "reject", "tier": tier, "reason": str(parsed.get("reason", ""))[:200]}
