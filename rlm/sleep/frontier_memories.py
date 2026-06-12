"""Frontier-curated memory candidates -> training data.

Consumes the ``frontier_quality_audit`` artifacts (see
``exhaustive_round1/FABLE_HANDOFF.md``): 71 typed, scoped, evidence-backed
memory candidates curated by a frontier model over the full claude.ai
archive. This module resolves each candidate's evidence IDs back to raw
conversation context and emits training examples, plus per-day shards
keyed on evidence dates for nightly-LoRA experiments.

Per Jake's explicit instruction (2026-06-12): this is his data and it is
NOT privacy-filtered or truncated for frontier-model consumption.
Credential redaction (security hygiene) still applies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rlm.sleep.cc_traces import redact_secrets

DISTILL_INSTRUCTION = (
    "Review this conversation evidence and distill the single durable, "
    "reusable memory a future agent working with this user would need. "
    "State it as one concrete, actionable note."
)


@dataclass
class Evidence:
    """One resolved evidence window: raw conversation context around a turn."""

    evidence_id: str
    conversation_title: str
    created_at: str  # YYYY-MM-DD
    context: str


@dataclass
class FrontierMemory:
    """One curated memory candidate with its resolved evidence."""

    memory_id: str
    tier: str
    memory_type: str
    scope: str
    memory: str
    rationale: str
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def day(self) -> str:
        """The candidate's nightly bucket: the earliest evidence date."""
        dates = [e.created_at for e in self.evidence if e.created_at]
        return min(dates) if dates else "unknown"


def _message_text(message: dict) -> str:
    text = str(message.get("text") or "").strip()
    if text:
        return text
    parts = []
    for block in message.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def resolve_evidence(
    index_row: dict,
    conversations_by_uuid: dict[str, dict],
    window: int = 4,
    max_message_chars: int = 4000,
) -> Evidence:
    """Raw conversation context around the evidence turn (sender-labeled)."""
    convo = conversations_by_uuid.get(index_row.get("conversation_uuid", "")) or {}
    messages = convo.get("chat_messages") or []
    turn = int(index_row.get("turn_index", 0))
    lo, hi = max(0, turn - window), min(len(messages), turn + window + 1)
    lines = []
    for message in messages[lo:hi]:
        text = _message_text(message)
        if not text:
            continue
        if len(text) > max_message_chars:
            text = text[:max_message_chars] + " [...]"
        lines.append(f"[{message.get('sender', '?')}] {text}")
    return Evidence(
        evidence_id=str(index_row.get("id", "")),
        conversation_title=str(convo.get("name") or index_row.get("title") or ""),
        created_at=str(index_row.get("created_at", ""))[:10],
        context=redact_secrets("\n\n".join(lines)),
    )


def load_frontier_memories(
    audit_dir: str | Path,
    conversations_path: str | Path | None = None,
    window: int = 4,
    max_message_chars: int = 4000,
    max_evidence_per_memory: int = 3,
) -> list[FrontierMemory]:
    """Load memory_commit_candidates.jsonl with evidence fully resolved."""
    audit_dir = Path(audit_dir).expanduser()
    conversations_path = Path(
        conversations_path or audit_dir.parent / "conversations.json"
    ).expanduser()

    index: dict[str, dict] = {}
    with (audit_dir / "candidate_index.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[row["id"]] = row
    conversations = json.loads(conversations_path.read_text(encoding="utf-8"))
    by_uuid = {c.get("uuid"): c for c in conversations}

    memories: list[FrontierMemory] = []
    candidates_path = audit_dir / "exhaustive_round1" / "memory_commit_candidates.jsonl"
    with candidates_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            evidence = [
                resolve_evidence(
                    index[eid], by_uuid, window=window, max_message_chars=max_message_chars
                )
                for eid in row.get("ids", [])[:max_evidence_per_memory]
                if eid in index
            ]
            memories.append(
                FrontierMemory(
                    memory_id=f"fm_{i:03d}",
                    tier=str(row.get("tier", "")),
                    memory_type=str(row.get("memory_type", "")),
                    scope=str(row.get("scope", "")),
                    memory=str(row.get("candidate_memory", "")),
                    rationale=str(row.get("why_future_agents_should_remember_this", "")),
                    evidence=evidence,
                )
            )
    return memories


def distillation_example(memory: FrontierMemory) -> dict:
    """Chat-format SFT example: evidence context -> the curated memory."""
    blocks = [
        f"=== {e.conversation_title} ({e.created_at}) ===\n{e.context}" for e in memory.evidence
    ]
    user = "\n\n".join(blocks) + f"\n\n{DISTILL_INSTRUCTION}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": memory.memory},
        ],
        "memory_id": memory.memory_id,
        "tier": memory.tier,
        "memory_type": memory.memory_type,
        "day": memory.day,
    }


def shard_by_day(examples: list[dict]) -> dict[str, list[dict]]:
    """Group examples into nightly buckets by their evidence day."""
    shards: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        shards[example.get("day", "unknown")].append(example)
    return dict(sorted(shards.items()))
