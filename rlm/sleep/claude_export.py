"""Load a claude.ai data export (conversations.json) as sleep-pipeline episodes.

The claude.ai export is conversational rather than agentic: the dominant
learning signal is the *human* pushing back ("no, I meant…"), plus
occasional tool blocks (search/analysis/artifacts). Segmentation mirrors
``cc_traces``: each human turn opens an episode; correction-shaped
follow-ups merge into the open episode so the error→correction arc stays
whole.

Same privacy posture as ``cc_traces``: secrets are redacted before
anything leaves this module, and exports must never be committed.
"""

from __future__ import annotations

import json
from pathlib import Path

from rlm.sleep.cc_traces import (
    CORRECTION_RE,
    _render_tool_result,
    _render_tool_use,
    _truncate,
    redact_secrets,
)
from rlm.sleep.types import Episode, Step


def _message_steps(message: dict, max_step_chars: int) -> list[Step]:
    """Render one export message's content blocks as steps."""
    role = "user" if message.get("sender") == "human" else "assistant"
    steps: list[Step] = []
    blocks = message.get("content") or []
    if not blocks and message.get("text"):
        blocks = [{"type": "text", "text": message["text"]}]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text", "").strip():
            steps.append(Step(role=role, content=block["text"]))
        elif kind == "tool_use":
            steps.append(Step(role="assistant", content=_render_tool_use(block)))
        elif kind == "tool_result":
            steps.append(Step(role="tool", content=_render_tool_result(block)))
    for step in steps:
        step.content = _truncate(redact_secrets(step.content), max_step_chars)
    return steps


def load_claude_export(
    path: str | Path,
    min_steps: int = 1,
    min_assistant_chars: int = 200,
    max_steps: int = 60,
    max_step_chars: int = 1500,
    limit: int | None = None,
) -> list[Episode]:
    """Parse conversations.json into episodes (one per human turn, corrections merged)."""
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "conversations.json"
    conversations = json.loads(path.read_text(encoding="utf-8"))

    episodes: list[Episode] = []
    for convo in conversations:
        convo_id = str(convo.get("uuid", ""))[:8]
        messages = convo.get("chat_messages") or []
        task: str | None = None
        steps: list[Step] = []
        n_corrections = 0
        index = 0

        def flush(convo: dict = convo, convo_id: str = convo_id) -> None:
            nonlocal task, steps, n_corrections, index
            assistant_chars = sum(len(s.content) for s in steps if s.role == "assistant")
            # A chat exchange's depth is its content, not its step count:
            # most useful turns are a single substantial assistant message.
            if task and len(steps) >= min_steps and assistant_chars >= min_assistant_chars:
                kept = (
                    steps
                    if len(steps) <= max_steps
                    else steps[: max_steps // 4] + steps[-(max_steps - max_steps // 4) :]
                )
                episodes.append(
                    Episode(
                        episode_id=f"cai_{convo_id}_{index:03d}",
                        source="claude_ai_export",
                        task=task,
                        steps=kept,
                        outcome="unknown",
                        meta={
                            "conversation": str(convo.get("name", ""))[:120],
                            "created_at": str(convo.get("created_at", ""))[:10],
                            "n_user_corrections": n_corrections,
                            "n_tool_calls": sum(1 for s in kept if s.content.startswith("[tool:")),
                        },
                    )
                )
                index += 1
            task, steps, n_corrections = None, [], 0

        for message in messages:
            if message.get("sender") == "human":
                text = redact_secrets(str(message.get("text") or "").strip())
                if not text:
                    continue
                if task is not None and CORRECTION_RE.search(text):
                    n_corrections += 1
                    steps.append(Step(role="user", content=_truncate(text, max_step_chars)))
                else:
                    flush()
                    task = _truncate(text, 800)
            elif task is not None:
                steps.extend(_message_steps(message, max_step_chars))
        flush()
        if limit is not None and len(episodes) >= limit:
            return episodes[:limit]
    return episodes
