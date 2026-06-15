"""Tests for the claude.ai export loader (fabricated export, no real data)."""

import json

from rlm.sleep.claude_export import load_claude_export


def msg(sender: str, text: str = "", blocks: list | None = None) -> dict:
    return {"sender": sender, "text": text, "content": blocks or []}


def fake_export() -> list[dict]:
    return [
        {
            "uuid": "abcd1234-0000",
            "name": "Debugging a talk outline",
            "created_at": "2026-03-01T10:00:00Z",
            "chat_messages": [
                msg("human", "Draft an outline for my agents talk"),
                msg(
                    "assistant",
                    blocks=[
                        {"type": "text", "text": "Here is a draft outline... " + "o" * 250},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "web_search",
                            "input": {"query": "agentic design patterns"},
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "search results here",
                        },
                    ],
                ),
                msg("human", "no, that's wrong - the audience is engineers, not execs"),
                msg("assistant", blocks=[{"type": "text", "text": "Revised for engineers."}]),
                msg("human", "Now a totally new topic: plan dinner"),
                msg("assistant", blocks=[{"type": "text", "text": "Dinner plan: " + "x" * 250}]),
            ],
        },
        {
            "uuid": "efgh5678-0000",
            "name": "One-liner",
            "created_at": "2024-05-01T10:00:00Z",
            "chat_messages": [
                msg("human", "hi"),
                msg("assistant", blocks=[{"type": "text", "text": "hello"}]),
            ],
        },
    ]


def write_export(tmp_path, convos):
    d = tmp_path / "export"
    d.mkdir(exist_ok=True)
    (d / "conversations.json").write_text(json.dumps(convos))
    return d


def test_segmentation_and_correction_merge(tmp_path):
    episodes = load_claude_export(write_export(tmp_path, fake_export()))
    # turn 1 (with merged correction) and turn 3 from convo 1; convo 2 too short
    assert len(episodes) == 2
    ep0, ep1 = episodes
    assert ep0.task == "Draft an outline for my agents talk"
    assert ep0.meta["n_user_corrections"] == 1
    assert any("audience is engineers" in s.content for s in ep0.steps if s.role == "user")
    assert any(s.content.startswith("[tool:web_search]") for s in ep0.steps)
    assert any(s.role == "tool" for s in ep0.steps)
    assert ep0.meta["conversation"] == "Debugging a talk outline"
    assert ep1.task == "Now a totally new topic: plan dinner"


def test_thin_exchange_dropped_by_content_gate(tmp_path):
    episodes = load_claude_export(write_export(tmp_path, fake_export()))
    assert all(not e.task.startswith("hi") for e in episodes)
    # but with the gate lowered the one-liner shows up
    episodes = load_claude_export(write_export(tmp_path, fake_export()), min_assistant_chars=1)
    assert any(e.task == "hi" for e in episodes)


def test_secrets_redacted_in_export(tmp_path):
    convos = [
        {
            "uuid": "x",
            "name": "k",
            "created_at": "2026-01-01",
            "chat_messages": [
                msg("human", "my key is sk-abcdefghijklmnopqrstuv1234 ok?"),
                msg(
                    "assistant",
                    blocks=[
                        {"type": "text", "text": "Noted."},
                        {"type": "text", "text": "Echo: sk-abcdefghijklmnopqrstuv1234"},
                    ],
                ),
            ],
        }
    ]
    episodes = load_claude_export(write_export(tmp_path, convos), min_assistant_chars=1)
    joined = episodes[0].task + "".join(s.content for s in episodes[0].steps)
    assert "sk-abcdefghijklmnopqrstuv1234" not in joined
