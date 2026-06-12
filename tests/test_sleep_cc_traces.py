"""Tests for the Claude Code transcript loader (fabricated session, no I/O beyond tmp)."""

import json

from rlm.sleep.cc_traces import (
    load_cc_episodes,
    load_cc_session,
    redact_secrets,
)


def rec(rtype: str, content, sidechain: bool = False) -> dict:
    return {"type": rtype, "isSidechain": sidechain, "message": {"role": rtype, "content": content}}


def tool_use(tid: str, name: str, **inp) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def tool_result(tid: str, text: str, is_error: bool = False) -> dict:
    return {"type": "tool_result", "tool_use_id": tid, "content": text, "is_error": is_error}


def fake_session() -> list[dict]:
    return [
        # harness noise records that must be ignored
        {"type": "file-history-snapshot", "data": "x"},
        {"type": "system", "message": {"role": "system", "content": "noise"}},
        # meta user record (slash command) — not a human turn
        rec("user", "<command-name>/clear</command-name>\n<command-args></command-args>"),
        # episode 0: error -> corrected retry -> success
        rec("user", "<system-reminder>injected</system-reminder>Run the test suite please"),
        rec(
            "assistant",
            [
                {"type": "text", "text": "Running tests."},
                tool_use("t1", "Bash", command="pytest -q"),
            ],
        ),
        rec("user", [tool_result("t1", "2 FAILED, 8 passed", is_error=False)]),
        rec(
            "assistant",
            [
                {"type": "text", "text": "Two failures; fixing the import."},
                tool_use("t2", "Edit", file_path="a.py"),
            ],
        ),
        rec("user", [tool_result("t2", "edit applied")]),
        rec("assistant", [tool_use("t3", "Bash", command="pytest -q")]),
        rec("user", [tool_result("t3", "10 passed")]),
        rec("assistant", [{"type": "text", "text": "All green now."}]),
        # sidechain records must be excluded
        rec("assistant", [{"type": "text", "text": "subagent chatter"}], sidechain=True),
        # episode 1 starts; then a correction turn merges into it
        rec("user", "Deploy the service"),
        rec(
            "assistant",
            [
                {"type": "text", "text": "Deploying with key sk-abcdefghijklmnopqrstuv123456."},
                tool_use("t4", "Bash", command="deploy --prod"),
            ],
        ),
        rec("user", [tool_result("t4", "Permission denied", is_error=True)]),
        rec("user", "no, that's wrong - use the staging target first"),
        rec("assistant", [tool_use("t5", "Bash", command="deploy --staging")]),
        rec("user", [tool_result("t5", "deployed ok")]),
        rec("assistant", [{"type": "text", "text": "Staging deploy done."}]),
        # episode 2: too short (no assistant step), must be dropped
        rec("user", "thanks!"),
    ]


def write_session(tmp_path, records, name="abc12345-6789-dead-beef-000000000000"):
    path = tmp_path / "-Users-jake-proj" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def test_segmentation_and_step_mapping(tmp_path):
    episodes = load_cc_session(write_session(tmp_path, fake_session()))
    assert len(episodes) == 2
    ep0, ep1 = episodes

    assert ep0.task == "Run the test suite please"  # system-reminder stripped
    roles = [s.role for s in ep0.steps]
    assert roles == [
        "assistant",
        "assistant",
        "tool",
        "assistant",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "[tool:Bash] pytest -q" in ep0.steps[1].content
    assert ep0.meta["n_tool_calls"] == 3


def test_soft_error_then_success_is_success_outcome(tmp_path):
    ep0 = load_cc_session(write_session(tmp_path, fake_session()))[0]
    assert ep0.meta["n_soft_errors"] == 1  # "2 FAILED, 8 passed"
    assert ep0.meta["n_tool_errors"] == 0
    assert ep0.outcome == "success"  # error was resolved by the retry


def test_correction_turn_merges_not_splits(tmp_path):
    ep1 = load_cc_session(write_session(tmp_path, fake_session()))[1]
    assert ep1.task == "Deploy the service"
    assert ep1.meta["n_user_corrections"] == 1
    user_steps = [s for s in ep1.steps if s.role == "user"]
    assert any("staging target" in s.content for s in user_steps)
    # hard error recorded, and the error step is rendered with the marker
    assert ep1.meta["n_tool_errors"] == 1
    assert any(s.content.startswith("[error]") for s in ep1.steps if s.role == "tool")
    assert ep1.outcome == "success"  # corrected deploy succeeded after the error


def test_secrets_are_redacted(tmp_path):
    ep1 = load_cc_session(write_session(tmp_path, fake_session()))[1]
    joined = "\n".join(s.content for s in ep1.steps)
    assert "sk-abcdefghijklmnopqrstuv123456" not in joined
    assert "[REDACTED]" in joined


def test_redact_secrets_patterns():
    assert "[REDACTED]" in redact_secrets("token ghp_ABCDEFGHIJKLMNOP1234")
    assert "[REDACTED]" in redact_secrets("aws AKIAABCDEFGHIJKLMNOP")
    assert "api_key = [REDACTED]" in redact_secrets("api_key = abcd1234efgh5678ijkl")
    assert redact_secrets("plain text stays") == "plain text stays"


def test_load_cc_episodes_excludes_subagents_by_default(tmp_path):
    write_session(tmp_path, fake_session())
    sub = tmp_path / "-Users-jake-proj" / "s1" / "subagents" / "agent-x.jsonl"
    sub.parent.mkdir(parents=True)
    sub.write_text(
        json.dumps(rec("user", "Subagent task"))
        + "\n"
        + json.dumps(
            rec(
                "assistant",
                [{"type": "text", "text": "did it"}, tool_use("t9", "Bash", command="ls")],
            )
        )
        + "\n"
        + json.dumps(rec("user", [tool_result("t9", "ok")]))
    )
    assert len(load_cc_episodes(tmp_path)) == 2
    assert len(load_cc_episodes(tmp_path, include_subagents=True, min_steps=2)) == 3


def test_unresolved_error_is_failure(tmp_path):
    records = [
        rec("user", "Try the thing"),
        rec(
            "assistant", [{"type": "text", "text": "Trying."}, tool_use("t1", "Bash", command="x")]
        ),
        rec("user", [tool_result("t1", "boom", is_error=True)]),
        rec("assistant", [{"type": "text", "text": "That failed."}]),
    ]
    ep = load_cc_session(write_session(tmp_path, records), min_steps=2)[0]
    assert ep.outcome == "failure"


def test_long_step_truncation_keeps_tail(tmp_path):
    records = [
        rec("user", "Big output"),
        rec("assistant", [tool_use("t1", "Bash", command="cat big")]),
        rec("user", [tool_result("t1", "HEAD" + "x" * 5000 + "TAIL_ERROR_HERE")]),
        rec("assistant", [{"type": "text", "text": "Saw it."}]),
    ]
    ep = load_cc_session(write_session(tmp_path, records), min_steps=2)[0]
    tool_step = next(s for s in ep.steps if s.role == "tool")
    assert len(tool_step.content) <= 1600
    assert "TAIL_ERROR_HERE" in tool_step.content
    assert "truncated" in tool_step.content
