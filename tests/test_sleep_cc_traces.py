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


def test_interrupt_marker_is_correction_signal_not_task(tmp_path):
    records = [
        rec("user", "Refactor the loader"),
        rec(
            "assistant",
            [{"type": "text", "text": "Starting."}, tool_use("t1", "Bash", command="x")],
        ),
        rec("user", [tool_result("t1", "running...")]),
        rec("user", "[Request interrupted by user]"),
        rec("user", "no, leave the public API alone, just fix the internals"),
        rec("assistant", [{"type": "text", "text": "Understood, internals only."}]),
    ]
    episodes = load_cc_session(write_session(tmp_path, records), min_steps=2)
    assert len(episodes) == 1  # the interrupt did NOT start a junk episode
    ep = episodes[0]
    assert ep.task == "Refactor the loader"
    assert ep.meta["n_user_corrections"] == 2  # interrupt + explicit correction


def test_continuation_summary_is_not_a_task(tmp_path):
    records = [
        rec("user", "This session is being continued from a previous conversation that ran out"),
        rec("assistant", [{"type": "text", "text": "orphan"}]),
        rec("user", "Real task here"),
        rec(
            "assistant", [{"type": "text", "text": "On it."}, tool_use("t1", "Bash", command="ls")]
        ),
        rec("user", [tool_result("t1", "ok")]),
    ]
    episodes = load_cc_session(write_session(tmp_path, records), min_steps=2)
    assert [e.task for e in episodes] == ["Real task here"]


def test_no_problem_opener_starts_new_episode(tmp_path):
    records = [
        rec("user", "First task"),
        rec("assistant", [{"type": "text", "text": "Done."}, tool_use("t1", "Bash", command="a")]),
        rec("user", [tool_result("t1", "ok")]),
        rec("user", "No problem, next please look at the README"),
        rec(
            "assistant", [{"type": "text", "text": "Looking."}, tool_use("t2", "Bash", command="b")]
        ),
        rec("user", [tool_result("t2", "ok")]),
    ]
    episodes = load_cc_session(write_session(tmp_path, records), min_steps=2)
    assert len(episodes) == 2
    assert episodes[1].task.startswith("No problem")


def test_env_var_assignment_and_url_creds_redacted():
    assert "OPENAI_API_KEY=[REDACTED]" in redact_secrets(
        "export OPENAI_API_KEY=abc123def456ghi789jkl"
    )
    assert "[REDACTED]" in redact_secrets(
        "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----"
    )
    assert redact_secrets("https://user:hunter22@host.com/x") == "https://[REDACTED]@host.com/x"
    out = redact_secrets("GET https://api.example.com/v1?access_token=abcd1234efgh&x=1")
    assert "abcd1234efgh" not in out and "access_token=[REDACTED]" in out
    # prose survives: no assignment separator
    assert redact_secrets("the password requirements documentation") == (
        "the password requirements documentation"
    )


def test_soft_error_regex_directions():
    from rlm.sleep.cc_traces import SOFT_ERROR_RE

    assert SOFT_ERROR_RE.search("Error: Cannot find module 'foo'")
    assert SOFT_ERROR_RE.search("ValueError: invalid literal for int()")
    assert SOFT_ERROR_RE.search("main.c:10: error: expected ';'")
    assert not SOFT_ERROR_RE.search("src/log.py:12: logger.error: retry handling")
    assert not SOFT_ERROR_RE.search("All checks passed and nothing else")


def test_non_dict_jsonl_lines_do_not_crash(tmp_path):
    path = tmp_path / "p" / "s.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '["a json array"]\n42\n{"type": "user", "message": "not a dict"}\n'
        + json.dumps(rec("user", "Task"))
        + "\n"
        + json.dumps(
            rec(
                "assistant",
                [{"type": "text", "text": "Done it."}, tool_use("t1", "Bash", command="ls")],
            )
        )
        + "\n"
        + json.dumps(rec("user", [tool_result("t1", "ok")]))
    )
    episodes = load_cc_session(path, min_steps=2)
    assert len(episodes) == 1 and episodes[0].task == "Task"


MEMORY_BODY = (
    "---\nname: prefer-debug-builds\ndescription: Skip --release during iteration\n---\n\n"
    "Release builds take a minute on this project; use cargo check/build/test "
    "without --release in the inner loop. Only ship --release for benchmarks."
)


def memory_session() -> list[dict]:
    return [
        rec("user", "Why are the builds so slow?"),
        rec(
            "assistant",
            [
                {"type": "text", "text": "Checking the build profile."},
                tool_use("t1", "Bash", command="cargo build --release --timings"),
            ],
        ),
        rec("user", [tool_result("t1", "Finished release in 61.2s")]),
        rec("user", "no, don't use --release while we iterate!"),
        rec(
            "assistant",
            [
                {"type": "text", "text": "Got it — noting this for future sessions."},
                tool_use(
                    "t2",
                    "Write",
                    file_path="/Users/jake/.claude/projects/-x/memory/prefer-debug-builds.md",
                    content=MEMORY_BODY,
                ),
                tool_use(
                    "t3",
                    "Edit",
                    file_path="/Users/jake/.claude/projects/-x/memory/MEMORY.md",
                    old_string="# Memory Index",
                    new_string="# Memory Index\n- prefer-debug-builds",
                ),
            ],
        ),
        rec("user", [tool_result("t2", "ok"), tool_result("t3", "ok")]),
        rec("assistant", [{"type": "text", "text": "Saved."}]),
    ]


def test_memory_pairs_extracted_with_context(tmp_path):
    from rlm.sleep.cc_traces import extract_memory_pairs

    pairs = extract_memory_pairs(write_session(tmp_path, memory_session()))
    assert len(pairs) == 1  # the MEMORY.md index Edit is not a pair
    pair = pairs[0]
    assert pair.file_name == "prefer-debug-builds.md"
    assert pair.task == "Why are the builds so slow?"
    assert pair.memory == MEMORY_BODY
    roles = [s.role for s in pair.context]
    assert "tool" in roles and "user" in roles  # build output + the correction
    assert any("don't use --release" in s.content for s in pair.context)
    messages = pair.to_messages()
    assert messages[0]["role"] == "user" and "Distill" in messages[0]["content"]
    assert messages[1]["content"] == MEMORY_BODY


def test_memory_pair_context_excludes_target_leakage(tmp_path):
    from rlm.sleep.cc_traces import extract_memory_pairs

    records = memory_session()
    # assistant drafts the memory verbatim in text right before writing it
    records.insert(
        4,
        rec("assistant", [{"type": "text", "text": "Here's the note I'll save:\n" + MEMORY_BODY}]),
    )
    pairs = extract_memory_pairs(write_session(tmp_path, records))
    assert len(pairs) == 1
    assert not any("Release builds take a minute" in s.content for s in pairs[0].context)


def test_episode_meta_counts_memory_writes(tmp_path):
    episodes = load_cc_session(write_session(tmp_path, memory_session()), min_steps=2)
    assert len(episodes) == 1
    assert episodes[0].meta["n_memory_writes"] == 2  # the file write + the index edit


def test_tiny_memory_writes_are_skipped(tmp_path):
    from rlm.sleep.cc_traces import extract_memory_pairs

    records = [
        rec("user", "Quick fix"),
        rec(
            "assistant",
            [
                tool_use(
                    "t1",
                    "Write",
                    file_path="/Users/jake/.claude/projects/-x/memory/stub.md",
                    content="tiny",
                )
            ],
        ),
        rec("user", [tool_result("t1", "ok")]),
    ]
    assert extract_memory_pairs(write_session(tmp_path, records)) == []


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
