"""Load Claude Code session transcripts as sleep-pipeline episodes.

Claude Code writes one JSONL per session under ``~/.claude/projects``
(snapshot them with ``scripts/sleep/extract_cc_sessions.py``). Unlike
AgentInstruct's clean expert demonstrations, these traces carry the three
signal classes the consolidation loop actually wants: hard tool errors
(``is_error`` results), soft errors (failing tests / tracebacks inside
"successful" outputs), and human corrections ("no, do X instead") — see
``docs/learning_signal.md``.

Segmentation: each human prompt starts an episode (the prompt is the
task); assistant text, tool calls, and tool results become steps until
the next human prompt. A human turn that *reads* like a correction is
merged into the open episode instead of starting a new one — the
error-followed-by-correction arc is exactly what we want the judge to
see whole.

Secrets are redacted before anything leaves this module; transcripts are
personal data and must never be committed or shipped to an external API
without review.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm.sleep.types import Episode, Step

# Human turns matching this continue the open episode rather than starting
# a fresh one: a correction is part of the same task's arc. "no problem" /
# "no worries"-style openers are exempted — they start new tasks.
CORRECTION_RE = re.compile(
    r"^\s*(no\b(?!\s+(problem|worries|rush|need|thanks))|wrong\b|that'?s (not|wrong)"
    r"|actually\b|instead\b|stop\b|undo\b|revert\b"
    r"|not what|didn'?t work|still (fails|failing|broken|wrong))",
    re.IGNORECASE,
)

# Harness-generated user turns that are neither a task nor a correction:
# Esc-interrupts and compaction-continuation summaries. The interrupt is
# itself weak correction signal (the human stopped a bad trajectory).
INTERRUPT_RE = re.compile(r"^\[Request interrupted by user( for tool use)?\]$")
CONTINUATION_RE = re.compile(r"^This session is being continued from a previous", re.IGNORECASE)

# Harness-injected spans inside user content that are not the human speaking.
META_SPAN_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>"
    r"|<local-command-caveat>.*?</local-command-caveat>"
    r"|<local-command-stdout>.*?</local-command-stdout>"
    r"|<command-name>.*?</command-name>"
    r"|<command-message>.*?</command-message>"
    r"|<command-args>.*?</command-args>"
    r"|<task-notification>.*?</task-notification>"
    r"|<bash-input>.*?</bash-input>"
    r"|<bash-stdout>.*?</bash-stdout>"
    r"|<bash-stderr>.*?</bash-stderr>",
    re.DOTALL,
)

SECRET_RES = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\b(sk|pk)-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}"),
]
# URL credentials: keep the URL shape, drop the secret part.
URL_BASIC_AUTH_RE = re.compile(r"(://)[^/\s:@]+:[^/\s@]+@")
URL_TOKEN_PARAM_RE = re.compile(
    r"(?i)([?&](?:token|key|apikey|api_key|access_token|sig|signature)=)[^&\s\"']+"
)
# Assignment-shaped credentials. Allows prefixed names (OPENAI_API_KEY) and
# requires an explicit =/: separator so prose mentioning "password" survives.
KV_SECRET_RE = re.compile(
    r"(?i)\b([A-Za-z0-9_]*(?:api[_-]?key|secret|token|password|passwd))"
    r"(['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9._~+/-]{12,})"
)

SOFT_ERROR_RE = re.compile(
    r"Traceback \(most recent call"
    r"|\bFAILED\b"
    r"|\b[A-Z][A-Za-z]*Error\b"  # ValueError, TypeError, "Error: Cannot find module"
    r"|command not found|No such file"
    r"|^\s*(?:\S{0,40}[:\s]\s*)?[Ee]rror:"  # line-anchored compiler/CLI "error:"
    r"|\bERROR\b"
    r"|[Ee]xit code [1-9]"
    r"|\bfatal:|Permission denied",
    re.MULTILINE,
)


def redact_secrets(text: str) -> str:
    """Best-effort scrub of credential-shaped substrings."""
    for pattern in SECRET_RES:
        text = pattern.sub("[REDACTED]", text)
    text = URL_BASIC_AUTH_RE.sub(r"\1[REDACTED]@", text)
    text = URL_TOKEN_PARAM_RE.sub(r"\1[REDACTED]", text)
    return KV_SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)


def _truncate(text: str, max_chars: int) -> str:
    """Keep head and tail — errors and their resolutions live at the tail."""
    if len(text) <= max_chars:
        return text
    half = (max_chars - 30) // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} chars truncated ...]\n{text[-half:]}"


def _human_text(content: Any) -> str | None:
    """The human's words from a user record, or None if it's not a human turn."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        text = "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        return None
    text = META_SPAN_RE.sub("", text).strip()
    return text or None


# A Write/Edit landing here is an explicit "this is worth remembering"
# decision made in-context by the session's (frontier) model — the
# highest-precision curation signal in the corpus.
MEMORY_FILE_RE = re.compile(r"/memory/(?!MEMORY\.md$)[^/]+\.md$")
MEMORY_DECISION_RE = re.compile(r"/memory/[^/]+\.md$|/Memory/[^/]+\.md$")


def _render_tool_use(block: dict) -> str:
    name = block.get("name", "?")
    inp = block.get("input") or {}
    if name == "Bash" and "command" in inp:
        detail = inp["command"]
    elif "file_path" in inp:
        detail = inp["file_path"]
    else:
        detail = json.dumps(inp, ensure_ascii=False)
    return f"[tool:{name}] {detail}"


def _render_tool_result(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif b.get("type") == "tool_reference":
                parts.append(f"<{b.get('tool_name', '?')}>")
        text = "\n".join(parts)
    else:
        text = str(content or "")
    prefix = "[error] " if block.get("is_error") else ""
    return f"{prefix}{text}"


class _EpisodeBuilder:
    def __init__(self, session_id: str, source: str, index: int, task: str, max_step_chars: int):
        self.session_id = session_id
        self.source = source
        self.index = index
        self.task = task
        self.max_step_chars = max_step_chars
        self.steps: list[Step] = []
        self.n_tool_calls = 0
        self.n_tool_errors = 0
        self.n_soft_errors = 0
        self.n_user_corrections = 0
        self.n_memory_writes = 0
        self.last_error_step = -1
        self.last_success_step = -1
        self.id_to_tool: dict[str, str] = {}

    def add(self, role: str, content: str) -> str:
        # Redact BEFORE truncating: truncation can split a credential and
        # leave a fragment the patterns no longer recognize.
        content = _truncate(redact_secrets(content), self.max_step_chars)
        self.steps.append(Step(role=role, content=content))
        return content

    def note_result(self, *, is_error: bool, soft_error: bool) -> None:
        step_idx = len(self.steps)
        if is_error:
            self.n_tool_errors += 1
            self.last_error_step = step_idx
        elif soft_error:
            self.n_soft_errors += 1
            self.last_error_step = step_idx
        else:
            self.last_success_step = step_idx

    def build(self, min_steps: int, max_steps: int) -> Episode | None:
        if len(self.steps) < min_steps:
            return None
        if not any(s.role == "assistant" for s in self.steps):
            return None
        steps = self.steps
        if len(steps) > max_steps:
            head = max_steps // 4
            steps = steps[:head] + steps[-(max_steps - head) :]
        had_errors = self.n_tool_errors + self.n_soft_errors > 0
        if not had_errors:
            outcome = "success" if self.n_tool_calls else "unknown"
        else:
            outcome = "success" if self.last_success_step > self.last_error_step else "failure"
        return Episode(
            episode_id=f"cc_{self.session_id[:8]}_{self.index:03d}",
            source=self.source,
            task=redact_secrets(_truncate(self.task, 800)),
            steps=steps,
            outcome=outcome,
            meta={
                "session_id": self.session_id,
                "n_tool_calls": self.n_tool_calls,
                "n_tool_errors": self.n_tool_errors,
                "n_soft_errors": self.n_soft_errors,
                "n_user_corrections": self.n_user_corrections,
                "n_memory_writes": self.n_memory_writes,
            },
        )


def load_cc_session(
    path: str | Path,
    min_steps: int = 3,
    max_steps: int = 80,
    max_step_chars: int = 1500,
) -> list[Episode]:
    """Parse one Claude Code session transcript into episodes."""
    path = Path(path)
    source = f"claude_code/{path.parent.name}"
    session_id = path.stem
    episodes: list[Episode] = []
    builder: _EpisodeBuilder | None = None

    def flush() -> None:
        nonlocal builder
        if builder is not None:
            episode = builder.build(min_steps, max_steps)
            if episode is not None:
                episodes.append(episode)
        builder = None

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("isSidechain") or rec.get("type") not in ("user", "assistant"):
                continue
            message = rec.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")

            if rec.get("type") == "user":
                human = _human_text(content)
                if human is not None:
                    if INTERRUPT_RE.match(human):
                        # Esc-interrupt: harness meta, but the human stopping
                        # a trajectory is itself weak correction signal.
                        if builder is not None:
                            builder.n_user_corrections += 1
                            builder.add("user", human)
                        continue
                    if CONTINUATION_RE.match(human):
                        continue  # compaction summary, not a human task
                    if builder is not None and CORRECTION_RE.search(human):
                        builder.n_user_corrections += 1
                        builder.add("user", human)
                    else:
                        flush()
                        builder = _EpisodeBuilder(
                            session_id, source, len(episodes), human, max_step_chars
                        )
                    continue
                if builder is None or not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        # Scan the SAME text that gets stored, so meta/outcome
                        # never contradict the visible steps.
                        stored = builder.add("tool", _render_tool_result(block))
                        builder.note_result(
                            is_error=bool(block.get("is_error")),
                            soft_error=bool(SOFT_ERROR_RE.search(stored)),
                        )
                continue

            # assistant record
            if builder is None or not isinstance(content, list):
                continue
            texts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if any(t.strip() for t in texts):
                builder.add("assistant", "\n".join(t for t in texts if t.strip()))
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    builder.n_tool_calls += 1
                    builder.id_to_tool[block.get("id", "")] = block.get("name", "?")
                    inp = block.get("input") or {}
                    if block.get("name") in ("Write", "Edit") and MEMORY_DECISION_RE.search(
                        str(inp.get("file_path", ""))
                    ):
                        builder.n_memory_writes += 1
                    builder.add("assistant", _render_tool_use(block))
    flush()
    return episodes


@dataclass
class MemoryWritePair:
    """One frontier-authored memory with the experience that produced it.

    Mined from Write calls targeting memory files: the session's model
    already decided this was worth remembering and wrote the distilled
    artifact — a (context -> memory) training pair by construction, with
    a far stronger author than any judge we run locally.
    """

    pair_id: str
    session_id: str
    file_name: str
    task: str
    context: list[Step]
    memory: str

    def to_messages(self) -> list[dict[str, str]]:
        transcript = "\n".join(f"[{s.role}] {s.content}" for s in self.context)
        user = (
            f"TASK: {self.task}\n\n{transcript}\n\n"
            "Distill the durable, reusable memory from this session that a "
            "future session would need. Write it as a standalone note."
        )
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": self.memory},
        ]


def _overlaps_target(step_content: str, target: str, window: int = 120) -> bool:
    """True when a context step quotes a sizeable chunk of the memory body
    (the assistant often drafts the note in text right before writing it —
    keeping that step would leak the target into the input)."""
    for start in range(0, max(1, len(step_content) - window), window // 2):
        if step_content[start : start + window] in target:
            return True
    return False


def extract_memory_pairs(
    path: str | Path,
    context_steps: int = 30,
    max_step_chars: int = 1500,
    min_memory_chars: int = 120,
) -> list[MemoryWritePair]:
    """Mine (preceding context -> written memory) pairs from one session."""
    path = Path(path)
    session_id = path.stem
    pairs: list[MemoryWritePair] = []
    task = ""
    window: list[Step] = []

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or rec.get("isSidechain"):
                continue
            if rec.get("type") not in ("user", "assistant"):
                continue
            message = rec.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")

            if rec.get("type") == "user":
                human = _human_text(content)
                if human is not None:
                    if not (INTERRUPT_RE.match(human) or CONTINUATION_RE.match(human)):
                        if not (task and CORRECTION_RE.search(human)):
                            task, window = human, []
                        else:
                            window.append(Step(role="user", content=human))
                    continue
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            window.append(Step(role="tool", content=_render_tool_result(block)))
                continue

            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text", "").strip():
                    window.append(Step(role="assistant", content=block["text"]))
                elif block.get("type") == "tool_use":
                    inp = block.get("input") or {}
                    file_path = str(inp.get("file_path", ""))
                    memory = str(inp.get("content") or "")
                    if (
                        block.get("name") == "Write"
                        and MEMORY_FILE_RE.search(file_path)
                        and len(memory) >= min_memory_chars
                    ):
                        memory = redact_secrets(memory)
                        context = [
                            Step(
                                role=s.role,
                                content=_truncate(redact_secrets(s.content), max_step_chars),
                            )
                            for s in window[-context_steps:]
                            if not _overlaps_target(s.content, memory)
                        ]
                        pairs.append(
                            MemoryWritePair(
                                pair_id=f"mem_{session_id[:8]}_{len(pairs):03d}",
                                session_id=session_id,
                                file_name=Path(file_path).name,
                                task=redact_secrets(_truncate(task, 800)),
                                context=context,
                                memory=memory,
                            )
                        )
                    window.append(Step(role="assistant", content=_render_tool_use(block)))
    return pairs


def load_memory_pairs(
    root: str | Path,
    include_subagents: bool = False,
    context_steps: int = 30,
) -> list[MemoryWritePair]:
    """Mine memory pairs from every session under ``root``."""
    root = Path(root).expanduser()
    pairs: list[MemoryWritePair] = []
    for path in sorted(root.rglob("*.jsonl")):
        if not include_subagents and "subagents" in path.parts:
            continue
        pairs.extend(extract_memory_pairs(path, context_steps=context_steps))
    return pairs


def load_cc_episodes(
    root: str | Path,
    include_subagents: bool = False,
    min_steps: int = 3,
    max_steps: int = 80,
    max_step_chars: int = 1500,
    limit: int | None = None,
) -> list[Episode]:
    """Load every session transcript under ``root`` (live dir or archive)."""
    root = Path(root).expanduser()
    episodes: list[Episode] = []
    for path in sorted(root.rglob("*.jsonl")):
        if not include_subagents and "subagents" in path.parts:
            continue
        if path.name == "manifest.json":  # pragma: no cover - jsonl glob excludes it
            continue
        episodes.extend(
            load_cc_session(
                path, min_steps=min_steps, max_steps=max_steps, max_step_chars=max_step_chars
            )
        )
        if limit is not None and len(episodes) >= limit:
            return episodes[:limit]
    return episodes
