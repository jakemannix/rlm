"""Loading, normalizing, splitting, and "day"-partitioning agentic traces.

Two sources:

* ``load_hf_traces(...)`` — HuggingFace datasets of agent trajectories.
  Known formats are normalized to :class:`Episode`:
    - ``conversations`` lists with ``from``/``value`` keys
      (THUDM/AgentInstruct and similar AgentTuning-style sets),
    - ``messages`` lists with ``role``/``content`` keys.
* ``synthetic_traces(...)`` — deterministic offline generator used by the
  test suite and for dry runs without network access.  It plants the
  failure / retry / surprise patterns the gate is supposed to find, and
  records the ground truth in ``episode.meta["planted_learnworthy"]``.

Splits are deterministic by episode-id hash so train/val/test membership
is stable across runs and machines.  Days are sequential chunks of the
*train* stream: the model trained on day N is evaluated on day N+1
(forward transfer) and on the held-out test split (generalization).
"""

from __future__ import annotations

import hashlib
import random

from rlm.sleep.types import Episode, Step

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "function_call": "assistant",
    "observation": "tool",
    "tool": "tool",
}


def normalize_record(record: dict, source: str, index: int) -> Episode:
    """Normalize one raw dataset record into an :class:`Episode`."""
    if "conversations" in record:
        raw_turns = [(t["from"], t["value"]) for t in record["conversations"]]
    elif "messages" in record:
        raw_turns = [(t["role"], t["content"]) for t in record["messages"]]
    else:
        raise ValueError(f"Unrecognized trace record keys: {sorted(record.keys())}")

    steps = [Step(role=ROLE_MAP[role], content=content) for role, content in raw_turns]
    task = next((s.content for s in steps if s.role == "user"), "")
    episode_id = record.get("id") or f"{source}-{index}"
    return Episode(
        episode_id=str(episode_id),
        source=source,
        task=task[:500],
        steps=steps,
        outcome="unknown",
    )


def load_hf_traces(
    dataset_name: str = "THUDM/AgentInstruct",
    config_name: str | None = "os",
    split: str = "train",
    limit: int | None = None,
) -> list[Episode]:
    """Load and normalize agentic traces from the HuggingFace Hub.

    Requires the ``sleep`` extra (``datasets``).  AgentInstruct exposes its
    domains (os, db, alfworld, webshop, kg, mind2web) as configs.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, config_name, split=split)
    episodes = []
    for i, record in enumerate(ds):
        if limit is not None and i >= limit:
            break
        episodes.append(normalize_record(dict(record), source=dataset_name, index=i))
    return episodes


# ---------------------------------------------------------------------------
# Synthetic traces (offline; used by tests and dry runs)
# ---------------------------------------------------------------------------

TASKS = [
    "List all files larger than 1MB under /var/log and report the largest.",
    "Find the user with the most orders in the orders table.",
    "Book the cheapest direct flight from SFO to JFK next Tuesday.",
    "Summarize the open critical bugs assigned to team-infra.",
    "Rename all .jpeg files in ~/photos to .jpg preserving timestamps.",
    "Compute the median latency from service logs for the last hour.",
]


def synthetic_traces(n: int, seed: int = 0) -> list[Episode]:
    """Generate deterministic synthetic agent episodes.

    Roughly 30% of episodes are "learnworthy": they contain tool errors,
    retries, or outright failure.  The remainder are smooth successes.
    """
    rng = random.Random(seed)
    episodes = []
    for i in range(n):
        task = TASKS[i % len(TASKS)]
        learnworthy = rng.random() < 0.3
        steps: list[Step] = [Step("user", f"TASK: {task}")]
        if learnworthy:
            flavor = rng.choice(["error_retry", "failure", "long_struggle"])
            bad_cmd = f"run_tool(arg={rng.randint(10, 99)})"
            steps.append(Step("assistant", f"I'll start with {bad_cmd}."))
            steps.append(Step("tool", "Error: invalid argument\nTraceback (most recent call last)"))
            if flavor == "error_retry":
                steps.append(Step("assistant", f"I'll start with {bad_cmd}."))
                steps.append(Step("tool", "Error: invalid argument"))
                steps.append(
                    Step("assistant", "Switching approach: validating the argument first.")
                )
                steps.append(Step("tool", "ok"))
                steps.append(Step("assistant", f"Done. The answer for '{task[:30]}' is 42."))
                outcome = "success"
            elif flavor == "failure":
                steps.append(Step("assistant", "I cannot complete the task."))
                outcome = "failure"
            else:
                for k in range(6):
                    steps.append(Step("assistant", f"Trying variation {k} of the command."))
                    steps.append(Step("tool", "partial output, still missing fields"))
                steps.append(Step("assistant", f"Eventually solved '{task[:30]}' with a script."))
                outcome = "success"
        else:
            steps.append(Step("assistant", "Plan: one direct tool call should suffice."))
            steps.append(Step("tool", "ok: result ready"))
            steps.append(Step("assistant", f"Completed '{task[:30]}'. Result: all good."))
            outcome = "success"
        episodes.append(
            Episode(
                episode_id=f"syn-{seed}-{i}",
                source="synthetic",
                task=task,
                steps=steps,
                outcome=outcome,
                meta={"planted_learnworthy": learnworthy},
            )
        )
    return episodes


# ---------------------------------------------------------------------------
# Splits and day partitioning
# ---------------------------------------------------------------------------


def split_episodes(
    episodes: list[Episode],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[list[Episode], list[Episode], list[Episode]]:
    """Deterministic train/val/test split by episode-id hash.

    Membership depends only on ``episode_id``, so it is stable across runs,
    machines, and dataset orderings — the leakage guard in
    ``rlm.sleep.dataset`` relies on this.
    """
    if not 0 < train_frac + val_frac < 1:
        raise ValueError("train_frac + val_frac must lie in (0, 1)")
    train, val, test = [], [], []
    for ep in episodes:
        digest = hashlib.sha256(ep.episode_id.encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        if u < train_frac:
            train.append(ep)
        elif u < train_frac + val_frac:
            val.append(ep)
        else:
            test.append(ep)
    return train, val, test


def partition_days(episodes: list[Episode], episodes_per_day: int) -> list[list[Episode]]:
    """Chunk a stream of episodes into sequential "days"."""
    if episodes_per_day < 1:
        raise ValueError("episodes_per_day must be >= 1")
    return [episodes[i : i + episodes_per_day] for i in range(0, len(episodes), episodes_per_day)]
