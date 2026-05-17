"""
Agentic tool-use benchmark.

A small in-house tool-use eval inspired by the Berkeley Function-Calling
Leaderboard (BFCL) and ToolBench.  The model is given a set of mock
domain tools (weather, currency, calendar, sql) registered as RLM
``custom_tools`` and asked to compose them to answer multi-step queries.

Grading is based on whether the final answer matches the gold value
(comparison is case- and whitespace-insensitive for strings, exact for
numbers within a small tolerance).

This benchmark does *not* need internet access — all tools are mocked.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any

from rlm.utils.exceptions import (
    BudgetExceededError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from scripts.benchmarks.common import (
    BenchmarkResult,
    add_common_args,
    build_rlm,
)

# ---------------------------------------------------------------------------
# Mock tools
# ---------------------------------------------------------------------------

_WEATHER_DB = {
    "san francisco": {"temp_f": 62, "condition": "foggy"},
    "new york": {"temp_f": 41, "condition": "snowy"},
    "tokyo": {"temp_f": 55, "condition": "clear"},
    "london": {"temp_f": 48, "condition": "rainy"},
    "sydney": {"temp_f": 78, "condition": "sunny"},
}


def get_weather(city: str) -> dict[str, Any]:
    """Return mock weather data for a city."""
    return _WEATHER_DB.get(city.lower(), {"error": f"unknown city: {city}"})


_FX_RATES = {
    ("USD", "EUR"): 0.92,
    ("USD", "JPY"): 152.0,
    ("USD", "GBP"): 0.78,
    ("EUR", "USD"): 1.09,
    ("EUR", "GBP"): 0.85,
    ("GBP", "USD"): 1.28,
    ("JPY", "USD"): 0.0066,
}


def convert_currency(amount: float, from_ccy: str, to_ccy: str) -> float:
    """Convert ``amount`` between currencies using the mock FX table."""
    if from_ccy == to_ccy:
        return float(amount)
    rate = _FX_RATES.get((from_ccy.upper(), to_ccy.upper()))
    if rate is None:
        raise ValueError(f"No rate for {from_ccy}->{to_ccy}")
    return float(amount) * rate


_CALENDAR: dict[str, list[dict[str, Any]]] = {
    "alice": [
        {"day": "monday", "time": "10:00", "title": "design review"},
        {"day": "tuesday", "time": "14:00", "title": "1:1 with manager"},
        {"day": "wednesday", "time": "09:00", "title": "all hands"},
    ],
    "bob": [
        {"day": "monday", "time": "11:00", "title": "standup"},
        {"day": "tuesday", "time": "16:00", "title": "deploy review"},
        {"day": "thursday", "time": "13:00", "title": "interview"},
    ],
}


def list_calendar(user: str) -> list[dict[str, Any]]:
    """Return all meetings on a user's calendar."""
    return list(_CALENDAR.get(user.lower(), []))


def add_calendar(user: str, day: str, time: str, title: str) -> dict[str, Any]:
    """Add a meeting and return the new entry."""
    entry = {"day": day.lower(), "time": time, "title": title}
    _CALENDAR.setdefault(user.lower(), []).append(entry)
    return entry


_ORDERS = [
    {"id": 1, "customer": "alice", "total": 120.0, "status": "shipped"},
    {"id": 2, "customer": "bob", "total": 45.5, "status": "pending"},
    {"id": 3, "customer": "alice", "total": 250.0, "status": "pending"},
    {"id": 4, "customer": "carol", "total": 99.99, "status": "shipped"},
    {"id": 5, "customer": "bob", "total": 1000.0, "status": "shipped"},
]


def query_orders(customer: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """Filter the orders table by customer and/or status."""
    out = list(_ORDERS)
    if customer is not None:
        out = [o for o in out if o["customer"] == customer.lower()]
    if status is not None:
        out = [o for o in out if o["status"] == status.lower()]
    return out


CUSTOM_TOOLS = {
    "get_weather": {
        "tool": get_weather,
        "description": "Return current weather for a city: {temp_f, condition} or {error}.",
    },
    "convert_currency": {
        "tool": convert_currency,
        "description": "Convert ``amount`` from ``from_ccy`` to ``to_ccy`` using mock FX rates.",
    },
    "list_calendar": {
        "tool": list_calendar,
        "description": "List all meetings on a user's calendar.",
    },
    "add_calendar": {
        "tool": add_calendar,
        "description": "Add a meeting to a user's calendar.  Returns the new entry.",
    },
    "query_orders": {
        "tool": query_orders,
        "description": "Filter the orders table by customer and/or status.",
    },
}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass
class ToolUseTask:
    prompt: str
    gold: Any
    grader: str  # 'number', 'string', 'set'


TASKS: list[ToolUseTask] = [
    ToolUseTask(
        prompt=(
            "Use the available tools to find out the temperature in Tokyo in "
            "degrees Fahrenheit.  Set answer['content'] to just the number."
        ),
        gold=55,
        grader="number",
    ),
    ToolUseTask(
        prompt=(
            "Convert 100 USD to EUR using the provided FX tool.  Set "
            "answer['content'] to the numeric amount of EUR (2 decimal places)."
        ),
        gold=92.0,
        grader="number",
    ),
    ToolUseTask(
        prompt=(
            "How many meetings does Alice have on her calendar this week? "
            "Set answer['content'] to just the integer count."
        ),
        gold=3,
        grader="number",
    ),
    ToolUseTask(
        prompt=(
            "Compute the total amount of all of Alice's orders (any status). "
            "Set answer['content'] to the numeric total in USD."
        ),
        gold=370.0,
        grader="number",
    ),
    ToolUseTask(
        prompt=(
            "List the unique customer names that have at least one pending "
            "order.  Set answer['content'] to a comma-separated list, "
            "alphabetised."
        ),
        gold="alice, bob",
        grader="set",
    ),
    ToolUseTask(
        prompt=(
            "Convert the total of Bob's shipped orders from USD to GBP. "
            "Set answer['content'] to the numeric amount in GBP, rounded "
            "to 2 decimals."
        ),
        gold=1000.0 * 0.78,
        grader="number",
    ),
    ToolUseTask(
        prompt=(
            "Among San Francisco, New York and Sydney, which city is "
            "warmest?  Set answer['content'] to just the city name "
            "(capitalised)."
        ),
        gold="Sydney",
        grader="string",
    ),
    ToolUseTask(
        prompt=(
            "Add a meeting titled 'titans review' to Alice's calendar on "
            "friday at 15:00.  Then return the total number of meetings "
            "Alice has via answer['content'] as an integer."
        ),
        gold=4,
        grader="number",
    ),
]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(text: str) -> float | None:
    if text is None:
        return None
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def grade(task: ToolUseTask, response: str) -> bool:
    if task.grader == "number":
        pred = _extract_number(response or "")
        if pred is None:
            return False
        gold = float(task.gold)
        return math.isclose(pred, gold, rel_tol=1e-3, abs_tol=1e-2)
    if task.grader == "string":
        if response is None:
            return False
        return str(task.gold).strip().lower() in response.strip().lower()
    if task.grader == "set":
        if response is None:
            return False
        gold = {x.strip().lower() for x in str(task.gold).split(",")}
        pred = {x.strip().lower() for x in response.split(",") if x.strip()}
        return gold == pred
    raise ValueError(f"Unknown grader: {task.grader}")


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic tool-use benchmark.")
    add_common_args(p)
    args = p.parse_args()

    rng = random.Random(args.seed)
    tasks = list(TASKS)
    rng.shuffle(tasks)
    tasks = tasks[: args.num_samples]

    result = BenchmarkResult(
        name="tool_use",
        backend=args.backend,
        model=args.model or "",
        n_examples=len(tasks),
        n_correct=0,
    )

    start = time.time()
    for i, task in enumerate(tasks):
        t0 = time.time()
        response = ""
        err = None
        completion = None
        try:
            # Build a fresh RLM per task so the persistent memory in
            # titans_hf doesn't bleed between tasks.
            rlm_t = build_rlm(args, custom_tools=CUSTOM_TOOLS)
            completion = rlm_t.completion(task.prompt)
            response = completion.response or ""
        except (
            BudgetExceededError,
            TimeoutExceededError,
            TokenLimitExceededError,
            ErrorThresholdExceededError,
        ) as e:
            err = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        elapsed = time.time() - t0
        correct = err is None and grade(task, response)
        result.per_example.append(
            {
                "index": i,
                "prompt": task.prompt,
                "gold": task.gold,
                "response": response,
                "correct": correct,
                "elapsed_seconds": elapsed,
                "error": err,
            }
        )
        if correct:
            result.n_correct += 1
        if err is not None:
            result.n_errors += 1
        if completion is not None and completion.usage_summary is not None:
            result.total_input_tokens += completion.usage_summary.total_input_tokens
            result.total_output_tokens += completion.usage_summary.total_output_tokens
        print(
            f"[{i + 1}/{len(tasks)}] correct={correct} gold={task.gold!r} "
            f"got={response[:80]!r} ({elapsed:.1f}s)"
        )

    result.total_time_seconds = time.time() - start
    print(
        f"\n=== tool_use accuracy={result.accuracy:.3f} "
        f"({result.n_correct}/{result.n_examples}) "
        f"errors={result.n_errors} ===",
    )
    path = result.save(args.output_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
