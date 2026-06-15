"""Sweep a model ladder over memory classification + generation via OpenRouter.

Finds the quality elbow across model scales on two tasks against the
frontier-curated gold set (see rlm/sleep/model_ladder.py):

    OPENROUTER_API_KEY=... python scripts/sleep/run_model_ladder.py \
        --models qwen/qwen-2.5-7b-instruct,meta-llama/llama-3.3-70b-instruct \
        --referee anthropic/claude-sonnet-4.5

Model IDs are validated against the live OpenRouter catalog before any
spend; unknown IDs are reported with close matches and skipped. All calls
are disk-cached, so extending the ladder re-uses every prior cell.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import requests

from rlm.sleep.gold import _CallCache
from rlm.sleep.model_ladder import (
    OPENROUTER_BASE_URL,
    ModelResult,
    build_items,
    purge_invalid,
    run_model,
)

# Picked from the LIVE OpenRouter catalog (2026-06) across the size
# spectrum — never from training-data memory; validate() re-checks at
# every launch. Claude rung is the ceiling anchor but carries a
# self-family bias caveat (the gold references are Claude-authored).
DEFAULT_MODELS = [
    # bottom of the curve — small dense + a within-family Ministral 3/8/14 sweep
    "mistralai/ministral-3b-2512",
    "mistralai/ministral-8b-2512",
    "ibm-granite/granite-4.1-8b",
    "qwen/qwen3.5-9b",
    # mid
    "mistralai/ministral-14b-2512",
    "qwen/qwen3.6-27b",
    "qwen/qwen3.5-35b-a3b",
    "qwen/qwen3.6-35b-a3b",
    # large
    "qwen/qwen3.5-122b-a10b",
    "z-ai/glm-5.1",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.7-max",
    # frontier ceiling
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.5-pro",
]


def catalog() -> set[str]:
    response = requests.get(f"{OPENROUTER_BASE_URL}/models", timeout=30)
    response.raise_for_status()
    return {m["id"] for m in response.json()["data"]}


def validate(models: list[str], available: set[str]) -> list[str]:
    valid = []
    for model in models:
        if model in available:
            valid.append(model)
        else:
            close = difflib.get_close_matches(model, available, n=3, cutoff=0.4)
            print(f"SKIPPING unknown model {model!r}; close matches: {close}")
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="comma-separated OpenRouter model ids"
    )
    parser.add_argument(
        "--referee",
        default="google/gemini-3.1-pro-preview",
        help="fixed grader for the generation task (non-Claude by default: "
        "the reference memories are Claude-authored)",
    )
    parser.add_argument("--tasks", default="classify,generate")
    parser.add_argument("--prep", default="runs/sleep/frontier_memories/prep.json")
    parser.add_argument("--authored", default="runs/sleep/frontier_memories/authored.json")
    parser.add_argument("--limit", type=int, default=None, help="cap items (smoke runs)")
    parser.add_argument("--evidence-chars", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="runs/sleep/model_ladder")
    args = parser.parse_args()

    from rlm.clients.openai import OpenAIClient

    prep = json.loads(Path(args.prep).read_text())
    authored = {a["memory_id"]: a for a in json.loads(Path(args.authored).read_text())}
    items = build_items(prep, authored, evidence_chars=args.evidence_chars)
    if args.limit:
        items = items[: args.limit]
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())

    available = catalog()
    models = validate([m.strip() for m in args.models.split(",") if m.strip()], available)
    referee_ids = validate([args.referee], available)
    if not referee_ids and "generate" in tasks:
        raise SystemExit("referee model not found on OpenRouter; pick another --referee")
    print(f"ladder: {len(models)} models x {len(items)} items, tasks={tasks}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _CallCache(out / "ladder_cache.json")
    purged = purge_invalid(cache)
    if purged:
        print(f"purged {purged} poisoned cache entries (empty/None responses)")
    referee = (
        OpenAIClient(model_name=referee_ids[0], base_url=OPENROUTER_BASE_URL)
        if referee_ids
        else None
    )

    report: dict[str, dict] = {}
    for model_id in models:
        lm = OpenAIClient(model_name=model_id, base_url=OPENROUTER_BASE_URL)
        try:
            result: ModelResult = run_model(
                lm, items, cache, tasks=tasks, referee=referee, workers=args.workers
            )
        except Exception as exc:  # noqa: BLE001 - one rung must not kill the sweep
            print(f"=== {model_id} FAILED: {str(exc)[:200]}")
            report[model_id] = {"error": str(exc)[:300]}
            continue
        entry = {}
        if "classify" in tasks:
            entry["classify"] = result.classify_metrics()
        if "generate" in tasks:
            entry["generate"] = result.generate_metrics()
        report[model_id] = entry
        with (out / f"rows_{model_id.replace('/', '_')}.json").open("w") as f:
            json.dump(
                {"classify": result.classify_rows, "generate": result.generate_rows}, f, indent=1
            )
        print(f"\n=== {model_id} ===\n{json.dumps(entry, indent=2)}", flush=True)

    (out / "ladder_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out / 'ladder_report.json'}")


if __name__ == "__main__":
    main()
