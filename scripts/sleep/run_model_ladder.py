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
from rlm.sleep.model_ladder import OPENROUTER_BASE_URL, ModelResult, build_items, run_model

DEFAULT_MODELS = [
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen2.5-32b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.5",
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
        default="anthropic/claude-sonnet-4.5",
        help="fixed grader for the generation task",
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
    referee = (
        OpenAIClient(model_name=referee_ids[0], base_url=OPENROUTER_BASE_URL)
        if referee_ids
        else None
    )

    report: dict[str, dict] = {}
    for model_id in models:
        lm = OpenAIClient(model_name=model_id, base_url=OPENROUTER_BASE_URL)
        result: ModelResult = run_model(
            lm, items, cache, tasks=tasks, referee=referee, workers=args.workers
        )
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
