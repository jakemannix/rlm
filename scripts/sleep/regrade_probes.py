"""Re-grade cached probe responses with a stronger (OpenRouter) judge.

The nightly run graded behavior/CoT with the 1.5B base policy — a noisy
grader, so its +9pp uptake number needs an audit. This reads the cached
base/adapted responses from a prior run and re-grades them with any
model, defaulting to the ladder-proven ``qwen/qwen3.6-35b-a3b``. No
retraining; generation is already cached on disk.

    OPENROUTER_API_KEY=... python scripts/sleep/regrade_probes.py \
        --judge-model qwen/qwen3.6-35b-a3b
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlm.sleep.gold import _CallCache
from rlm.sleep.grading import grade_behavior, grade_cot
from rlm.sleep.model_ladder import OPENROUTER_BASE_URL, _NonEmpty, purge_invalid


class _CachedJudge:
    """Wrap a BaseLM so each grading call is disk-cached and non-empty-guarded."""

    def __init__(self, lm, cache: _CallCache, kind: str):
        self._guarded = _NonEmpty(lm)
        self._cache = cache
        self._kind = kind
        self.model_name = lm.model_name
        self.temperature = getattr(lm, "temperature", None)

    def completion(self, prompt: str) -> str:
        return self._cache.completions(self._guarded, self._kind, prompt, 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-results", default="runs/sleep/memory_nights/probe_results.jsonl")
    parser.add_argument(
        "--probes-heldout", default="runs/sleep/frontier_memories/probes_heldout.jsonl"
    )
    parser.add_argument("--judge-model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="runs/sleep/memory_nights")
    args = parser.parse_args()

    from rlm.clients.openai import OpenAIClient

    results = [json.loads(line) for line in Path(args.probe_results).open()]
    # principle (the memory text) + markers come from the held-out probe rows
    probes = {
        p["memory_id"]: p for p in (json.loads(line) for line in Path(args.probes_heldout).open())
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _CallCache(out / f"regrade_cache_{args.judge_model.replace('/', '_')}.json")
    purge_invalid(cache)
    lm = OpenAIClient(model_name=args.judge_model, base_url=OPENROUTER_BASE_URL)
    judge = _CachedJudge(lm, cache, "grade")

    def regrade(row: dict) -> dict:
        probe = probes.get(row["memory_id"], {})
        marker = probe.get("behavior_marker", "")
        principle = probe.get("memory", "")
        out_row = {k: row[k] for k in ("memory_id", "tier", "memory_type")}
        for arm in ("base", "adapted"):
            resp = row[arm]["response"]
            out_row[arm] = {
                "behavior": grade_behavior(judge, probe.get("scenario", ""), resp, marker),
                "cot": grade_cot(judge, resp, principle),
            }
        return out_row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        regraded = list(pool.map(regrade, results))

    def pct(arm: str, key: str) -> str:
        hits = sum(1 for r in regraded if r[arm][key])
        return f"{hits}/{len(regraded)} ({100 * hits / len(regraded):.0f}%)"

    # original 1.5B-grader numbers for side-by-side
    def orig_pct(arm: str, key: str) -> str:
        hits = sum(1 for r in results if r[arm][key])
        return f"{hits}/{len(results)} ({100 * hits / len(results):.0f}%)"

    summary = {
        "judge_model": args.judge_model,
        "n_probes": len(regraded),
        "behavior_uptake": {
            "regraded": {"base": pct("base", "behavior"), "adapted": pct("adapted", "behavior")},
            "original_1.5b": {
                "base": orig_pct("base", "behavior"),
                "adapted": orig_pct("adapted", "behavior"),
            },
        },
        "cot_surfacing": {
            "regraded": {"base": pct("base", "cot"), "adapted": pct("adapted", "cot")},
            "original_1.5b": {
                "base": orig_pct("base", "cot"),
                "adapted": orig_pct("adapted", "cot"),
            },
        },
    }
    (out / "regraded_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "regraded_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in regraded)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
