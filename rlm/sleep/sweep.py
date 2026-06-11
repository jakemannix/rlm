"""Hyperparameter sweeps over the nightly cycle (Colab-friendly: sequential).

A sweep grid is a dict of dotted config keys to candidate values, e.g.::

    grid = {
        "adapter.lr": [1e-4, 2e-4, 5e-4],
        "adapter.rank": [8, 16, 32],
        "gate.budget_fraction": [0.1, 0.25, 0.5],
    }

Each point runs one nightly cycle on the same day of episodes and is
scored on next-day / test NLL deltas (improvement vs. base) and the
retention delta (forgetting guard).  Results land in a CSV.
"""

from __future__ import annotations

import csv
import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.sleep.config import SleepConfig
from rlm.sleep.loop import EvalFn, TrainFn, default_eval_fn, default_train_fn, run_night
from rlm.sleep.types import DayResult, Episode

SWEEP_FIELDS = [
    "run",
    "n_examples",
    "next_day_nll_base",
    "next_day_nll_adapted",
    "next_day_delta",
    "test_nll_base",
    "test_nll_adapted",
    "test_delta",
    "retention_delta",
]


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a dotted-key grid, as override dicts."""
    keys = sorted(grid.keys())
    return [
        dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))
    ]


def result_row(run_name: str, result: DayResult) -> dict[str, Any]:
    base, adapted = result.base_report, result.adapted_report
    if base is None or adapted is None:
        return {"run": run_name, "n_examples": result.n_examples}
    return {
        "run": run_name,
        "n_examples": result.n_examples,
        "next_day_nll_base": round(base.next_day_nll, 4),
        "next_day_nll_adapted": round(adapted.next_day_nll, 4),
        "next_day_delta": round(adapted.next_day_nll - base.next_day_nll, 4),
        "test_nll_base": round(base.test_nll, 4),
        "test_nll_adapted": round(adapted.test_nll, 4),
        "test_delta": round(adapted.test_nll - base.test_nll, 4),
        "retention_delta": round(adapted.retention_nll - base.retention_nll, 4),
    }


def run_sweep(
    grid: dict[str, list[Any]],
    base_config: SleepConfig,
    day_episodes: list[Episode],
    next_day_episodes: list[Episode],
    test_episodes: list[Episode],
    judge_lm: BaseLM,
    out_dir: str | Path,
    train_fn: TrainFn = default_train_fn,
    eval_fn: EvalFn = default_eval_fn,
    progress: Callable[[str], None] = print,
) -> Path:
    """Run every grid point sequentially; write sweep_results.csv and return it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = expand_grid(grid)
    rows = []
    for i, overrides in enumerate(points):
        run_name = "_".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(overrides.items()))
        progress(f"[sweep {i + 1}/{len(points)}] {run_name}")
        config = base_config.with_overrides(overrides)
        result = run_night(
            day_index=0,
            day_episodes=day_episodes,
            next_day_episodes=next_day_episodes,
            test_episodes=test_episodes,
            config=config,
            judge_lm=judge_lm,
            out_dir=out_dir / f"point_{i:03d}",
            train_fn=train_fn,
            eval_fn=eval_fn,
            # Shared across points: grid points with identical gate/judge
            # settings pay for the judge exactly once.
            judge_cache_path=out_dir / "judge_cache.json",
        )
        rows.append(result_row(run_name, result) | overrides)

    fieldnames = SWEEP_FIELDS + sorted(grid.keys())
    csv_path = out_dir / "sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    scored = [r for r in rows if "next_day_delta" in r]
    if scored:
        progress("\n=== leaderboard (most negative next_day_delta = best) ===")
        for r in sorted(scored, key=lambda r: r["next_day_delta"])[:10]:
            progress(
                f"  {r['run']}: next_day {r['next_day_delta']:+.4f}  "
                f"test {r['test_delta']:+.4f}  retention {r['retention_delta']:+.4f}"
            )
    return csv_path
