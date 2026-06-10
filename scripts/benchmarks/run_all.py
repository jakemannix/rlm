"""
Run all four bench scripts back-to-back and print a single summary table.

This is the convenience runner used by the Colab notebook and CI smoke
tests.  It executes each bench as a subprocess so a failure in one bench
doesn't take down the whole run.

Example
-------
    python scripts/benchmarks/run_all.py --backend titans_hf \
        --model google/gemma-3-1b-it --num-samples 5

    python scripts/benchmarks/run_all.py --backend openai \
        --model gpt-5-nano --num-samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BENCHES = [
    ("reasoning", "scripts.benchmarks.bench_reasoning"),
    ("tool_use", "scripts.benchmarks.bench_tool_use"),
    ("codegen", "scripts.benchmarks.bench_codegen"),
    ("long_context", "scripts.benchmarks.bench_long_context"),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Run all Titans benchmarks.")
    p.add_argument("--backend", default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--memory-flavor", choices=["titans", "miras"], default="titans")
    p.add_argument("--retention", choices=["l2", "l1", "huber", "kl"], default="l2")
    p.add_argument("--output-dir", default="./bench_results")
    p.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Bench names to skip (may be passed multiple times).",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=8,
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    summaries: dict[str, dict] = {}
    repo_root = Path(__file__).resolve().parents[2]

    for name, module in BENCHES:
        if name in args.skip:
            print(f"--- skipping {name} ---")
            continue
        cmd = [
            sys.executable,
            "-m",
            module,
            "--backend",
            args.backend,
            "--num-samples",
            str(args.num_samples),
            "--memory-flavor",
            args.memory_flavor,
            "--retention",
            args.retention,
            "--output-dir",
            args.output_dir,
            "--max-iterations",
            str(args.max_iterations),
            "--seed",
            str(args.seed),
        ]
        if args.model is not None:
            cmd += ["--model", args.model]
        if args.base_url is not None:
            cmd += ["--base-url", args.base_url]
        print("\n===", " ".join(cmd), "===")
        res = subprocess.run(cmd, cwd=repo_root, env={**os.environ, "PYTHONPATH": str(repo_root)})
        if res.returncode != 0:
            summaries[name] = {"error": f"exit code {res.returncode}"}
            continue
        # Find the latest result file for this bench.
        out_dir = Path(args.output_dir).resolve()
        if not out_dir.exists():
            summaries[name] = {"error": "no results dir"}
            continue
        bench_prefix_map = {
            "reasoning": "reasoning_gsm8k",
            "tool_use": "tool_use",
            "codegen": "codegen_humaneval",
            "long_context": "long_context_niah",
        }
        prefix = bench_prefix_map[name]
        candidates = sorted(out_dir.glob(f"{prefix}-*.json"))
        if not candidates:
            summaries[name] = {"error": "no result file"}
            continue
        with open(candidates[-1]) as f:
            data = json.load(f)
        summaries[name] = {
            "accuracy": data.get("n_correct", 0) / max(data.get("n_examples", 1), 1),
            "n_examples": data.get("n_examples"),
            "n_correct": data.get("n_correct"),
            "n_errors": data.get("n_errors"),
            "total_time_seconds": data.get("total_time_seconds"),
            "result_file": str(candidates[-1]),
        }

    print("\n=========================================")
    print("Summary")
    print("=========================================")
    for name, s in summaries.items():
        if "error" in s:
            print(f"  {name:14s}  ERROR: {s['error']}")
        else:
            print(
                f"  {name:14s}  acc={s['accuracy']:.3f}  "
                f"({s['n_correct']}/{s['n_examples']})  "
                f"errs={s['n_errors']}  time={s['total_time_seconds']:.1f}s"
            )

    summary_path = Path(args.output_dir) / "summary.json"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "backend": args.backend,
                "model": args.model,
                "memory_flavor": args.memory_flavor,
                "retention": args.retention,
                "benches": summaries,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
