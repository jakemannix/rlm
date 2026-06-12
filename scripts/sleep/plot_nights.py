"""Plot multi-night learning curves from run_nightly.py output dirs.

Reads ``day_*/day_result.json`` under each run dir and plots test NLL and
retention NLL across nights for every policy (e.g. independent nights vs
--cumulative re-distill), with the base model as the reference line.

Example:
    python scripts/sleep/plot_nights.py runs/sleep/g4_indep runs/sleep/g4_cumul \
        --labels independent cumulative --out docs/figures/sleep_g4_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_curve(run_dir: Path) -> list[dict]:
    results = []
    for day_dir in sorted(run_dir.glob("day_*")):
        path = day_dir / "day_result.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    if not results:
        raise FileNotFoundError(f"No day_*/day_result.json under {run_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="run_nightly.py --out dirs to compare")
    parser.add_argument("--labels", nargs="*", default=None, help="one label per run dir")
    parser.add_argument("--out", default="sleep_nights.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = args.labels or [Path(d).name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("Need exactly one label per run dir")

    fig, (ax_test, ax_ret) = plt.subplots(1, 2, figsize=(11, 4.2))
    base_drawn = False
    for run_dir, label in zip(args.run_dirs, labels, strict=True):
        curve = load_curve(Path(run_dir))
        nights = [r["day_index"] for r in curve]
        trained = [r for r in curve if r.get("adapted_report")]
        if not base_drawn and trained:
            ax_test.plot(
                [r["day_index"] for r in trained],
                [r["base_report"]["test_nll"] for r in trained],
                "k--",
                alpha=0.6,
                label="base",
            )
            ax_ret.plot(
                [r["day_index"] for r in trained],
                [r["base_report"]["retention_nll"] for r in trained],
                "k--",
                alpha=0.6,
                label="base",
            )
            base_drawn = True
        ax_test.plot(
            [r["day_index"] for r in trained],
            [r["adapted_report"]["test_nll"] for r in trained],
            marker="o",
            label=label,
        )
        ax_ret.plot(
            [r["day_index"] for r in trained],
            [r["adapted_report"]["retention_nll"] for r in trained],
            marker="o",
            label=label,
        )
        skipped = [r["day_index"] for r in curve if not r.get("adapted_report")]
        if skipped:
            print(f"{label}: nights with no adapter (nothing distilled): {skipped}")
        print(f"{label}: nights={nights}")

    ax_test.set_xlabel("night"), ax_test.set_ylabel("test NLL (nats/token)")
    ax_test.set_title("Held-out test NLL"), ax_test.legend()
    ax_ret.set_xlabel("night"), ax_ret.set_ylabel("retention NLL (nats/token)")
    ax_ret.set_title("Retention probe NLL"), ax_ret.legend()
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
