"""Plot the model-quality ladder: where is the curation-quality elbow?

Reads ladder_report.json and draws classification quality (balanced
accuracy, gold-tier agreement) and generation quality (coverage,
equivalence) across the model ladder, ordered by classification skill.

    python scripts/sleep/plot_ladder.py --out docs/figures/model_ladder_elbow.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="runs/sleep/model_ladder/ladder_report.json")
    parser.add_argument("--out", default="docs/figures/model_ladder_elbow.png")
    parser.add_argument("--csv", default="docs/data/model_ladder.csv")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    rows = []
    for model, entry in report.items():
        if "error" in entry:
            continue
        c, g = entry.get("classify", {}), entry.get("generate", {})
        if c.get("balanced_accuracy") is None:
            continue
        rows.append(
            {
                "model": model,
                "balanced_acc": c.get("balanced_accuracy"),
                "tier_agree": c.get("gold_tier_agreement"),
                "coverage": g.get("coverage"),
                "equiv": g.get("equivalent_rate"),
            }
        )
    rows.sort(key=lambda r: r["balanced_acc"])

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["model", "balanced_acc", "tier_agree", "coverage", "equiv"]
        )
        w.writeheader()
        w.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["model"].split("/")[-1] for r in rows]
    x = range(len(rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(x, [r["balanced_acc"] for r in rows], "o-", label="balanced accuracy")
    ax1.plot(x, [r["tier_agree"] for r in rows], "s--", alpha=0.7, label="gold-tier agreement")
    ax1.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax1.set_ylabel("score"), ax1.set_title("Curation JUDGING (classify)"), ax1.legend()
    ax1.set_ylim(0.4, 1.0)

    cov = [r["coverage"] / 5.0 if r["coverage"] else None for r in rows]
    ax2.plot(x, cov, "o-", label="coverage (/5)")
    ax2.plot(x, [r["equiv"] for r in rows], "s--", alpha=0.7, label="equivalence rate")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    (
        ax2.set_ylabel("score"),
        ax2.set_title("Curation GENERATION (vs frontier reference)"),
        ax2.legend(),
    )
    ax2.set_ylim(0.0, 1.0)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} and {args.csv} ({len(rows)} rungs)")


if __name__ == "__main__":
    main()
