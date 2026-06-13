"""Plot the Phase-E breadth size-sweep: uptake Δ and blind win-rate vs policy size.

Reads the verified two-judge CSV and emits the figure used in
docs/memory_experiment_status.md (Phase E).

    uv run --with matplotlib python scripts/sleep/plot_breadth_sweep.py
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

CSV = "docs/data/breadth_size_sweep_2026-06-13.csv"
OUT = "docs/figures/breadth_size_sweep.png"

rows = list(csv.DictReader(open(CSV)))
x = [float(r["size_b"]) for r in rows]
lbl = [r["model"] for r in rows]
dq = [float(r["delta_qwen"]) * 100 for r in rows]
dd = [float(r["delta_deepseek"]) * 100 for r in rows]
sq = [float(r["delta_qwen_std"]) * 100 for r in rows]
wq = [float(r["winrate_qwen"]) for r in rows]
wd = [float(r["winrate_deepseek"]) for r in rows]
oq = [float(r["overapply_qwen"]) * 100 for r in rows]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
a1.errorbar(x, dq, yerr=sq, marker="o", lw=2, capsize=4, label="Qwen3.6-35b judge (±seed std)", color="#1f77b4")
a1.plot(x, dd, marker="s", lw=2, ls="--", label="DeepSeek-v4 judge (independent)", color="#ff7f0e")
a1.plot(x, oq, marker="x", lw=1.5, ls=":", color="gray", label="over-application (neg control)")
a1.axhline(0, color="k", lw=0.6)
for xi, yi, label in zip(x, dq, lbl, strict=True):
    a1.annotate(label, (xi, yi), textcoords="offset points", xytext=(4, 7), fontsize=8)
a1.set_xscale("log")
a1.set_xticks(x)
a1.set_xticklabels([r["size"] for r in rows])
a1.set_xlabel("policy size")
a1.set_ylabel("held-out behavior uptake Δ (pp)")
a1.set_title("Memory uptake scales with policy size\n(breadth SFT: 555 distinct memories → 71 held-out probes)")
a1.legend(fontsize=8)
a1.grid(alpha=0.3)

a2.plot(x, wq, marker="o", lw=2, label="Qwen3.6 judge", color="#1f77b4")
a2.plot(x, wd, marker="s", lw=2, ls="--", label="DeepSeek-v4 judge", color="#ff7f0e")
a2.axhline(0.5, color="k", lw=0.6, ls=":")
a2.set_xscale("log")
a2.set_xticks(x)
a2.set_xticklabels([r["size"] for r in rows])
a2.set_xlabel("policy size")
a2.set_ylabel("blind win-rate (adapted vs base)")
a2.set_title("Blind win-rate vs base")
a2.legend(fontsize=8)
a2.grid(alpha=0.3)
a2.set_ylim(0.45, 0.8)

for ax in (a1, a2):
    ax.xaxis.set_minor_locator(NullLocator())
plt.tight_layout()
pathlib.Path("docs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
