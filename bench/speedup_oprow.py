#!/usr/bin/env python3
"""
Speedup of the RVV indexed kernel over scalar as a function of mean intermediate
products per row (Op). Excludes synthetic gen_ matrices and failed-validation rows.

Usage:
    python3 speedup_vs_oprow.py [summary.csv]
"""

import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "summary.csv"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "figure.dpi": 150,
})

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        rows.append(r)

real = [r for r in rows if not r["matrix"].startswith("gen_")]

pts = []
for r in real:
    if r["kernel"] == "rvv_f32" and r["threads"] == "1" and r["status"] == "ok":
        op = fnum(r["op_mean"])
        s = fnum(r["speedup_vs_gc_scalar"])
        if op is not None and s is not None:
            pts.append((op, s))

pts.sort()
op = np.array([p[0] for p in pts])
spd = np.array([p[1] for p in pts])

fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8, zorder=1)
ax.scatter(op, spd, s=20, color="#1f4e79", edgecolor="white", linewidth=0.4, zorder=3)

ax.set_xscale("log")
ax.set_xlabel(r"Mean intermediate products per row ($\overline{Op}$)")
ax.set_ylabel("Speedup over scalar")
ax.set_ylim(0.85, 1.05)
ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.3)
fig.savefig("speedup_vs_oprow.pdf", bbox_inches="tight")
fig.savefig("speedup_vs_oprow.png", bbox_inches="tight", dpi=200)

print(f"indexed: mean={spd.mean():.3f} min={spd.min():.3f} max={spd.max():.3f} n={len(spd)}")
print("wrote speedup_vs_oprow.pdf and .png")