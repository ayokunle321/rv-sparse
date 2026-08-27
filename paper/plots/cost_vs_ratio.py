#!/usr/bin/env python3
"""Cycles per output nonzero vs accumulator/L2 ratio.
Usage: cost_vs_ratio.py <summary.csv> <accumulator_ratio.csv>"""
import os
import sys, pandas as pd, numpy as np, matplotlib.pyplot as plt

DATA = "paper/data"
FIGDIR = "paper/figures"
os.makedirs(FIGDIR, exist_ok=True)

ARG1 = sys.argv[1] if len(sys.argv) > 1 else DATA + "/summary.csv"
ARG2 = sys.argv[2] if len(sys.argv) > 2 else DATA + "/accumulator_ratio.csv"


plt.rcParams.update({"font.size": 9, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4, "savefig.bbox": "tight"})

S = pd.read_csv(ARG1)
acc = pd.read_csv(ARG2)[["matrix", "acc_over_l2"]]
S = S.merge(acc, on="matrix")

sc = S[(S.arm == "baseline") & (S.threads == 1)].dropna(subset=["cycles_per_nnz_c"])
rv = S[(S.kernel == "rvv_f32_m1") & (S.threads == 1)].dropna(subset=["cycles_per_nnz_c"])

fig, ax = plt.subplots(figsize=(4.2, 3.0))
if len(sc):
    ax.scatter(sc.acc_over_l2, sc.cycles_per_nnz_c, s=30, marker="o",
               facecolors="none", edgecolors="C0", linewidths=0.9, label="scalar")
ax.scatter(rv.acc_over_l2, rv.cycles_per_nnz_c, s=30, marker="^",
           color="C3", label="RVV (LMUL 1)")
ax.axvline(1.0, color="k", lw=0.7, ls=":")
ax.text(1.0, 0.02, "acc = L2", transform=ax.get_xaxis_transform(),
        rotation=90, va="bottom", ha="right", fontsize=7)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("accumulator size / L2 cache")
ax.set_ylabel("cycles per output nonzero")
ax.legend(frameon=False, loc="upper left")

# report the correlation in the corner so the trend is stated, not just drawn
r = np.corrcoef(np.log10(rv.acc_over_l2), np.log10(rv.cycles_per_nnz_c))[0, 1]
ax.text(0.97, 0.08, f"RVV  r = {r:.2f}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7)

fig.savefig(FIGDIR + "/cost_vs_ratio.pdf")
print("wrote " + FIGDIR + "/cost_vs_ratio.pdf")