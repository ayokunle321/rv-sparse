#!/usr/bin/env python3
"""Backend stall fraction vs accumulator/L2 ratio.
Usage: fig_stall_vs_ratio.py <summary.csv> <accumulator_ratio.csv>"""
import os
import sys, pandas as pd, matplotlib.pyplot as plt

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

sc = S[(S.arm == "baseline") & (S.threads == 1)]
rv = S[(S.kernel == "rvv_f32_m1") & (S.threads == 1)]

fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.scatter(sc.acc_over_l2, sc.backend_stall_frac, s=30, marker="o",
           facecolors="none", edgecolors="C0", linewidths=0.9, label="scalar")
ax.scatter(rv.acc_over_l2, rv.backend_stall_frac, s=30, marker="^",
           color="C3", label="RVV (LMUL 1)")
ax.axvline(1.0, color="k", lw=0.7, ls=":")
ax.text(1.05, 0.05, "accumulator = L2", fontsize=7, rotation=90, va="bottom")
ax.set_xscale("log")
ax.set_xlabel("accumulator size / L2 cache")
ax.set_ylabel("backend stall fraction")
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, loc="lower right")
fig.savefig(FIGDIR + "/stall_vs_ratio.pdf")
print("wrote " + FIGDIR + "/stall_vs_ratio.pdf")