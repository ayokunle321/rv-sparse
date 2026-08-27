#!/usr/bin/env python3
"""LMUL sweep: m1/m2/m4 speedup over scalar per matrix.
Usage: fig_lmul.py <summary.csv> <vector_fill.csv>"""
import os
import sys, pandas as pd, numpy as np, matplotlib.pyplot as plt

DATA = "paper/data"
FIGDIR = "paper/figures"
os.makedirs(FIGDIR, exist_ok=True)

ARG1 = sys.argv[1] if len(sys.argv) > 1 else DATA + "/summary.csv"
ARG2 = sys.argv[2] if len(sys.argv) > 2 else DATA + "/vector_fill.csv"


plt.rcParams.update({"font.size": 9, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4, "savefig.bbox": "tight"})

S = pd.read_csv(ARG1)
vf = pd.read_csv(ARG2)[["matrix", "exec_weighted_mean_row"]]
S = S.merge(vf, on="matrix")
S = S[S.threads == 1]

order = S[S.kernel == "rvv_f32_m1"].sort_values("exec_weighted_mean_row").matrix.tolist()
kernels = [("rvv_f32_m1", "LMUL 1", "C0"),
           ("rvv_f32_m2", "LMUL 2", "C1"),
           ("rvv_f32_m4", "LMUL 4", "C3")]

x = np.arange(len(order))
w = 0.26
fig, ax = plt.subplots(figsize=(7.5, 3.0))
for i, (k, lab, c) in enumerate(kernels):
    d = S[S.kernel == k].set_index("matrix").reindex(order)
    ax.bar(x + (i - 1) * w, d.speedup_vs_baseline, w, label=lab, color=c, alpha=0.85)
ax.axhline(1.0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=60, ha="right", fontsize=6.5)
ax.set_ylabel("speedup over scalar")
ax.set_ylim(0.55, 1.10)
ax.legend(frameon=False, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.02))
ax.margins(x=0.01)
fig.savefig(FIGDIR + "/lmul.pdf")
print("wrote " + FIGDIR + "/lmul.pdf")