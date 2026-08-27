#!/usr/bin/env python3
"""RVV LMUL-1 speedup over scalar, per matrix, ordered by execution-weighted VL.
Usage: fig_parity.py <summary.csv> <vector_fill.csv>"""
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

rv = S[(S.kernel == "rvv_f32_m1") & (S.threads == 1)].sort_values("exec_weighted_mean_row")
x = np.arange(len(rv))
err = np.vstack([rv.speedup_vs_baseline - rv.speedup_ci_lo,
                 rv.speedup_ci_hi - rv.speedup_vs_baseline])

fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.bar(x, rv.speedup_vs_baseline, yerr=err, capsize=2, color="C3",
       alpha=0.85, error_kw={"lw": 0.6})
ax.axhline(1.0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(rv.matrix, rotation=60, ha="right", fontsize=6.5)
ax.set_ylabel("speedup over scalar")
ax.set_ylim(0.5, max(1.05, rv.speedup_vs_baseline.max() * 1.05))
ax.margins(x=0.01)
fig.savefig(FIGDIR + "/parity.pdf")
print("wrote " + FIGDIR + "/parity.pdf")