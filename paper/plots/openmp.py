#!/usr/bin/env python3
"""OpenMP scaling and stall fraction vs thread count.
Usage: openmp.py <summary.csv>"""
import os
import sys, pandas as pd, numpy as np, matplotlib.pyplot as plt

DATA = "paper/data"
FIGDIR = "paper/figures"
os.makedirs(FIGDIR, exist_ok=True)

ARG1 = sys.argv[1] if len(sys.argv) > 1 else DATA + "/summary.csv"


plt.rcParams.update({"font.size": 9, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4, "savefig.bbox": "tight"})

S = pd.read_csv(ARG1)
omp = S[S.arm == "omp"].copy()
threads = sorted(omp.threads.unique())

def stats(col):
    m  = np.array([omp[omp.threads == t][col].mean() for t in threads])
    lo = np.array([omp[omp.threads == t][col].quantile(0.10) for t in threads])
    hi = np.array([omp[omp.threads == t][col].quantile(0.90) for t in threads])
    return m, lo, hi

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 4.6), sharex=True)

# top: speedup vs 1 thread
m, lo, hi = stats("speedup_vs_1thread")
ax1.fill_between(threads, lo, hi, color="C3", alpha=0.15)
ax1.plot(threads, m, color="C3", lw=1.8, marker="o", ms=4, label="mean")
ax1.plot(threads, threads, color="k", lw=0.7, ls=":", label="ideal")
ax1.set_ylabel("speedup over 1 thread")
ax1.legend(frameon=False, loc="upper left")

# bottom: stall fraction
m, lo, hi = stats("backend_stall_frac")
ax2.fill_between(threads, lo, hi, color="C3", alpha=0.15)
ax2.plot(threads, m, color="C3", lw=1.8, marker="o", ms=4)
ax2.set_ylabel("backend stall fraction")
ax2.set_xlabel("threads")
ax2.set_ylim(0, 1.0)
ax2.set_xticks(threads)

fig.align_ylabels()
fig.savefig(FIGDIR + "/openmp.pdf")
print("wrote " + FIGDIR + "/openmp.pdf")