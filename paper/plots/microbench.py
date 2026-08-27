#!/usr/bin/env python3
"""Microbenchmark: fill sweep and working-set sweep, scalar vs RVV.
Usage: microbench.py <gather_scatter.csv>"""
import os
import sys, pandas as pd, matplotlib.pyplot as plt

DATA = "paper/data"
FIGDIR = "paper/figures"
os.makedirs(FIGDIR, exist_ok=True)

ARG1 = sys.argv[1] if len(sys.argv) > 1 else DATA + "/gather_scatter.csv"


plt.rcParams.update({"font.size": 9, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4})

D = pd.read_csv(ARG1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# --- left: fill sweep, ns/elem vs VL, in L1 ---
f = D[D.sweep == "fill"]
fsc = f[f.kernel == "scalar"]
frv = f[f.kernel == "rvv"].groupby("swept", as_index=False).ns_per_elem.median()
if len(fsc):
    ax1.axhline(fsc.ns_per_elem.median(), color="C0", lw=1.4, ls="--", label="scalar")
ax1.plot(frv.swept, frv.ns_per_elem, color="C3", lw=1.6, marker="^", ms=5, label="RVV")
ax1.set_xlabel("vector length (elements)")
ax1.set_ylabel("ns per element")
ax1.set_xticks(sorted(frv.swept.unique()))
ax1.legend(frameon=False, loc="upper right")
ax1.set_title("fill sweep (L1-resident)", fontsize=9)

# --- right: working-set sweep, ns/elem vs accumulator size ---
w = D[D.sweep == "wss"]
wsc = w[w.kernel == "scalar"].groupby("swept", as_index=False).ns_per_elem.median()
wrv = w[w.kernel == "rvv"].groupby("swept", as_index=False).ns_per_elem.median()
ax2.plot(wsc.swept, wsc.ns_per_elem, color="C0", lw=1.4, marker="o", ms=4,
         mfc="none", label="scalar")
ax2.plot(wrv.swept, wrv.ns_per_elem, color="C3", lw=1.6, marker="^", ms=5, label="RVV")
ax2.axvline(512, color="k", lw=0.7, ls=":")
ax2.set_xscale("log")
ax2.set_yscale("log")
# anchor the L2 label in axes coords (0-1), not data coords, so log scale can't blow it up
ax2.text(512, 0.03, " L2", transform=ax2.get_xaxis_transform(),
         fontsize=7, va="bottom", ha="left")
ax2.set_xlabel("working set (KB)")
ax2.set_ylabel("ns per element")
ax2.legend(frameon=False, loc="upper left")
ax2.set_title("working-set sweep", fontsize=9)

fig.tight_layout()
fig.savefig(FIGDIR + "/microbench.pdf")
print("wrote " + FIGDIR + "/microbench.pdf")