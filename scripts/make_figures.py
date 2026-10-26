#!/usr/bin/env python3
"""
make_figures.py  -  SpGEMM characterization figures from summary.csv

Usage:
    python3 make_figures.py RISCV_SUMMARY.csv [INTEL_SUMMARY.csv] [--outdir figs]

Reads one or two summary.csv files (the schema written by the rv-sparse
bench driver) and writes poster figures as PNGs plus a matrix table CSV.

Figures:
  1. speedup_wall_riscv.png    RVV m2 vs scalar baseline, 19 matrices, sorted
  2. lmul_degradation.png      m1/m2/m4 vs scalar for representative matrices
  3. speedup_vs_structure.png  RVV m2 speedup vs mean row length (op_mean)
  4. compare_x60_xeon.png      X60 (RVV m2) vs Xeon (AVX-512), per matrix   [only if Intel CSV given]
  5. matrix_table.csv          name, nnz_c, op_mean, op_max for the methodology slide

Nothing is hard-coded about which matrices exist; it uses whatever rows are present.
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- EuroRV-ish palette ----
TEAL   = "#1a9aa6"   # wins / highlight
GREY   = "#6f7071"   # losses / neutral bars
DARK   = "#333333"   # text, axes
LIGHT  = "#d9dcdd"   # gridlines
RED    = "#c0504d"   # below-1.0 emphasis (optional)
XEON_C = "#e0a030"   # Xeon series in comparison

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": DARK,
    "axes.labelcolor": DARK,
    "text.color": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


def load(path):
    df = pd.read_csv(path)
    # normalise: keep only single-thread rows for the vectorization story
    # (omp rows have threads>1; baseline/autovec/intrinsic are threads==1)
    return df


def best_baseline(df):
    """Return the scalar baseline row set (arm=='baseline')."""
    return df[df["arm"] == "baseline"]


def intrinsic_rows(df, lmul=None):
    r = df[df["arm"] == "intrinsic"]
    if lmul is not None:
        r = r[r["lmul"] == lmul]
    return r


def speedup_vs_baseline_by_matrix(df, arm, kernel_contains=None, lmul=None):
    """
    Return dict matrix -> speedup_vs_baseline for the chosen arm/lmul,
    using the CSV's own speedup_vs_baseline column (median-based).
    """
    sub = df[df["arm"] == arm].copy()
    if lmul is not None and "lmul" in sub.columns:
        sub = sub[sub["lmul"] == lmul]
    if kernel_contains is not None:
        sub = sub[sub["kernel"].str.contains(kernel_contains, na=False)]
    out = {}
    for _, row in sub.iterrows():
        out[row["matrix"]] = float(row["speedup_vs_baseline"])
    return out


# --------------------------------------------------------------------------
# Figure 1: speedup wall (RVV m2 vs scalar)
# --------------------------------------------------------------------------
def fig_speedup_wall(df, outdir, lmul=2, label="RVV m2"):
    sp = speedup_vs_baseline_by_matrix(df, "intrinsic", lmul=lmul)
    if not sp:
        print("  [wall] no intrinsic rows for lmul=%s, skipping" % lmul)
        return
    items = sorted(sp.items(), key=lambda kv: kv[1])  # worst first
    names = [k for k, v in items]
    vals  = [v for k, v in items]
    colors = [TEAL if v >= 1.0 else GREY for v in vals]

    fig, ax = plt.subplots(figsize=(8, 7))
    y = range(len(names))
    ax.barh(list(y), vals, color=colors, height=0.72)
    ax.axvline(1.0, color=DARK, linewidth=1.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Speedup vs scalar  (>1 faster, <1 slower)")
    ax.set_title("Indexed %s vs scalar on the in-order X60\n(19 SuiteSparse matrices)" % label,
                 fontsize=12, color=DARK)
    ax.set_xlim(0, max(1.1, max(vals) * 1.05))
    for i, v in zip(y, vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=DARK)
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(outdir, "speedup_wall_riscv.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    nlose = sum(1 for v in vals if v < 1.0)
    print(f"  [wall] wrote {p}  ({nlose}/{len(vals)} below 1.0, "
          f"min {min(vals):.2f}, median {sorted(vals)[len(vals)//2]:.2f})")


# --------------------------------------------------------------------------
# Figure 2: LMUL degradation for representative matrices
# --------------------------------------------------------------------------
def fig_lmul(df, outdir, reps=None):
    have = df[df["arm"] == "intrinsic"]["matrix"].unique().tolist()
    if reps is None:
        # pick matrices that show the clearest monotonic drop if present,
        # else first 6 available
        preferred = ["roadNet-CA", "cit-Patents", "web-Google", "amazon0312",
                     "patents_main", "mario002"]
        reps = [m for m in preferred if m in have][:6]
        if len(reps) < 3:
            reps = have[:6]
    m1 = speedup_vs_baseline_by_matrix(df, "intrinsic", lmul=1)
    m2 = speedup_vs_baseline_by_matrix(df, "intrinsic", lmul=2)
    m4 = speedup_vs_baseline_by_matrix(df, "intrinsic", lmul=4)
    reps = [m for m in reps if m in m1 and m in m2 and m in m4]
    if not reps:
        print("  [lmul] no matrices with all of m1/m2/m4, skipping")
        return

    import numpy as np
    x = np.arange(len(reps))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, [m1[m] for m in reps], width=w, label="LMUL 1", color=TEAL)
    ax.bar(x,     [m2[m] for m in reps], width=w, label="LMUL 2", color=GREY)
    ax.bar(x + w, [m4[m] for m in reps], width=w, label="LMUL 4", color=DARK)
    ax.axhline(1.0, color=DARK, linewidth=1.2, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Speedup vs scalar")
    ax.set_title("Wider vector groups make it worse (LMUL 1 \u2192 4)", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(outdir, "lmul_degradation.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [lmul] wrote {p}  (matrices: {', '.join(reps)})")


# --------------------------------------------------------------------------
# Figure 3: speedup vs matrix structure (row length)
# --------------------------------------------------------------------------
def fig_structure(df, outdir, lmul=2):
    sp = speedup_vs_baseline_by_matrix(df, "intrinsic", lmul=lmul)
    # op_mean is mean work per output row (row-length-ish); grab per matrix
    base = df[df["arm"] == "intrinsic"]
    if lmul is not None:
        base = base[base["lmul"] == lmul]
    xs, ys, labels = [], [], []
    for _, row in base.iterrows():
        m = row["matrix"]
        if m in sp:
            xs.append(float(row["op_mean"]))
            ys.append(sp[m])
            labels.append(m)
    if not xs:
        print("  [structure] no data, skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, color=TEAL, s=40, zorder=3)
    ax.axhline(1.0, color=DARK, linewidth=1.2)
    for xi, yi, li in zip(xs, ys, labels):
        ax.annotate(li, (xi, yi), fontsize=7, xytext=(3, 3),
                    textcoords="offset points", color=DARK)
    ax.set_xscale("log")
    ax.set_xlabel("Mean work per output row  (op_mean, log scale)")
    ax.set_ylabel("RVV m%d speedup vs scalar" % (lmul or 2))
    ax.set_title("Does the slowdown track matrix structure?", fontsize=12)
    ax.grid(True, color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(outdir, "speedup_vs_structure.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [structure] wrote {p}")


# --------------------------------------------------------------------------
# Figure 4: X60 vs Xeon comparison (needs Intel CSV)
# --------------------------------------------------------------------------
def fig_compare(dr, di, outdir):
    # X60: intrinsic lmul 2 ; Xeon: intrinsic (avx512, no lmul)
    x60 = speedup_vs_baseline_by_matrix(dr, "intrinsic", lmul=2)
    xeon = speedup_vs_baseline_by_matrix(di, "intrinsic")  # avx512, lmul blank
    common = [m for m in x60 if m in xeon]
    if not common:
        print("  [compare] no shared matrices, skipping")
        return
    common = sorted(common, key=lambda m: x60[m])
    import numpy as np
    x = np.arange(len(common))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(x - w/2, [x60[m]  for m in common], height=w, color=TEAL,   label="X60 (RVV m2)")
    ax.barh(x + w/2, [xeon[m] for m in common], height=w, color=XEON_C, label="Xeon (AVX-512)")
    ax.axvline(1.0, color=DARK, linewidth=1.4)
    ax.set_yticks(x)
    ax.set_yticklabels(common, fontsize=9)
    ax.set_xlabel("Speedup vs scalar  (>1 faster, <1 slower)")
    ax.set_title("Vectorization vs scalar: in-order X60 and out-of-order Xeon", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(outdir, "compare_x60_xeon.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    nx60 = sum(1 for m in common if x60[m] < 1.0)
    nxe  = sum(1 for m in common if xeon[m] < 1.0)
    print(f"  [compare] wrote {p}  (X60 {nx60}/{len(common)} below 1.0, "
          f"Xeon {nxe}/{len(common)} below 1.0)")


# --------------------------------------------------------------------------
# Matrix table
# --------------------------------------------------------------------------
def write_matrix_table(df, outdir):
    base = df[df["arm"] == "baseline"][["matrix", "nnz_c", "op_mean", "op_max"]].copy()
    base = base.drop_duplicates(subset=["matrix"]).sort_values("matrix")
    p = os.path.join(outdir, "matrix_table.csv")
    base.to_csv(p, index=False)
    print(f"  [table] wrote {p}  ({len(base)} matrices)")



# --------------------------------------------------------------------------
# Figure 5: OpenMP scaling (speedup vs 1 thread), one line per matrix
# --------------------------------------------------------------------------
def fig_omp_scaling(df, outdir):
    import numpy as np
    sub = df[df["arm"] == "omp"].copy()
    if sub.empty:
        print("  [omp] no omp rows, skipping")
        return
    # threads present, sorted
    threads = sorted(int(t) for t in sub["threads"].unique())
    fig, ax = plt.subplots(figsize=(8, 6))
    # ideal linear reference
    ax.plot(threads, threads, color=DARK, linewidth=1.2, linestyle="--",
            label="ideal (linear)")
    mats = sorted(sub["matrix"].unique())
    # colour spread
    cmap = plt.cm.viridis
    for i, m in enumerate(mats):
        r = sub[sub["matrix"] == m].sort_values("threads")
        xs = [int(t) for t in r["threads"]]
        ys = [float(v) for v in r["speedup_vs_1thread"]]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.0,
                color=cmap(i / max(1, len(mats) - 1)), alpha=0.85)
    ax.set_xticks(threads)
    ax.set_xlabel("OpenMP threads (one per physical core)")
    ax.set_ylabel("Speedup vs 1 thread")
    ax.set_title("Scalar SpGEMM OpenMP scaling on the X60\n(sublinear; one line per matrix)", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(outdir, "omp_scaling.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    # report 8-thread spread
    at8 = sub[sub["threads"] == 8]["speedup_vs_1thread"].astype(float)
    if len(at8):
        print(f"  [omp] wrote {p}  (8-thread speedup {at8.min():.1f}x - {at8.max():.1f}x)")
    else:
        print(f"  [omp] wrote {p}")


# --------------------------------------------------------------------------
# Figure 6: parallel efficiency at max threads (bar per matrix)
# --------------------------------------------------------------------------
def fig_omp_efficiency(df, outdir):
    sub = df[df["arm"] == "omp"].copy()
    if sub.empty:
        print("  [omp-eff] no omp rows, skipping")
        return
    tmax = max(int(t) for t in sub["threads"].unique())
    r = sub[sub["threads"] == tmax].copy()
    r = r.dropna(subset=["parallel_efficiency"])
    if r.empty:
        print("  [omp-eff] no parallel_efficiency values, skipping")
        return
    r = r.sort_values("parallel_efficiency")
    names = r["matrix"].tolist()
    effs = [float(v) for v in r["parallel_efficiency"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    y = range(len(names))
    ax.barh(list(y), effs, color=GREY, height=0.72)
    ax.axvline(1.0, color=DARK, linewidth=1.2, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Parallel efficiency at %d threads  (1.0 = ideal)" % tmax)
    ax.set_title("How far OpenMP falls short of linear scaling", fontsize=12)
    ax.set_xlim(0, 1.05)
    for i, v in zip(y, effs):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=DARK)
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(outdir, "omp_efficiency.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [omp-eff] wrote {p}  (efficiency {min(effs):.2f} - {max(effs):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("riscv_csv")
    ap.add_argument("intel_csv", nargs="?", default=None)
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("reading", args.riscv_csv)
    dr = load(args.riscv_csv)

    print("figures:")
    fig_speedup_wall(dr, args.outdir, lmul=2)
    fig_lmul(dr, args.outdir)
    fig_structure(dr, args.outdir, lmul=2)
    write_matrix_table(dr, args.outdir)
    fig_omp_scaling(dr, args.outdir)
    fig_omp_efficiency(dr, args.outdir)

    if args.intel_csv:
        print("reading", args.intel_csv)
        di = load(args.intel_csv)
        fig_compare(dr, di, args.outdir)
    else:
        print("  (no Intel CSV given; skipping comparison figure)")

    print("done ->", args.outdir)


if __name__ == "__main__":
    main()