#!/usr/bin/env python3
"""
scrape_results.py — flatten results/runs/*.json into two CSVs.


Output (in {out-dir}/):
    results.csv         one row per ok run (matrix x kernel x build)
    matrix_stats.csv    static matrix properties, one row per matrix

Usage:
    ./scrape_results.py --runs-dir results/runs --matrix-dir matrices --out-dir eval
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd


def find_matrix_file(matrix_dir: Path, name: str) -> Path | None:
    direct = matrix_dir / name / f"{name}.mtx"
    if direct.is_file():
        return direct
    skip = re.compile(r"_(b|nodename|coord)\.mtx$")
    hits = [p for p in matrix_dir.rglob(f"{name}.mtx") if not skip.search(str(p))]
    return hits[0] if hits else None


def matrix_row_stats(mtx_path: Path):
    """Return (rows, cols, nnz, avg_nnz_row, max_nnz_row, max_over_median)."""
    with open(mtx_path) as f:
        header = f.readline()
        is_symmetric = "symmetric" in header or "hermitian" in header
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        rows, cols, entries = map(int, line.split()[:3])
        row_counts = [0] * rows
        for _ in range(entries):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            r, c = int(parts[0]) - 1, int(parts[1]) - 1
            row_counts[r] += 1
            if is_symmetric and r != c:
                row_counts[c] += 1
    nnz = sum(row_counts)
    avg = nnz / rows if rows else 0
    mx = max(row_counts) if row_counts else 0
    sorted_rc = sorted(row_counts)
    med = sorted_rc[len(sorted_rc) // 2] if sorted_rc else 0
    ratio = (mx / med) if med else float("inf")
    return rows, cols, nnz, avg, mx, ratio


def to_float(d: dict, key):
    # Sidecar values arrive as strings and metrics may be missing or None.
    # Return a float where possible, otherwise None, so the column stays numeric
    # without blowing up on a stray value.
    v = d.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    ap.add_argument("--matrix-dir", type=Path, default=Path("matrices"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for f in sorted(args.runs_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("status") == "ok":
            records.append(rec)

    if not records:
        raise SystemExit(f"no ok records found in {args.runs_dir}")

    # ---- flat performance table, one row per run ----
    perf_rows = []
    for r in records:
        sc, m = r["sidecar"], r["metrics"]
        madds = to_float(sc, "madd_pairs")
        cycles = m.get("roi_numCycles")
        sim_seconds = m.get("roi_simSeconds")
        sim_insts = m.get("roi_simInsts")
        ops_per_sec = (2 * madds / sim_seconds
                       if (madds is not None and sim_seconds) else None)
        ipc = (sim_insts / cycles) if (sim_insts and cycles) else None
        perf_rows.append({
            "matrix": sc["matrix"],
            "kernel": r["kernel"],
            "build": r["build"],
            "mode": r["mode"],
            "cycles": cycles,
            "sim_seconds": sim_seconds,
            "ipc": ipc,
            "cycles_per_madd": m.get("cycles_per_madd"),
            "ops_per_sec": ops_per_sec,
            "vec_ratio": m.get("vec_ratio"),
            "AI_analytical": to_float(sc, "AI_analytical"),
            "AI_measured": m.get("AI_measured"),
            "dram_traffic_total": m.get("dram_traffic_total"),
            "l1d_miss_rate": m.get("l1d_miss_rate"),
            "l2_miss_rate": m.get("l2_miss_rate"),
            "mpki": m.get("mpki"),
            "C_nnz": int(sc["C_nnz"]),
            "compression": to_float(sc, "compression"),
        })
    perf_df = (pd.DataFrame(perf_rows)
               .sort_values(["matrix", "kernel", "build"]))

    results_path = args.out_dir / "results.csv"
    perf_df.to_csv(results_path, index=False)
    print(f"wrote {results_path} ({len(perf_df)} rows)")

    # ---- matrix_stats.csv ----
    matrices = sorted(perf_df["matrix"].unique())
    mat_rows = []
    for name in matrices:
        mtx = find_matrix_file(args.matrix_dir, name)
        sc = next(r["sidecar"] for r in records if r["sidecar"]["matrix"] == name)
        row = {"matrix": name, "M": int(sc["M"]), "A_nnz": int(sc["A_nnz"])}
        row["density"] = row["A_nnz"] / (row["M"] ** 2) if row["M"] else 0
        if mtx:
            _, _, _, avg, mx, ratio = matrix_row_stats(mtx)
            row.update(avg_nnz_row=round(avg, 2), max_nnz_row=mx,
                       max_over_median_nnz_row=round(ratio, 2))
        madds_by_kernel = {r["kernel"]: to_float(r["sidecar"], "madd_pairs")
                           for r in records if r["sidecar"]["matrix"] == name}
        row["madd_pairs"] = next(iter(madds_by_kernel.values()), None)
        mat_rows.append(row)

    stats_path = args.out_dir / "matrix_stats.csv"
    pd.DataFrame(mat_rows).to_csv(stats_path, index=False)
    print(f"wrote {stats_path} ({len(mat_rows)} matrices)")


if __name__ == "__main__":
    main()