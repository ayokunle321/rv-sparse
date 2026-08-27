#!/usr/bin/env python3
"""Execution-weighted vector length for C = A*A.

Mean nnz/row of B understates the vector work the kernel actually does. The
Gustavson inner loop visits row k of B once for every nonzero in column k of
A, so heavily referenced B rows dominate execution. Weight each B row length
by how often it is executed:

    exec_weighted_mean = sum_k nnz(A col k) * nnz(B row k)
                         / sum_k nnz(A col k)

The numerator is the total intermediate product count and the denominator is
nnz(A), so this is also products-per-nonzero-of-A.

Also reports the fraction of executed product work coming from B rows shorter
than 8, 16 and 32 -- the share of real work that runs in an under-filled
vector at LMUL 1, 2 and 4 on a 256-bit VLEN.

Writes paper/data/vector_fill.csv by default.

Usage:
    python3 paper/analysis/vector_fill.py
    python3 paper/analysis/vector_fill.py --out somewhere.csv
    python3 paper/analysis/vector_fill.py --stdout matrices/cage12.mtx
"""

import csv
import glob
import os
import sys

import numpy as np
import scipy.io as sio


def analyse(path):
    A = sio.mmread(path).tocsr()
    A.sort_indices()

    rows, cols = A.shape

    if rows != cols:
        print(f"{os.path.basename(path)}: not square ({rows}x{cols}), "
              f"A*A undefined, skipped", file=sys.stderr)
        return None

    # B = A, so B row lengths are A row lengths, and the visit count for
    # B row k is the number of nonzeros in column k of A.
    row_len = np.diff(A.indptr).astype(np.int64)
    visits = np.bincount(A.indices, minlength=cols).astype(np.int64)

    # Work contributed by each B row across the whole product.
    work = visits * row_len
    total_work = int(work.sum())
    total_visits = int(visits.sum())

    plain_mean = A.nnz / rows if rows else 0.0
    exec_mean = total_work / total_visits if total_visits else 0.0

    shares = {}
    for t in (8, 16, 32):
        under = int(work[row_len < t].sum())
        shares[t] = under / total_work if total_work else 0.0

    return {
        "matrix": os.path.splitext(os.path.basename(path))[0],
        "rows": rows,
        "nnz_a": A.nnz,
        "intermediate_products": total_work,
        "plain_mean_nnz_row": plain_mean,
        "exec_weighted_mean_row": exec_mean,
        "work_frac_len_lt_8": shares[8],
        "work_frac_len_lt_16": shares[16],
        "work_frac_len_lt_32": shares[32],
    }


DEFAULT_OUT = os.path.join("paper", "data", "vector_fill.csv")


def parse_args(argv):
    """Split argv into an output path and a list of matrix paths."""
    out = DEFAULT_OUT
    paths = []
    i = 0

    while i < len(argv):
        if argv[i] == "--out":
            if i + 1 >= len(argv):
                print("--out needs a path", file=sys.stderr)
                sys.exit(2)
            out = argv[i + 1]
            i += 2
        elif argv[i].startswith("--out="):
            out = argv[i].split("=", 1)[1]
            i += 1
        elif argv[i] in ("--stdout", "-"):
            out = None
            i += 1
        else:
            paths.append(argv[i])
            i += 1

    return out, paths


def open_out(out):
    """Return a writable handle, creating the parent directory if needed."""
    if out is None:
        return sys.stdout, False

    parent = os.path.dirname(os.path.abspath(out))
    os.makedirs(parent, exist_ok=True)

    return open(out, "w", newline=""), True


def main():
    out, paths = parse_args(sys.argv[1:])

    paths = paths or sorted(glob.glob("matrices/*.mtx"))

    if not paths:
        print("no .mtx files found under matrices/", file=sys.stderr)
        sys.exit(1)

    cols = ["matrix", "rows", "nnz_a", "intermediate_products",
            "plain_mean_nnz_row", "exec_weighted_mean_row",
            "work_frac_len_lt_8", "work_frac_len_lt_16",
            "work_frac_len_lt_32"]

    handle, owned = open_out(out)
    writer = csv.writer(handle)
    writer.writerow(cols)

    written = 0

    for path in paths:
        try:
            r = analyse(path)
        except Exception as exc:                      # noqa: BLE001
            print(f"{path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        if r is None:
            continue

        writer.writerow([
            r["matrix"], r["rows"], r["nnz_a"], r["intermediate_products"],
            f"{r['plain_mean_nnz_row']:.4f}",
            f"{r['exec_weighted_mean_row']:.4f}",
            f"{r['work_frac_len_lt_8']:.6f}",
            f"{r['work_frac_len_lt_16']:.6f}",
            f"{r['work_frac_len_lt_32']:.6f}",
        ])

        handle.flush()
        written += 1

    if owned:
        handle.close()
        print(f"wrote {written} row(s) -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
