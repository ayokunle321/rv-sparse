#!/usr/bin/env python3
"""Dense accumulator size relative to L2, as a continuous ratio.

The Gustavson kernel keeps one dense fp32 accumulator of ncols(C) entries
live across a row. For C = A*A, ncols(C) = ncols(A). Reporting a fits/does-not-
fit binary against L2 loses the distinction between just over and far over, so
report the ratio itself.

    accumulator_bytes = 4 * ncols(A)
    ratio             = accumulator_bytes / L2_BYTES

Writes paper/data/accumulator_ratio.csv by default.

Usage:
    python3 paper/analysis/accumulator_ratio.py
    python3 paper/analysis/accumulator_ratio.py --out somewhere.csv
    L2_BYTES=262144 python3 paper/analysis/accumulator_ratio.py --stdout
"""

import csv
import glob
import os
import sys

import scipy.io as sio

# SpacemiT X60: 512 KB L2, shared per 4-core cluster.
L2_BYTES = int(os.environ.get("L2_BYTES", 512 * 1024))


def analyse(path):
    A = sio.mmread(path).tocsr()

    rows, cols = A.shape

    if rows != cols:
        print(f"{os.path.basename(path)}: not square ({rows}x{cols}), "
              f"A*A undefined, skipped", file=sys.stderr)
        return None

    acc_bytes = 4 * cols

    return {
        "matrix": os.path.splitext(os.path.basename(path))[0],
        "rows": rows,
        "cols": cols,
        "nnz_a": A.nnz,
        "acc_bytes": acc_bytes,
        "acc_kib": acc_bytes / 1024.0,
        "l2_bytes": L2_BYTES,
        "acc_over_l2": acc_bytes / L2_BYTES,
    }


DEFAULT_OUT = os.path.join("paper", "data", "accumulator_ratio.csv")


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

    handle, owned = open_out(out)
    writer = csv.writer(handle)
    writer.writerow(["matrix", "rows", "cols", "nnz_a", "acc_bytes",
                     "acc_kib", "l2_bytes", "acc_over_l2"])

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
            r["matrix"], r["rows"], r["cols"], r["nnz_a"], r["acc_bytes"],
            f"{r['acc_kib']:.2f}", r["l2_bytes"],
            f"{r['acc_over_l2']:.6f}",
        ])

        handle.flush()
        written += 1

    if owned:
        handle.close()
        print(f"wrote {written} row(s) -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
