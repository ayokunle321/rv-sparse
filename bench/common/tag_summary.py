#!/usr/bin/env python3
"""tag_summary.py — add machine, isa, compiler and flags columns to a summary.

Takes a summary.csv from analyze.py, keeps every column as it is, and appends
four columns so summaries from different machines can be concatenated and
compared side by side.

A summary from before the pinning, perf_note, error and provenance columns
gets them, set to "unrecorded", so it still lines up with a newer one.

flags is the exact compile line of each row's build variant when a
variants.tsv (written by bench/intel/run_bench.sh) is given, matched on build
and cflags. Without one, flags is --base-flags followed by the row's cflags.

Usage:
    python3 bench/common/tag_summary.py bench/intel/results/summary.csv \\
        --out bench/intel/results/summary_tagged.csv \\
        --machine "Intel Xeon 6787P" --isa "x86_64 AVX-512" \\
        --compiler "gcc-16 (GCC) 16.1.0" \\
        --variants bench/intel/results/variants.tsv

    # A RISC-V summary, whose runs predate variants.tsv:
    python3 bench/common/tag_summary.py summary.csv --out riscv_tagged.csv \\
        --machine "SpacemiT X60" --isa "rv64gcv" --compiler "gcc 16.1.0" \\
        --base-flags "-Wall -Wextra -Wpedantic -O3 -flto -std=c11"

Pure standard library.
"""

import argparse
import csv
import sys

NEW_COLUMNS = ["machine", "isa", "compiler", "flags"]

# Columns analyze.py gained later; older summaries get them filled.
LATER_COLUMNS = [
    "pinning", "perf_note",
    "tol_fail", "max_rel_err", "mean_rel_err", "err_class",
    "warmup", "git_commit", "team_size", "bound_cpus",
]


def load_variants(path):
    flags = {}

    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            flags[(row["build"], row["cflags"])] = row["flags"]

    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--isa", required=True)
    ap.add_argument("--compiler", required=True)
    ap.add_argument("--variants")
    ap.add_argument("--base-flags", default="")
    args = ap.parse_args()

    variants = load_variants(args.variants) if args.variants else None

    with open(args.summary, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    clash = [c for c in NEW_COLUMNS if c in fields]

    if clash:
        sys.exit(f"{args.summary} already has column(s) {clash}")

    for col in LATER_COLUMNS:
        if col not in fields:
            fields.append(col)
            for row in rows:
                row[col] = "unrecorded"

    for row in rows:
        cflags = row.get("cflags", "-")

        if variants is not None:
            key = (row.get("build", ""), cflags)

            if key not in variants:
                sys.exit(f"no variant for build={key[0]!r} cflags={key[1]!r} "
                         f"in {args.variants}")

            flags = variants[key]
        else:
            parts = [args.base_flags, row.get("march", "")]

            if cflags and cflags != "-":
                parts.append(cflags)

            flags = " ".join(p for p in parts if p and p != "-")

        row["machine"] = args.machine
        row["isa"] = args.isa
        row["compiler"] = args.compiler
        row["flags"] = flags

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + NEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
