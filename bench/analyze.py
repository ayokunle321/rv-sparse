#!/usr/bin/env python3
"""
Summarizes raw benchmark results into per-kernel execution time and GOP/s
statistics, and computes bootstrap confidence intervals for speedups.

Usage:
    python3 analyze.py RAW.csv [--csv-out summary.csv]
"""
import sys
import csv
import random
import statistics
from collections import defaultdict

# scalar baseline kernel per dtype, for speedup computation
BASELINE = {"f32": "scalar_f32", "f64": "scalar_f64", "i8": "scalar_i8"}

# bootstrap settings for the speedup CI. Fixed seed so reruns are reproducible.
BOOT_ITERS = 2000
BOOT_SEED = 20240517
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5

# a config needs spread below this to be trustworthy for a small-effect claim
CLEAN_SPREAD = 0.02   # 2%
NOISY_SPREAD = 0.10   # 10%


def _fnum(row, key, cast=float, default=None):
    """Parse an optional numeric column; return default if absent/unparseable."""
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except ValueError:
        return default


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("kernel"):
                continue
            # accept both new (gops) and old (gflops) column names
            metric = r.get("gops", r.get("gflops"))
            try:
                r["time_s"] = float(r["time_s"])
                r["gops"] = float(metric)
                r["correct"] = int(r["correct"])
                r["nnz_c"] = int(r["nnz_c"])
                r["flops"] = float(r["flops"])
            except (ValueError, KeyError, TypeError):
                continue
            r["cycles"] = _fnum(r, "cycles", float)
            r["instructions"] = _fnum(r, "instructions", float)
            rows.append(r)
    return rows


def _iqr(vals):
    """Interquartile range, a robust spread measure. Needs at least 4 points."""
    if len(vals) < 4:
        return 0.0
    s = sorted(vals)
    n = len(s)
    q1 = statistics.median(s[: n // 2])
    q3 = statistics.median(s[(n + 1) // 2:])
    return q3 - q1


def _pct(sorted_vals, p):
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    idx = int(round(p / 100.0 * (n - 1)))
    idx = min(n - 1, max(0, idx))
    return sorted_vals[idx]


def _bootstrap_speedup_ci(base_times, kern_times):
    """95% CI on speedup = median(base) / median(kernel).

    Resamples each kernel's timings independently (they came from separate
    process invocations, so they are not paired) and takes the ratio of medians
    each iteration. Returns (lo, hi). NaN if either side has < 4 runs.
    """
    if len(base_times) < 4 or len(kern_times) < 4:
        return (float("nan"), float("nan"))
    rnd = random.Random(BOOT_SEED)
    nb, nk = len(base_times), len(kern_times)
    ratios = []
    for _ in range(BOOT_ITERS):
        bs = [base_times[rnd.randrange(nb)] for _ in range(nb)]
        ks = [kern_times[rnd.randrange(nk)] for _ in range(nk)]
        mk = statistics.median(ks)
        if mk > 0:
            ratios.append(statistics.median(bs) / mk)
    if not ratios:
        return (float("nan"), float("nan"))
    ratios.sort()
    return (_pct(ratios, CI_LO_PCT), _pct(ratios, CI_HI_PCT))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["kernel"], r["dtype"])].append(r)

    summary = {}
    for (label, kernel, dtype), rs in groups.items():
        times = [r["time_s"] for r in rs]
        gfs = [r["gops"] for r in rs]
        corrects = [r["correct"] for r in rs]
        if 0 in corrects:
            status = "FAIL"
        elif all(c == -1 for c in corrects):
            status = "noref"
        else:
            status = "ok"

        entry = {
            "runs": len(rs),
            "_times": times,     # kept for the bootstrap; not written to CSV
            "time_median": statistics.median(times),
            "time_mean": statistics.mean(times),
            "time_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
            "time_min": min(times),
            "time_max": max(times),
            "time_iqr": _iqr(times),
            "gops_median": statistics.median(gfs),
            "gops_mean": statistics.mean(gfs),
            "gops_std": statistics.pstdev(gfs) if len(gfs) > 1 else 0.0,
            "gops_min": min(gfs),
            "gops_max": max(gfs),
            "nnz_c": rs[0]["nnz_c"],
            "status": status,
        }

        entry["time_rel_spread"] = (
            (entry["time_max"] - entry["time_min"]) / entry["time_median"]
            if entry["time_median"] > 0 else 0.0
        )

        cyc = [r["cycles"] for r in rs if r["cycles"] is not None]
        ins = [r["instructions"] for r in rs if r["instructions"] is not None]
        entry["cycles_median"] = statistics.median(cyc) if cyc else None
        entry["instructions_median"] = statistics.median(ins) if ins else None
        entry["ipc"] = (
            entry["instructions_median"] / entry["cycles_median"]
            if entry["cycles_median"] and entry["instructions_median"] else None
        )
        entry["cycles_per_nnz_c"] = (
            entry["cycles_median"] / entry["nnz_c"]
            if entry["cycles_median"] and entry["nnz_c"] > 0 else None
        )

        summary[(label, kernel, dtype)] = entry
    return summary


def add_speedups(summary):
    base_time = {}
    base_times = {}
    for (label, kernel, dtype), s in summary.items():
        if kernel == BASELINE.get(dtype):
            base_time[(label, dtype)] = s["time_median"]
            base_times[(label, dtype)] = s["_times"]

    for (label, kernel, dtype), s in summary.items():
        bt = base_time.get((label, dtype))
        s["speedup"] = (bt / s["time_median"]) if (bt and s["time_median"] > 0) else float("nan")

        is_baseline = (kernel == BASELINE.get(dtype))
        if is_baseline:
            s["speedup_lo"] = s["speedup_hi"] = 1.0
            s["significant"] = None   # baseline vs itself, meaningless
            continue

        bts = base_times.get((label, dtype))
        if bts is None:
            s["speedup_lo"] = s["speedup_hi"] = float("nan")
            s["significant"] = None
            continue

        lo, hi = _bootstrap_speedup_ci(bts, s["_times"])
        s["speedup_lo"], s["speedup_hi"] = lo, hi
        if lo != lo or hi != hi:            # NaN, not enough runs
            s["significant"] = None
        else:
            s["significant"] = (lo > 1.0) or (hi < 1.0)   # CI excludes 1.0x


def print_table(summary, show_perf):
    keys = sorted(summary.keys(), key=lambda k: (k[0], k[2], k[1]))

    if show_perf:
        hdr = (f"{'matrix':<20} {'kernel':<14} {'dt':<4} {'n':>3} "
               f"{'median_ms':>10} {'spread':>7} {'gops':>8} "
               f"{'speedup':>8} {'95% CI':>16} {'sig':>4} "
               f"{'cyc/nnzC':>9} {'IPC':>5} {'st':>5}")
    else:
        hdr = (f"{'matrix':<20} {'kernel':<14} {'dt':<4} {'n':>3} "
               f"{'median_ms':>10} {'spread':>7} {'gops':>8} "
               f"{'speedup':>8} {'95% CI':>16} {'sig':>4} "
               f"{'nnz_C':>10} {'st':>5}")
    print(hdr)
    print("-" * len(hdr))

    last_label = None
    for k in keys:
        label, kernel, dtype = k
        s = summary[k]
        if last_label is not None and label != last_label:
            print()
        last_label = label

        sp = s["speedup"]
        sp_str = f"{sp:6.2f}x" if sp == sp else "     - "
        spread = f"{s['time_rel_spread']*100:6.1f}%"

        lo, hi = s.get("speedup_lo", float("nan")), s.get("speedup_hi", float("nan"))
        if lo == lo and hi == hi:
            ci_str = f"[{lo:4.2f},{hi:4.2f}]"
        else:
            ci_str = "          -"

        sig = s.get("significant")
        if sig is None:
            sig_str = "  - "
        else:
            sig_str = " sig" if sig else "  ns"

        if show_perf:
            cyc_nnz = f"{s['cycles_per_nnz_c']:9.1f}" if s["cycles_per_nnz_c"] else "        -"
            ipc = f"{s['ipc']:5.2f}" if s["ipc"] else "    -"
            print(f"{label:<20} {kernel:<14} {dtype:<4} {s['runs']:>3} "
                  f"{s['time_median']*1e3:>10.4f} {spread:>7} {s['gops_median']:>8.3f} "
                  f"{sp_str:>8} {ci_str:>16} {sig_str:>4} "
                  f"{cyc_nnz} {ipc} {s['status']:>5}")
        else:
            print(f"{label:<20} {kernel:<14} {dtype:<4} {s['runs']:>3} "
                  f"{s['time_median']*1e3:>10.4f} {spread:>7} {s['gops_median']:>8.3f} "
                  f"{sp_str:>8} {ci_str:>16} {sig_str:>4} "
                  f"{s['nnz_c']:>10} {s['status']:>5}")


def write_csv(summary, path):
    keys = sorted(summary.keys(), key=lambda k: (k[0], k[2], k[1]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "kernel", "dtype", "runs",
                    "time_median_s", "time_mean_s", "time_std_s",
                    "time_min_s", "time_max_s", "time_iqr_s", "time_rel_spread",
                    "gops_median", "gops_mean", "gops_std",
                    "gops_min", "gops_max",
                    "speedup_vs_scalar", "speedup_ci_lo", "speedup_ci_hi",
                    "speedup_significant", "nnz_c",
                    "cycles_median", "instructions_median", "ipc", "cycles_per_nnz_c",
                    "status"])
        for k in keys:
            label, kernel, dtype = k
            s = summary[k]

            def opt(v, fmt="{:.4f}"):
                return fmt.format(v) if v is not None else ""

            def num(v, fmt="{:.4f}"):
                return fmt.format(v) if (v is not None and v == v) else ""

            sig = s.get("significant")
            sig_str = "" if sig is None else ("yes" if sig else "no")

            w.writerow([label, kernel, dtype, s["runs"],
                        f"{s['time_median']:.9f}", f"{s['time_mean']:.9f}", f"{s['time_std']:.9f}",
                        f"{s['time_min']:.9f}", f"{s['time_max']:.9f}", f"{s['time_iqr']:.9f}",
                        f"{s['time_rel_spread']:.4f}",
                        f"{s['gops_median']:.4f}", f"{s['gops_mean']:.4f}", f"{s['gops_std']:.4f}",
                        f"{s['gops_min']:.4f}", f"{s['gops_max']:.4f}",
                        num(s['speedup']), num(s.get('speedup_lo')), num(s.get('speedup_hi')),
                        sig_str, s["nnz_c"],
                        opt(s["cycles_median"], "{:.0f}"),
                        opt(s["instructions_median"], "{:.0f}"),
                        opt(s["ipc"], "{:.4f}"),
                        opt(s["cycles_per_nnz_c"], "{:.4f}"),
                        s["status"]])
    print(f"\nwrote summary csv -> {path}")


def main():
    if len(sys.argv) < 2:
        print("usage: python3 analyze.py RAW.csv [--csv-out summary.csv]")
        sys.exit(2)
    path = sys.argv[1]
    csv_out = None
    if "--csv-out" in sys.argv:
        csv_out = sys.argv[sys.argv.index("--csv-out") + 1]

    rows = load(path)
    if not rows:
        print("no valid rows found in", path)
        sys.exit(1)

    summary = summarize(rows)
    add_speedups(summary)

    show_perf = any(s["cycles_median"] is not None for s in summary.values())
    print_table(summary, show_perf)

    if not show_perf:
        print("\n(no perf counters in this CSV — run the driver with PERF=1 to add "
              "cycles/instructions)")

    # the real guard for small-effect claims, any kernel whose CI includes
    # 1.0 cannot be called faster or slower
    not_sig = [(k, s) for k, s in summary.items()
               if s.get("significant") is False]
    if not_sig:
        print("\n*** EFFECT WITHIN NOISE — do NOT claim faster/slower for these ***")
        print("    (95% speedup CI includes 1.0x — the sign of the effect is not resolved)")
        for (label, kernel, dtype), s in sorted(not_sig):
            print(f"    {label} / {kernel} ({dtype}): "
                  f"{s['speedup']:.3f}x, CI [{s['speedup_lo']:.3f}, {s['speedup_hi']:.3f}]")

    # configs clean enough that a sub-5% effect would even be measurable
    dirty = [(k, s) for k, s in summary.items()
             if s["time_rel_spread"] > CLEAN_SPREAD and s["runs"] >= 4]
    if dirty:
        print(f"\n*** SPREAD > {CLEAN_SPREAD*100:.0f}% — too noisy to trust a small effect here ***")
        for (label, kernel, dtype), s in sorted(dirty):
            print(f"    {label} / {kernel}: {s['time_rel_spread']*100:.1f}% "
                  f"(min-max). A 2-4% claim needs spread well under 2%.")

    noisy = [(k, s) for k, s in summary.items() if s["time_rel_spread"] > NOISY_SPREAD]
    if noisy:
        print(f"\n*** HIGH VARIANCE (>{NOISY_SPREAD*100:.0f}% min-max spread) ***")
        for (label, kernel, dtype), s in sorted(noisy):
            print(f"    {label} / {kernel}: {s['time_rel_spread']*100:.1f}% "
                  f"— more runs or a quieter machine")

    fails = [k for k, s in summary.items() if s["status"] == "FAIL"]
    if fails:
        print("\n*** CORRECTNESS FAILURES ***")
        for label, kernel, dtype in fails:
            print(f"    {label} / {kernel} ({dtype}) produced wrong output — perf number invalid")

    if csv_out:
        write_csv(summary, csv_out)


if __name__ == "__main__":
    main()
