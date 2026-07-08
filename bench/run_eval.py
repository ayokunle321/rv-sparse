#!/usr/bin/env python3
"""
run_eval.py — SpGEMM evaluation orchestrator for gem5 (RISC-V SE mode).

Pipeline:
    experiments.json → build → gem5 runs (parallel, resumable) →
    per-run JSON in results/runs/ → results.csv + verification

Usage:
    ./run_eval.py                            # full suite
    ./run_eval.py --kernels i8               # subset
    ./run_eval.py --matrices wiki-Vote       # single matrix
    ./run_eval.py --parallel 8 --mode cold   # overrides
    ./run_eval.py --reverify                 # re-run verification + CSV only
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Sidecar schema (must match bench_spgemm.c)
SIDECAR_FIELDS = [
    "matrix", "kernel", "mode", "M", "A_nnz", "madd_pairs",
    "ops_analytical", "bytes_analytical", "AI_analytical",
    "C_nnz", "compression", "struct_hash", "val_hash", "val_sum",
    "determinism_ok",
]

# ROI stat keys → output column. Candidates tried in order. A missing
# REQUIRED stat fails the run — a 48h sweep that silently produces an
# empty CSV column is worse than an early abort. All names below were
# verified against a smoke-run stats.txt on this gem5 build, so
# everything load-bearing is required. Only per-op-class counters that
# could legitimately restructure stay optional.
REQUIRED_STATS = {
    "roi_simInsts":     ["simInsts"],
    "roi_simOps":       ["simOps"],
    "roi_numCycles":    ["system.cpu.numCycles"],
    "roi_simSeconds":   ["simSeconds"],
    "vec_insts":        ["system.cpu.commitStats0.numVecInsts"],
    "l1d_misses":       ["system.cpu.dcache.overallMisses::total"],
    "l1d_accesses":     ["system.cpu.dcache.overallAccesses::total"],
    "l2_misses":        ["system.l2.overallMisses::total"],
    "l2_accesses":      ["system.l2.overallAccesses::total"],
    "dram_read_bytes":  ["system.mem_ctrls.dram.dramBytesRead",
                         "system.mem_ctrls.dram.bytesRead::total"],
    "dram_write_bytes": ["system.mem_ctrls.dram.dramBytesWritten",
                         "system.mem_ctrls.dram.bytesWritten::total"],
}
OPTIONAL_STATS = {
    "committed_ops":    ["system.cpu.commitStats0.committedInstType::total"],
}
STAT_KEYS = {**REQUIRED_STATS, **OPTIONAL_STATS}

CSV_COLUMNS = (
    ["build"] + SIDECAR_FIELDS +
    ["roi_simInsts", "roi_simOps", "roi_numCycles", "roi_simSeconds",
     "vec_insts", "vec_ratio",
     "l1d_misses", "l1d_miss_rate", "l2_misses", "l2_miss_rate", "mpki",
     "dram_read_bytes", "dram_write_bytes", "dram_traffic_total",
     "AI_measured", "cycles_per_madd"]
)


def expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(
            f"command failed ({r.returncode}): {' '.join(map(str, cmd))}\n"
            f"stderr:\n{r.stderr[-2000:]}")
    return r.stdout


def stable_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# stats.txt parsing
# --------------------------------------------------------------------------
STAT_LINE = re.compile(r"^(\S+)\s+(-?[\d.]+(?:e[+-]?\d+)?|nan|inf)\s")


def parse_roi_stats(stats_path: Path) -> dict:
    """Parse the first stat block (ROI) of stats.txt into {name: float}."""
    stats, in_block, seen_begin = {}, False, False
    with open(stats_path, errors="replace") as f:
        for line in f:
            if "Begin Simulation Statistics" in line:
                if seen_begin:
                    break
                seen_begin, in_block = True, True
                continue
            if "End Simulation Statistics" in line:
                break
            if not in_block:
                continue
            m = STAT_LINE.match(line)
            if m:
                try:
                    stats[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return stats


def pick(stats: dict, candidates) -> float | None:
    for name in candidates:
        if name in stats:
            return stats[name]
    return None


def count_stat_blocks(stats_path: Path) -> int:
    n = 0
    with open(stats_path, errors="replace") as f:
        for line in f:
            if "Begin Simulation Statistics" in line:
                n += 1
    return n


# --------------------------------------------------------------------------
# configuration / jobs
# --------------------------------------------------------------------------
class Rig:
    def __init__(self, cfg_path: Path, args):
        self.cfg_path = cfg_path
        self.cfg = json.loads(cfg_path.read_text())

        p = self.cfg["paths"]
        self.cc        = expand(p["cc"])
        self.gem5      = expand(p["gem5"])
        self.se_config = expand(p["se_config"])
        self.m5_inc    = expand(p["m5_inc"])
        self.m5_lib    = expand(p["m5_lib"])
        self.harness   = Path(p["harness"])
        self.kdir      = Path(p["kernel_dir"])
        self.inc       = Path(p["include"])
        self.mdir      = Path(p["matrix_dir"])
        self.out       = Path(p["out_root"])

        self.runs_dir    = self.out / "runs"
        self.sidecar_dir = self.out / "sidecar"
        self.gem5_dir    = self.out / "gem5"
        for d in (self.runs_dir, self.sidecar_dir, self.gem5_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.builds = self.cfg["builds"]
        self.kernels = self.cfg["kernels"]
        if args.variant == "gc":
            self.kernels = [k for k in self.kernels if "gcv" not in k["tag"]]
        elif args.variant == "gcv":
            self.kernels = [k for k in self.kernels if "gcv" in k["tag"]]
        if args.kernels:
            self.kernels = [k for k in self.kernels if k["tag"] in args.kernels]
        
        self.matrices = args.matrices or self.cfg["matrices"]
        self.parallel = args.parallel or self.cfg["run"]["parallel"]
        self.timeout  = self.cfg["run"]["timeout_sec"]

        proto = self.cfg["protocol"]
        self.default_mode = args.mode or proto["mode"]
        self.mode_overrides = {} if args.mode else proto.get("mode_overrides", {})

        g = self.cfg["gem5_config"]
        self.gem5_config = g
        self.gem5_args = [
            f"--cpu-type={g['cpu_type']}", "--caches", "--l2cache",
            f"--l1d_size={g['l1d_size']}", f"--l1i_size={g['l1i_size']}",
            f"--l2_size={g['l2_size']}", f"--sys-clock={g['sys_clock']}",
        ] + g.get("extra_args", [])

    def job_hash(self, kernel_cfg: dict, build_cfg: dict, mode: str) -> str:
        # Hash only what affects THIS run's validity. Whole-config hashing
        # meant adding one matrix to the list invalidated every completed
        # run in the suite.
        return stable_hash({
            "gem5_config": self.gem5_config,
            "kernel": kernel_cfg,
            "build": build_cfg,
            "mode": mode,
        })

    def preflight(self):
        problems = []
        for name, path in [("cross compiler", self.cc), ("gem5", self.gem5),
                           ("se.py", self.se_config),
                           ("libm5.a", self.m5_lib / "libm5.a"),
                           ("harness", self.harness)]:
            if not path.exists():
                problems.append(f"missing {name}: {path}")
        for k in self.kernels:
            if not (self.kdir / k["src"]).exists():
                problems.append(f"missing kernel source: {self.kdir / k['src']}")
        if problems:
            sys.exit("preflight failed:\n  " + "\n  ".join(problems))

    def resolve_matrix(self, name: str) -> Path | None:
        direct = self.mdir / name / f"{name}.mtx"
        if direct.is_file():
            return direct
        skip = re.compile(r"_(b|nodename|coord)\.mtx$")
        hits = [q for q in self.mdir.rglob(f"{name}.mtx") if not skip.search(str(q))]
        if len(hits) > 1:
            # Ambiguity is a setup bug — pin it, don't pick rglob-order.
            sys.exit(f"ambiguous matrix '{name}': " +
                     ", ".join(str(h) for h in hits))
        return hits[0] if hits else None

    def build_binaries(self) -> dict:
        bins = {}
        for k in self.kernels:
            for b in self.builds:
                out = Path(f"bench_{k['tag']}_{b['tag']}")
                srcs = [str(self.harness), str(self.kdir / k["src"])]
                cmd = ([str(self.cc), "-O3", f"-march={b['march']}"]
                       + b.get("extra_cflags", [])
                       + ["-static", f"-I{self.inc}", f"-I{self.m5_inc}",
                          f"-D{k['define']}"] + srcs
                       + [str(self.m5_lib / "libm5.a"), "-o", str(out)])
                stale = (not out.exists() or any(
                    Path(s).stat().st_mtime > out.stat().st_mtime for s in srcs))
                if stale:
                    print(f"  CC {out}")
                    sh(cmd)
                bins[(k["tag"], b["tag"])] = out.resolve()
        return bins

    def run_one(self, job) -> dict:
        kernel_cfg, build_cfg, matrix, mtx_path, binary = job
        kernel, build = kernel_cfg["tag"], build_cfg["tag"]
        mode = self.mode_overrides.get(matrix, self.default_mode)
        jhash = self.job_hash(kernel_cfg, build_cfg, mode)
        run_id = f"{matrix}__{kernel}__{build}"
        record_path = self.runs_dir / f"{run_id}.json"

        # skip if already complete with same job config
        if record_path.exists():
            try:
                rec = json.loads(record_path.read_text())
                if rec.get("status") == "ok" and rec.get("job_hash") == jhash:
                    return {"run_id": run_id, "status": "skipped"}
            except json.JSONDecodeError:
                pass

        outdir  = self.gem5_dir / run_id
        outdir.mkdir(parents=True, exist_ok=True)
        sidecar = self.sidecar_dir / f"{run_id}.csv"
        simlog  = outdir / "sim.log"

        cmd = [str(self.gem5), f"--outdir={outdir}", str(self.se_config),
               f"--cmd={binary}",
               f"--options={mtx_path} {sidecar} {mode}"] + self.gem5_args

        record = {
            "run_id": run_id, "matrix": matrix, "kernel": kernel,
            "build": build, "mode": mode, "job_hash": jhash,
            "cmd": " ".join(map(shlex.quote, cmd)),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }

        t0 = time.time()
        try:
            with open(simlog, "w") as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=self.timeout)
            record["wall_sec"] = round(time.time() - t0, 1)
            record["gem5_returncode"] = r.returncode
            if r.returncode != 0:
                raise RuntimeError(f"gem5 exit {r.returncode} (see {simlog})")

            stats_path = outdir / "stats.txt"
            if not stats_path.is_file():
                raise RuntimeError("no stats.txt produced")
            nblocks = count_stat_blocks(stats_path)
            record["stat_blocks"] = nblocks
            if nblocks < 2:
                raise RuntimeError(
                    f"only {nblocks} stat block(s): ROI markers did not fire")
            if not sidecar.is_file() or sidecar.stat().st_size == 0:
                raise RuntimeError("no sidecar written by harness")

            sc_vals = sidecar.read_text().strip().split(",")
            if len(sc_vals) != len(SIDECAR_FIELDS):
                raise RuntimeError(
                    f"sidecar has {len(sc_vals)} fields, expected "
                    f"{len(SIDECAR_FIELDS)}")
            record["sidecar"] = dict(zip(SIDECAR_FIELDS, sc_vals))

            roi = parse_roi_stats(stats_path)
            record["roi_stats"] = roi

            missing = [col for col, cands in REQUIRED_STATS.items()
                       if pick(roi, cands) is None]
            if missing:
                raise RuntimeError(
                    "required ROI stats missing (gem5 version renamed them? "
                    "pin names in REQUIRED_STATS): " + ", ".join(missing))
            record["missing_optional_stats"] = [
                col for col, cands in OPTIONAL_STATS.items()
                if pick(roi, cands) is None]

            record["metrics"] = self.derive_metrics(record["sidecar"], roi)
            record["status"] = "ok"
        except subprocess.TimeoutExpired:
            record.update(status="timeout", wall_sec=round(time.time() - t0, 1))
        except Exception as e:
            record.update(status="failed", error=str(e))

        tmp = record_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        tmp.rename(record_path)
        return record

    @staticmethod
    def derive_metrics(sc: dict, roi: dict) -> dict:
        m = {}
        for col, cands in STAT_KEYS.items():
            m[col] = pick(roi, cands)

        def ratio(a, b):
            return (a / b) if (a is not None and b) else None

        m["vec_ratio"]      = ratio(m["vec_insts"], m["committed_ops"])
        m["l1d_miss_rate"]  = ratio(m["l1d_misses"], m["l1d_accesses"])
        m["l2_miss_rate"]   = ratio(m["l2_misses"], m["l2_accesses"])
        mpki = ratio(m["l2_misses"], m["roi_simInsts"])
        m["mpki"] = mpki * 1000 if mpki is not None else None

        rd, wr = m["dram_read_bytes"], m["dram_write_bytes"]
        m["dram_traffic_total"] = (rd + wr) if (rd is not None and wr is not None) else None
        m["AI_measured"] = ratio(float(sc["ops_analytical"]), m["dram_traffic_total"])
        m["cycles_per_madd"] = ratio(m["roi_numCycles"], float(sc["madd_pairs"]))
        return m

    def verify(self, records) -> list[str]:
        """Cross-run verification: structural agreement per matrix,
        bitwise value agreement within f32, determinism flags.
        determinism_ok: 1 pass, 0 fail, -1 n/a (cold — no reference run)."""
        errors = []
        by_matrix, f32_by_matrix = {}, {}
        for r in records:
            if r["status"] != "ok":
                errors.append(f"{r['run_id']}: status={r['status']}"
                              + (f" ({r.get('error','')})" if r.get("error") else ""))
                continue
            sc = r["sidecar"]
            if sc["determinism_ok"] == "0":
                errors.append(f"{r['run_id']}: determinism check FAILED")
            key = sc["matrix"]
            ref = by_matrix.setdefault(key, (sc["C_nnz"], sc["struct_hash"], r["run_id"]))
            if (sc["C_nnz"], sc["struct_hash"]) != ref[:2]:
                errors.append(f"{key}: structural mismatch {r['run_id']} vs {ref[2]} "
                              f"(C_nnz {sc['C_nnz']} vs {ref[0]})")
            if "f32" in sc["kernel"]:
                fref = f32_by_matrix.setdefault(key, (sc["val_hash"], r["run_id"]))
                if sc["val_hash"] != fref[0]:
                    errors.append(f"{key}: f32 value hash mismatch "
                                  f"{r['run_id']} vs {fref[1]}")
        return errors

    def load_all_records(self) -> list[dict]:
        recs = []
        for f in sorted(self.runs_dir.glob("*.json")):
            try:
                recs.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                recs.append({"run_id": f.stem, "status": "corrupt-record"})
        return recs

    def write_csv(self, records):
        out = Path("results.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in sorted((r for r in records if r["status"] == "ok"),
                            key=lambda r: (r["matrix"], r["kernel"], r["build"])):
                row = {"build": r["build"], **r["sidecar"], **r["metrics"]}
                w.writerow(row)
        return out

    def write_manifest(self):
        def first_line(cmd):
            try:
                return sh(cmd).splitlines()[0]
            except Exception as e:
                return f"unavailable: {e}"
        manifest = {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "host": os.uname().nodename,
            "config_file": str(self.cfg_path),
            "config": self.cfg,
            "gem5_version": first_line([str(self.gem5), "--version"]),
            "toolchain": first_line([str(self.cc), "--version"]),
            "repo_commit": first_line(["git", "rev-parse", "HEAD"]),
            "repo_dirty": bool(sh(["git", "status", "--porcelain"]).strip()),
        }
        (self.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="bench/experiments.json", type=Path)
    ap.add_argument("--kernels", nargs="*", help="subset of kernel tags")
    ap.add_argument("--variant", choices=["gc", "gcv", "both"], default="both",
        help="Run scalar kernels, vector kernels, or both.",
    )
    ap.add_argument("--matrices", nargs="*", help="subset of matrices")
    ap.add_argument("--parallel", type=int)
    ap.add_argument("--mode", choices=["warm", "cold"],
                    help="override protocol for ALL matrices")
    ap.add_argument("--reverify", action="store_true",
                    help="skip runs; regenerate CSV + verification from records")
    args = ap.parse_args()

    rig = Rig(args.config, args)

    if not args.reverify:
        rig.preflight()
        print("building binaries...")
        bins = rig.build_binaries()

        jobs, missing = [], []
        for m in rig.matrices:
            mtx = rig.resolve_matrix(m)
            if mtx is None:
                missing.append(m)
                continue
            for k in rig.kernels:
                for b in rig.builds:
                    jobs.append((k, b, m, mtx, bins[(k["tag"], b["tag"])]))
        if missing:
            print(f"WARN: missing matrices: {', '.join(missing)}", file=sys.stderr)

        rig.write_manifest()
        print(f"{len(jobs)} runs, {rig.parallel} in parallel "
              f"(protocol: {rig.default_mode}"
              + (f", overrides: {rig.mode_overrides}" if rig.mode_overrides else "")
              + ")")
        done = 0
        with cf.ThreadPoolExecutor(max_workers=rig.parallel) as pool:
            futures = {pool.submit(rig.run_one, j): j for j in jobs}
            for fut in cf.as_completed(futures):
                rec, done = fut.result(), done + 1
                tag = {"ok": "done", "skipped": "skip"}.get(rec["status"],
                                                            rec["status"].upper())
                wall = f" [{rec['wall_sec']}s]" if rec.get("wall_sec") else ""
                print(f"  [{done}/{len(jobs)}] {tag:>8}  {rec['run_id']}{wall}")

    records = rig.load_all_records()
    csv_path = rig.write_csv(records)

    errors = rig.verify(records)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n{n_ok}/{len(records)} runs ok -> {csv_path}")
    print(f"per-run JSON: {rig.runs_dir}/")
    print(f"manifest: {rig.out / 'MANIFEST.json'}")
    if errors:
        print("\n=== VERIFICATION FAILURES ===", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        sys.exit(1)
    print("verification: all checks passed")


if __name__ == "__main__":
    main()
    