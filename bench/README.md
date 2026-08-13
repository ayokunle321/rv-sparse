# rv-sparse benchmark harness

Measures the SpGEMM kernels on one core, validates every result, and writes
self-describing rows to an append-only CSV.

## Setup

`CC` is the only thing you have to set. It must be a compiler that can build and
run `-march=rv64gcv` programs on this machine.

```bash
cp bench/env.sh.example bench/env.sh
$EDITOR bench/env.sh
bash bench/run_bench.sh --check
```

`--check` runs preflight and exits without measuring anything. It compiles and
runs a small vector program to confirm the toolchain works, derives runtime
library paths from `CC`, and warns about anything that would make small effects
untrustworthy, such as an unwritable cpufreq governor or a core that is not
isolated.

`bench/env.sh` is per machine and gitignored.

## Running

```bash
bash bench/run_bench.sh                              # everything in the table
bash bench/run_bench.sh --kernels contig_f32         # one kernel plus its baseline
bash bench/run_bench.sh --kernels intrinsic          # every row with that arm
bash bench/run_bench.sh --dtype f32 --runs 30
```

| Flag | Meaning |
| --- | --- |
| `--check` | preflight only |
| `--check-matrices` | verify canonical CSR over the matrix set, then exit |
| `--dtype f32` | restrict by dtype, repeatable or comma-separated |
| `--kernels a,b` | restrict by kernel name or by arm |
| `--baseline NAME` | kernel supplying the speedup denominator, default `scalar_f32` |
| `--runs N` | timed runs per config, default 10 |
| `--experiments PATH` | alternate experiment table |

Runs are resumable. A config is complete only at exactly `RUNS` rows, and a
partial group is purged and re-run rather than topped up, so interrupting a
sweep is safe. Re-running skips what is already done.

Narrowing with `--kernels` keeps the baseline in the batch automatically.
Without it every speedup in the summary would be blank.

## Analysis

```bash
python3 bench/analyze.py bench/results/spgemm_raw.csv
```

Groups by matrix, build, cflags, kernel and dtype, then reports median, mean,
spread and GOP/s, plus a speedup with a bootstrap 95% confidence interval.

The denominator is the `arm=baseline` group of the same dtype and matrix from
the `gc` build. That is deliberately cross-build. `autovec` is the same scalar
source compiled with the vector extension available, so scoring it against its
own build would score it against itself and print 1.00x forever.

Any config whose confidence interval includes 1.0 is flagged as not
significant. A 3% win only gets reported where the data supports it.

Use `--baseline NAME` to score everything against a different kernel. It
re-reads the same CSV, so re-baselining costs nothing and re-runs nothing.

## The experiment table

`bench/experiments.tsv` is the only place experiments are declared. Tab
separated, one row per cell of the sweep.

```text
arm       kernel        dtype  build  cflags
baseline  scalar_f32    f32    gc     -
autovec   scalar_f32    f32    gcv    -
intrinsic rvv_f32       f32    gcv    -
```

Columns above are space-aligned for reading. In the file they are separated by
single tabs, and the driver rejects a row that is not.

| Column | Meaning |
| --- | --- |
| `arm` | what the row is evidence for. One of `baseline`, `autovec`, `intrinsic`, `scalar_unroll`, `adaptive` |
| `kernel` | must match a name in `KERNELS[]` in `bench/bench.c` |
| `dtype` | `f32`, `f64` or `i8` |
| `build` | `gc` or `gcv`. Mapped to march flags in `build_march()` and nowhere else |
| `cflags` | extra compile flags for this cell, or `-` |

Rows one and two are the same kernel and the same source at two different
builds. That is the point. An arm is a compilation and a kernel together, not a
property of the code alone, which is why `bench.c` cannot infer its own arm and
the driver passes it in.

`cflags` exists because the tunables are compile-time macros. Sweeping
`RVSP_CONTIG_MIN` means a different binary, not a different argument, so the
driver builds one library and one bench binary per unique `(build, cflags)`
pair and records both in every row.

A vector kernel paired with `build=gc` is rejected up front. Those sources
`#error` without the V extension and the Makefile excludes them from that
build, so the row could not link.

## Adding a kernel to the evaluation

The kernel itself comes first. See the "Adding a strategy" section in
[CONTRIBUTING.md](../CONTRIBUTING.md). Once it exists and builds, three edits
put it in the sweep.

**1. Register it in `bench/bench.c`.**

Wrap it and add a registry entry. Guard both with `__riscv_vector` if it is a
vector kernel, so the scalar build still compiles.

```c
#if defined(__riscv_vector)
KERNEL_WRAP(mykernel_f32_w, rvsp_spgemm_mykernel_f32)
#endif

static const kernel_entry_t KERNELS[] = {
    /* ... */
#if defined(__riscv_vector)
    { "mykernel_f32", mykernel_f32_w, RVSP_DTYPE_FP32, "f32",
      scalar_f32_w, "scalar_f32" },
#endif
};
```

The fifth and sixth fields are the validation reference. Point them at a kernel
of the same family. Every timed run is checked against it, and a mismatch is
reported as `correct=0` in the CSV rather than being silently timed.

**2. Add a row to `bench/experiments.tsv`.**

```text
intrinsic	mykernel_f32	f32	gcv	-
```

Pick the arm that says what the row is evidence for. Use `gcv` for a vector
kernel.

**3. Check the vector glob in `bench/run_bench.sh`.**

```bash
vector_kernel() {
    case "$1" in
        rvv_*|contig_*|adaptive_*)
```

If the name does not match one of those prefixes and the kernel needs the V
extension, add it. This is what rejects an invalid `gc` pairing before the
sweep starts instead of failing at link time an hour in.

Then check the wiring before committing to a long run.

```bash
bash bench/run_bench.sh --kernels mykernel_f32 --runs 2
```

That builds only what the row needs, runs the smoke test, and produces a few
rows. If the name does not match `KERNELS[]` the driver stops immediately and
lists the valid names.

To sweep a compile-time tunable, add one row per setting. Each distinct
`cflags` string costs an extra library and bench build, so add them
deliberately.

```text
intrinsic	mykernel_f32	f32	gcv	-DRVSP_MYKERNEL_THRESHOLD=8
intrinsic	mykernel_f32	f32	gcv	-DRVSP_MYKERNEL_THRESHOLD=32
```

## Files

```text
experiments.tsv    the experiment table, the only place experiments live
run_bench.sh       driver: preflight, build, sweep, resume, integrity
bench.c            timing harness, one kernel on one product
csr_check.c        canonical CSR check over the matrix set
analyze.py         raw rows to summary with confidence intervals
env.sh.example     per machine config template
results/           raw and summary CSVs, created on first run
```

## Why the numbers can be trusted

One core, one process, one config at a time, so nothing contends for cache or
bandwidth. Only the kernel call is timed. Matrix construction, validation,
workspace allocation and teardown all sit outside the timed region, and the
workspace is allocated once and reused across every run.

Every timed run is validated against a reference of the same family. The
governor is pinned and restored, the core frequency is read before and after
each config and flagged if it moved, and the order of experiments is shuffled
per matrix with a recorded seed so thermal drift cannot masquerade as a
regression.

For effects below about 5%, also isolate the benchmark core with `isolcpus` or
a cpuset. `taskset` pins the process but does not stop the scheduler putting
other work on that core.
