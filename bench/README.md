# rv-sparse benchmark harness

Times the SpGEMM kernels on one core, checks every result against a reference
kernel, and writes self-describing rows to a CSV.

## Setup

Set `CC` to a compiler that can build and run `-march=rv64gcv` programs on the
machine. Then run the preflight check.

```bash
cp bench/env.sh.example bench/env.sh
$EDITOR bench/env.sh
bash bench/run_bench.sh --check
```

`--check` builds and runs a small vector program to verify the toolchain,
derives runtime library paths from `CC`, and warns about conditions that can
make small effects unreliable, such as an unpinned cpufreq governor. It does
not run any benchmarks. `bench/env.sh` is per machine and gitignored.

## Running

```bash
bash bench/run_bench.sh                        # the whole table
bash bench/run_bench.sh --kernels contig_f32   # one kernel and its baseline
bash bench/run_bench.sh --kernels intrinsic    # every row with that arm
bash bench/run_bench.sh --dtype f32 --runs 30
```

| Flag                 | Meaning                                                       |
| -------------------- | ------------------------------------------------------------- |
| `--check`            | preflight only                                                |
| `--check-matrices`   | verify canonical CSR over the matrix set, then exit           |
| `--dtype f32`        | restrict by dtype, repeatable or comma-separated              |
| `--kernels a,b`      | restrict by kernel name or arm                                |
| `--baseline NAME`    | kernel used for the speedup denominator, default `scalar_f32` |
| `--runs N`           | timed runs per config, default 10                             |
| `--experiments PATH` | use an alternate experiment table                             |

Runs are resumable. A config counts as complete only when it has exactly
`RUNS` rows. Partial groups are removed and run again instead of being topped
up, so interrupting a sweep is safe.

When `--kernels` is used, the baseline stays in the batch so that speedups can
still be calculated.

## Analysis

```bash
python3 bench/analyze.py bench/results/spgemm_raw.csv
```

The analysis groups results by matrix, build, cflags, kernel, and dtype. It
reports median, mean, spread, GOP/s, and speedup with a bootstrap 95% confidence
interval. Configurations whose interval includes 1.0 are marked as not
significant.

The speedup denominator is the `arm=baseline` group for the same dtype and
matrix from the `gc` build. This is intentional. `autovec` uses the same scalar
source with the vector extension enabled, so comparing it with the scalar
kernel from its own build would compare it against itself.

Use `--baseline NAME` to score against a different kernel. This only re-reads
the CSV and does not run the benchmarks again.

## The experiment table

`bench/experiments.tsv` is the only place where experiments are declared. It is
tab separated with one row per cell in the sweep.

```text
arm       kernel        dtype  build  cflags
baseline  scalar_f32    f32    gc     -
autovec   scalar_f32    f32    gcv    -
intrinsic rvv_f32       f32    gcv    -
```

The columns are space-aligned above for readability. The actual file uses
single tabs, and the driver rejects rows that do not.

| Column   | Meaning                                                                                             |
| -------- | --------------------------------------------------------------------------------------------------- |
| `arm`    | what the row is testing, such as `baseline`, `autovec`, `intrinsic`, `adaptive`, or `omp` |
| `kernel` | must match a name in `KERNELS[]` in `bench.c`                                                       |
| `dtype`  | `f32`, `f64`, or `i8`                                                                               |
| `build`  | `gc` or `gcv`, mapped to march flags in `build_march()`                                             |
| `cflags` | extra compile flags for the cell, or `-`                                                            |

An arm describes both a compilation and a kernel. For example, the first two
rows use the same kernel but different builds.

The tunables are compile-time macros, so a `cflags` sweep produces a different
binary. The driver builds one library and one benchmark binary for each unique
`(build, cflags)` pair.

A vector kernel paired with `build=gc` is rejected before the build since it
would not link.

## Adding a kernel to the evaluation

Build the kernel first. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for how to add a strategy.

Then make three changes to include it in the benchmark sweep.

**1. Register it in `bench.c`.**

Wrap the kernel and add it to the registry. Both should be guarded with
`__riscv_vector` when the kernel uses vector intrinsics.

```c
#if defined(__riscv_vector)
KERNEL_WRAP(mykernel_f32_w, rvsp_spgemm_mykernel_f32)
#endif

static const kernel_entry_t KERNELS[] = {
#if defined(__riscv_vector)
    { "mykernel_f32", mykernel_f32_w, RVSP_DTYPE_FP32, "f32",
      scalar_f32_w, "scalar_f32" },
#endif
};
```

The fifth and sixth fields are the validation reference. Use a kernel from the
same family. Every run is checked against it, and a mismatch is recorded as
`correct=0`.

**2. Add a row to `experiments.tsv`.**

```text
intrinsic	mykernel_f32	f32	gcv	-
```

**3. Add the kernel to the vector check in `run_bench.sh`** if it needs the V
extension. This makes an invalid `gc` pairing fail before the sweep instead of
at link time.

```bash
vector_kernel() {
    case "$1" in
        rvv_*|contig_*|adaptive_*)
```

Smoke-test the wiring before starting a longer run.

```bash
bash bench/run_bench.sh --kernels mykernel_f32 --runs 2
```

For a compile-time tunable, add one row for each setting. Each distinct
`cflags` string requires an additional build.

```text
intrinsic	mykernel_f32	f32	gcv	-DRVSP_MYKERNEL_THRESHOLD=8
intrinsic	mykernel_f32	f32	gcv	-DRVSP_MYKERNEL_THRESHOLD=32
```

## Files

```text
experiments.tsv    experiment table
run_bench.sh       preflight, build, sweep, and resume logic
bench.c            timing harness, one kernel on one product
csr_check.c        canonical CSR check over the matrix set
analyze.py         raw rows to summary with confidence intervals
env.sh.example     per machine configuration template
results/           raw and summary CSVs, created on first run
```

## Notes

Only the kernel call is timed. Matrix construction, validation, workspace
allocation, and teardown are outside the timed region. The workspace is
allocated once and reused.

The governor is pinned and restored, core frequency is checked before and after
each config, and experiment order is shuffled per matrix with a recorded seed.

For effects below about 5%, isolate the benchmark core with `isolcpus` or a
cpuset. `taskset` pins the process but does not keep the scheduler off the
core.
