# Benchmark harness

Times each SpGEMM kernel against a reference, checks every result, and appends
one row per run to a CSV.

The harness is shared and the drivers are per machine. `common/` holds the
timing harness, the CSR validator and the analysis, which are ISA-neutral and
built for both. `riscv/` drives the SpacemiT X60 runs and `intel/` the x86
comparison arm. Both drivers write the same raw and summary CSV schema, so the
two machines' results line up row for row.

```text
common/bench.c            timing harness, one kernel on one product
common/csr_check.c        canonical CSR check over the matrix set
common/analyze.py         raw rows to summary with confidence intervals
common/tag_summary.py     adds machine, isa, compiler, flags columns
common/check_disasm.sh    disassembly checks: scalar, vector, report
riscv/run_bench.sh        RISC-V driver
riscv/experiments.tsv     RISC-V experiment table
riscv/env.sh.example      RISC-V per-machine config template
intel/run_bench.sh        x86 driver, a port of the RISC-V one
intel/experiments.tsv     x86 experiment table, cells A, B, C
intel/env.sh.example      x86 per-machine config template
<isa>/results/            CSVs, created on first run, gitignored
```

On a board that ran the sweep before this split, move the old config and
results into the RISC-V folder, then convert the raw CSV to the current
columns (below):

```bash
mv bench/env.sh bench/riscv/env.sh
mv bench/results bench/riscv/results
bash bench/riscv/run_bench.sh --migrate-csv
```

## Pinning, disassembly checks, perf

Both drivers record how every run was pinned in the raw CSV's `pinning`
column, carried into the summary. Serial arms run under `taskset -c PIN_CORE`.
The omp arm runs one thread per physical core with `OMP_PLACES=cores
OMP_PROC_BIND=close`: on RISC-V under `taskset -c OMP_CPUS` (default `0-7`),
so T threads take the first T of those cores; on x86 under `taskset` to the
first hardware thread of T cores from `PIN_CORE`, which never includes an SMT
sibling.

Before a sweep, every build is disassembled with `common/check_disasm.sh`,
by arm:

| arm | check | fails when |
| --- | --- | --- |
| baseline, omp | scalar | the kernel has any vector instruction: any RVV `v*` on RISC-V; ymm, zmm, mask registers or packed xmm arithmetic on x86 |
| intrinsic | vector | the kernel lacks the gather, FMA and scatter (`vluxei32`, `vfmacc` or `vfmadd`, `vsuxei32`; or `vgatherdps` `vfmadd` `vscatterdps`) |
| autovec | report | never; lists what the compiler vectorized |

On RISC-V a scalar check also refuses to pass if `OBJDUMP` cannot decode an
instruction, and the preflight confirms `OBJDUMP` understands RVV.
`--check-disasm` builds every variant, runs the checks and exits without
benchmarking.

With `PERF=1` the counter columns hold user-space counts for the calling
thread. When the kernel refuses them, they stay empty and `perf_note` says
why, e.g. `unavailable: Permission denied (perf_event_paranoid=4)`. `perf_note`
is `off` when `PERF` is not set.

Every timed run is checked against the scalar reference for its matrix.
`correct` is 1 only when the structure matches exactly and every value is
within `|c - ref| <= 1e-5 + 1e-4 |ref|`, and also no value is further off than
float32 rounding can explain: `4 (k+1) 2^-24 sum|a_ik b_kj|`, with k the A row
length. The run's `err_class` says which case it was: `exact`, `rounding`,
`value` (beyond rounding, or NaN), `structural` (nnz, row_ptr or col_idx
differ) or `failed` (the kernel returned an error), next to `tol_fail` (values
outside the tolerance) and `max_rel_err` / `mean_rel_err` (`|c - ref| / |ref|`
over entries with ref != 0). `analyze.py` prints these per matrix and kernel.
A `value`, `structural` or `failed` run stops the sweep: its rows go to
`results/rejected.csv` and nothing is appended. A `rounding` run outside the
tolerance is recorded with `correct=0` and the sweep continues; that is the
cancellation case seen on 2cubes_sphere, cop20k_A and offshore.

Each warmup and each timed run is the same sequence: C's values are wiped to
NaN, `compute()` runs, the result is validated. Only `compute()` is timed. The
wipe means a kernel that stopped writing C fails instead of passing on the
previous run's output, and validating after warmups too means the first timed
run starts from the same cache state as the rest. `compute()` clears the
accumulators only the first time it sees a workspace; every numeric pass
leaves them zeroed, so the timed region no longer contains a serial memset of
`b_cols x threads` floats.

The harness exits with status 3, and the drivers stop the sweep, when a
precondition for a trustworthy number fails:

- A or B is not canonical CSR (checked at load with `rvsp_csr_check`);
- no reference could be built;
- the reference itself is further from a float64 product than rounding
  allows, so the baseline is checked against something other than itself;
- with `--verify-pinning`, which the drivers always pass, the binding is not
  what was asked for: wrong OpenMP team size, a thread not bound to exactly
  one CPU, two threads on one CPU, or two threads on SMT siblings. The probe
  reads the real team and each thread's CPU and mask, and the result goes in
  `team_size` and `bound_cpus`.

The speedup denominator is the `baseline` arm. `--baseline NAME` narrows it to
a kernel, and still takes that kernel's baseline-arm row when there is one, so
the autovec build of the same `scalar_f32` source can never become the
denominator. Two candidate denominators for one matrix is an error.

The raw CSV gained `pinning`, `perf_note`, the four error columns and
`warmup`, `git_commit` (with `-dirty` for uncommitted changes), `team_size`
and `bound_cpus`. `--migrate-csv` converts an
older RISC-V raw CSV: it backs the file up to `spgemm_raw.csv.pre-pinning.bak`,
moves the old omp rows (which ran every thread on one core) to
`spgemm_raw_omp_singlecore.csv`, and marks the remaining rows `unrecorded`.

## Running a sweep (RISC-V)

Set `CC` to a compiler that can build and run `-march=rv64gcv`, then run the
preflight. It builds a small vector program to confirm the toolchain and warns
about anything that would make small effects unreliable.

```bash
cp bench/riscv/env.sh.example bench/riscv/env.sh   # then set CC
bash bench/riscv/run_bench.sh --check
```

Fetch the matrices.

```bash
cd matrices && bash getResources.sh && cd ..
```

The full sweep takes hours, so run it in a detachable session. If it stops, run
the same command again and completed configs are skipped.

```bash
mkdir -p bench/riscv/results
tmux new -s bench
bash bench/riscv/run_bench.sh 2>&1 | tee -a bench/riscv/results/run.log
```

Then summarise:

```bash
python3 bench/common/analyze.py bench/riscv/results/spgemm_raw.csv --csv-out bench/riscv/results/summary.csv
```

`spgemm_raw.csv` holds one row per timed run and is only ever appended to.
`summary.csv` holds one row per config, with medians and confidence intervals.

Figures are produced separately, from `paper/`. Copy the summary across and
run the plot scripts; see [paper/README.md](../paper/README.md).

```bash
cp bench/riscv/results/summary.csv paper/data/
for f in paper/plots/*.py; do python3 "$f"; done
```

To reproduce a published table, run the sweep in one pass as above and
summarise once. Speedups are computed within a single file, so the baseline and
the kernels being scored against it have to be in the same raw CSV.

## Narrowing the run

The whole table runs by default. To run less:

```bash
bash bench/riscv/run_bench.sh --kernels rvv_f32_m2   # one kernel and its baseline
bash bench/riscv/run_bench.sh --kernels intrinsic    # every row with that arm
bash bench/riscv/run_bench.sh --dtype f32 --runs 30
```

| Flag | Meaning |
| --- | --- |
| `--check` | preflight only |
| `--check-matrices` | verify canonical CSR over the matrix set, then exit |
| `--dtype` | restrict by dtype |
| `--kernels` | restrict by kernel name or arm |
| `--baseline NAME` | speedup denominator, default `scalar_f32` |
| `--runs N` | timed runs per config, default 10 |
| `--experiments PATH` | alternate experiment table |

Runs resume safely. A config counts as done only at exactly `RUNS` rows, and a
partial group is rerun rather than topped up, so interrupting a sweep is fine.

Narrowing is safe to combine with a full sweep. Every run appends to the same
raw CSV, `--kernels` only chooses which configs execute, and a filter that
excludes the baseline gets it added back so speedups still compute. Several
narrow runs therefore accumulate into one file and one summary.

The way to get this wrong is to set `CSV` or `OUT_DIR` and split runs across
separate files. A summary can only divide by a baseline present in its own
file, so a CSV holding just the vector kernels yields blank speedups. Keep one
raw file unless you are deliberately separating a second toolchain.

## Running a sweep (Intel)

The x86 arm runs three cells on the same 19 matrices, with the same warmups,
runs, reference check and resume rules as the RISC-V sweep.

| Cell | arm | kernel | build | threads |
| --- | --- | --- | --- | --- |
| A | baseline | `scalar_f32` | `-march=native -fno-tree-vectorize -fno-tree-slp-vectorize` | 1 |
| – | autovec | `scalar_f32` | `-march=native` | 1 |
| B | intrinsic | `avx512_f32` | `-march=native` | 1 |
| C | omp | `scalar_omp_f32` | `-march=native -fopenmp -fno-tree-vectorize -fno-tree-slp-vectorize` | 1 2 4 8 |

`CC` must be GCC. Cell C's cflags and thread counts match the RISC-V omp rows
exactly.

```bash
cp bench/intel/env.sh.example bench/intel/env.sh   # then set CC
bash bench/intel/run_bench.sh --check
tmux new -s bench
bash bench/intel/run_bench.sh 2>&1 | tee -a bench/intel/results/run.log
```

The driver summarises at the end. `results/summary.csv` has the RISC-V
summary's schema, and `results/summary_tagged.csv` is the same with
`machine`, `isa`, `compiler` and `flags` columns appended. `flags` is the exact compile line CMake used for each variant, and
`env.txt` records the compiler banner, what `-march=native` resolved to, the
pinning, and the perf counter status.

To tag a RISC-V summary the same way so the two can be concatenated:

```bash
python3 bench/common/tag_summary.py bench/riscv/results/summary.csv \
    --out bench/riscv/results/summary_tagged.csv \
    --machine "SpacemiT X60" --isa "rv64gcv" --compiler "gcc 16.1.0" \
    --base-flags "-O3 -DNDEBUG -std=c11 -Wall -Wextra -Wpedantic -flto"
```

## Analysis

`analyze.py` groups by matrix, build, cflags, kernel, and dtype, and reports
median, spread, GOP/s, and a speedup with a bootstrap 95% confidence interval. A
config whose interval includes 1.0 is flagged not significant.

The speedup denominator is the `baseline` arm for the same matrix. Every arm
builds at `rv64gcv`, so the baseline is not a different ISA, it is the same
scalar source with the vectorizer switched off. `autovec` and `baseline` are
then the same code compiled for the same target, and the only thing that varies
is whether the compiler was allowed to vectorize. `--baseline NAME` re-scores
against a different kernel without rerunning anything.

## The experiment table

`experiments.tsv` is the only place experiments are declared, one tab-separated
row per cell of the sweep.

```text
arm        kernel       dtype  build  cflags
baseline   scalar_f32   f32    gcv    -fno-tree-vectorize -fno-tree-slp-vectorize
autovec    scalar_f32   f32    gcv    -
intrinsic  rvv_f32_m2   f32    gcv    -
```

`arm` is what the row is evidence for, `kernel` must match a name in `bench.c`,
`build` is `gc` or `gcv`, and `cflags` is extra compile flags or `-`. An arm is
a kernel and a compilation together, which is why the first two rows share a
kernel and a build and differ only in cflags. Each distinct `(build, cflags)`
combination is its own binary and its own config. A vector kernel on
`build=gc` is rejected before building, since it would not link.

The no-vectorize flags are GCC spelling. Clang ignores `-fno-tree-vectorize`
with a warning rather than an error, so a clang build of that row would be
silently vectorized and stop being a baseline. Use `-fno-vectorize
-fno-slp-vectorize` there.

## Adding a kernel

Build the kernel first, following the pattern in
[src/kernels/spgemm/README.md](../src/kernels/spgemm/README.md). Then three
edits put it in the sweep.

Register it in `common/bench.c`, guarding both the wrapper and the registry entry with
`__riscv_vector` (or `__AVX512F__` on x86) for a vector kernel. The last two registry fields are the
validation reference. Point them at a kernel of the same family, and every run
is checked against it, with a mismatch recorded as `correct=0`.

```c
#if defined(__riscv_vector)
KERNEL_WRAP(mykernel_f32_w, rvsp_spgemm_mykernel_f32)
static const kernel_entry_t KERNELS[] = {
    { "mykernel_f32", mykernel_f32_w, RVSP_SPGEMM_ALGO_MYKERNEL, 0,
      RVSP_DTYPE_FP32, "f32", scalar_f32_w, "scalar_f32" },
};
#endif
```

The fourth field is the LMUL, or 0 for a kernel that is not RVV. It feeds the
`lmul` CSV column, so the sweep never parses it out of the kernel name.

Add a row to the ISA's `experiments.tsv`. If the kernel needs the V extension, add its
name to `vector_kernel()` in `run_bench.sh` so a bad `gc` pairing fails before
the sweep rather than at link time, and to the exclusion regex in
`CMakeLists.txt`. Then smoke-test before committing to a long
run.

```bash
bash bench/riscv/run_bench.sh --kernels mykernel_f32 --runs 2
```

A compile-time tunable is swept with one row per setting, each costing an extra
build.

```text
intrinsic	mykernel_f32	f32	gcv	-DRVSP_MYKERNEL_THRESHOLD=8
intrinsic	mykernel_f32	f32	gcv	-DRVSP_MYKERNEL_THRESHOLD=32
```
