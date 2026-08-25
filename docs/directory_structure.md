# rv-sparse Directory Structure

## Orientation

`include/` is the public surface, and everything under `src/` is private. The
private code splits in two. `src/core/` is the plumbing around a multiply, and
`src/kernels/spgemm/` is the multiply itself. Everything outside `src/` serves
those kernels. `bench/` measures them, `tools/` generates and loads matrices
while `matrices/` fetches the real ones, and `tests/` and `examples/` exercise
the public API.

## Layout

```text
rv-sparse/
├── include/                      public API
│   ├── rv_sparse.h
│   └── rv_sparse_types.h
│
├── src/
│   ├── core/                     dispatch, matrix lifecycle, error strings
│   │   ├── error.c
│   │   ├── matrix.c
│   │   ├── spgemm.c              one-shot entry point
│   │   └── spgemm_descr.c        two phase descriptor API
│   │
│   └── kernels/spgemm/           current SpGEMM kernels, fp32
│       ├── rvsp_spgemm.h         kernel and phase declarations
│       ├── rvsp_common.h         workspace layout, compiler shims
│       ├── rvsp_sort.h           comparator free column sort
│       ├── rvsp_symbolic.c       symbolic phases, shared
│       ├── rvsp_support.c        buffer size query, canonical CSR check
│       ├── gustavson_core_f32.inc   shared driver, not compiled alone
│       ├── accum_scalar_f32.c    accumulate strategies, one per file
│       ├── accum_rvv_f32.c
│
├── bench/                        benchmark harness
│   ├── experiments.tsv           the experiment table
│   ├── run_bench.sh              driver
│   ├── bench.c                   timing harness
│   ├── csr_check.c               input validator
│   ├── analyze.py                statistics
│   ├── env.sh.example            per machine config template
│   └── results/                  raw and summary CSVs, created on first run
│
├── tools/                        matrix generation and loading
│   ├── include/
│   └── src/                      genmat, mtx_to_csr_formatter, vec
│
├── tests/                        correctness tests
├── examples/                     API usage programs
├── matrices/                     matrix download list and fetch script
└── docs/                         design notes and figures
```

Build output goes to `obj/`, `lib/` and `bin/`. All three are gitignored and safe to delete.

## Notes

All four SpGEMM strategies share the Gustavson driver in
`gustavson_core_f32.inc` and differ only in the accumulate loop, each inlining
it after defining `RVSP_KERNEL_NAME`. The symbolic phases live in
`rvsp_symbolic.c` and are compiled once rather than per strategy.

A scalar build contains no vector kernel. The vector sources require the V
extension, and the Makefile excludes them when `-march` does not enable it.

The Matrix Market loader in `tools/` canonicalizes on load, sorting columns and
summing duplicate entries.