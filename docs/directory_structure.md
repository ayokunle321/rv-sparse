# rv-sparse Directory Structure

## Orientation

`include/` is the public surface, and everything under `src/` is private. The
private code splits in two. `src/core/` is the plumbing around a multiply, and
`src/kernels/spgemm/` is the multiply itself, with ISA-neutral code in
`common/` and one folder of kernels per ISA. Everything outside `src/` serves
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
│   └── kernels/spgemm/           SpGEMM kernels, fp32
│       ├── common/               ISA-neutral, built for every target
│       │   ├── rvsp_spgemm.h     kernel and phase declarations
│       │   ├── rvsp_common.h     workspace layout, compiler shims
│       │   ├── rvsp_sort.h       comparator free column sort
│       │   ├── rvsp_symbolic.c   symbolic phases, shared
│       │   ├── rvsp_support.c    buffer size query, canonical CSR check
│       │   └── gustavson_core_f32.inc   shared driver, not compiled alone
│       ├── riscv/                one accumulate loop per file
│       │   ├── accum_scalar_f32.c
│       │   ├── accum_scalar_omp_f32.c
│       │   ├── accum_rvv_f32_m1.c
│       │   ├── accum_rvv_f32_m2.c
│       │   └── accum_rvv_f32_m4.c
│       └── intel/
│           ├── accum_scalar_f32.c
│           ├── accum_scalar_omp_f32.c
│           └── accum_avx512_f32.c
│
├── paper/                        analysis, plots and the microbenchmark
│
├── bench/                        benchmark harness
│   ├── common/                   shared by both machines
│   │   ├── bench.c               timing harness
│   │   ├── csr_check.c           input validator
│   │   ├── analyze.py            statistics
│   │   └── tag_summary.py        machine, isa, compiler, flags columns
│   ├── riscv/                    SpacemiT X60 driver
│   │   ├── run_bench.sh
│   │   ├── experiments.tsv
│   │   └── env.sh.example
│   └── intel/                    x86 driver
│       ├── run_bench.sh
│       ├── experiments.tsv
│       ├── env.sh.example
│       └── check_scalar_disasm.sh
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

Build output goes to `build/`, which is gitignored and safe to delete.

## Notes

Every kernel shares the Gustavson driver in `common/gustavson_core_f32.inc` and
differs only in the accumulate loop, each inlining the driver after defining
`RVSP_KERNEL_NAME`. The symbolic phases live in `common/rvsp_symbolic.c` and
are compiled once rather than per kernel. A build compiles `common/` and one
ISA folder, selected by `RVSP_ISA`.

The three RVV kernels are the same gather, FMA and scatter at LMUL 1, 2 and 4.
The AVX-512 kernel is the LMUL 2 one ported to x86, one statement per RVV
statement.
The OpenMP kernel puts the parallel region in the numeric function rather than
in a one-shot entry point, so it needs one accumulator per thread.

A scalar build contains no vector kernel. The vector sources require the V
extension, and `CMakeLists.txt` excludes them when `RVSP_ARCH_FLAGS` does not
enable it.

The Matrix Market loader in `tools/` canonicalizes on load, sorting columns and
summing duplicate entries.