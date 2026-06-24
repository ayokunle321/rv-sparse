# SpGEMM Benchmark Harness

Benchmarks the scalar CSR SpGEMM kernels and reports arithmetic intensity
(operations per byte moved).

## Usage

```
./run.sh
```

Builds all three kernels and runs a density sweep, writing output to
`results.txt`. To run one kernel manually:

```
riscv64-linux-gnu-gcc -O3 -march=rv64gc -static -Iinclude -DKERNEL_F32 \
    bench_csr_spgemm.c ../src/kernels/spgemm/csr_scalar_f32.c -o bench_f32
qemu-riscv64 ./bench_f32 [M K N density reps]
```

Select the kernel at compile time with one of:
`-DKERNEL_F32`, `-DKERNEL_UNROLL4`, `-DKERNEL_I8`.

## Inputs

Synthetic CSR matrices with **unstructured sparsity**: each
entry is independently a nonzero with probability `density`, so nonzeros are
scattered with no pattern (no banding, blocking, or clustering).

A and B use fixed seeds (1 and 2) so every kernel runs on identical matrices.
The kernel runs reps times (used for instruction-count/timing measurements later; it does not affect the analytical metrics here).

## Metrics

Computed analytically from the matrix structure, not from emulated timers
(unreliable under QEMU).

- **ops** = `2 × Σ over A-nonzeros aᵢₖ of nnz(B row k)` — one multiply and one
  add per (A nonzero × matching B-row nonzero).
- **bytes** — compulsory hot-loop traffic per multiply-add: B column index (4 B)
  + B value + accumulator read-modify-write. FP32: 16 B; INT8/INT32: 13 B.
- **arithmetic intensity** = ops ÷ bytes.

This is a traffic *model*, not measured DRAM traffic; QEMU exposes no cache
statistics.