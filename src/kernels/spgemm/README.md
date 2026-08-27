# rv-sparse SpGEMM kernels

Five fp32 CSR SpGEMM kernels using the same Gustavson driver and differing only
in the accumulate loop.

## Files

```text
rvsp_spgemm.h             kernel and phase declarations
rvsp_common.h             workspace layouts and compiler helpers
rvsp_sort.h               comparator free column sort
gustavson_core_f32.inc    shared driver, not compiled alone

rvsp_support.c            workspace query and CSR validation
rvsp_symbolic.c           symbolic count and fill

accum_scalar_f32.c        scalar accumulation
accum_scalar_omp_f32.c    scalar, row loop over OpenMP threads
accum_rvv_f32_m1.c        gather, FMA, scatter at LMUL=1
accum_rvv_f32_m2.c        gather, FMA, scatter at LMUL=2
accum_rvv_f32_m4.c        gather, FMA, scatter at LMUL=4
```

A kernel file defines `rvsp_accum_row`, then defines `RVSP_KERNEL_NAME` and
includes the driver, which emits both the numeric function and the one-shot
entry point. The OpenMP kernel is the exception: its parallel region lives in
the numeric function, so it writes that function directly.

The RVV sources `#error` without the V extension rather than falling back to
scalar, and `CMakeLists.txt` excludes them when `RVSP_ARCH_FLAGS` does not
enable it.
