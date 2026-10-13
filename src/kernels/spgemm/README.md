# rv-sparse SpGEMM kernels

fp32 CSR SpGEMM kernels using the same Gustavson driver and differing only in
the accumulate loop. Shared, ISA-neutral code lives in `common/`, and each ISA
has its own folder of kernels. A build compiles `common/` plus exactly one ISA
folder, chosen by `RVSP_ISA` in `CMakeLists.txt` (default: from the host
processor).

## Files

```text
common/rvsp_spgemm.h             kernel and phase declarations
common/rvsp_common.h             workspace layouts and compiler helpers
common/rvsp_sort.h               comparator free column sort
common/gustavson_core_f32.inc    shared driver, not compiled alone
common/rvsp_support.c            workspace query and CSR validation
common/rvsp_symbolic.c           symbolic count and fill

riscv/accum_scalar_f32.c         scalar accumulation
riscv/accum_scalar_omp_f32.c     scalar, row loop over OpenMP threads
riscv/accum_rvv_f32_m1.c         gather, FMA, scatter at LMUL=1
riscv/accum_rvv_f32_m2.c         gather, FMA, scatter at LMUL=2
riscv/accum_rvv_f32_m4.c         gather, FMA, scatter at LMUL=4

intel/accum_scalar_f32.c         copy of riscv/accum_scalar_f32.c
intel/accum_scalar_omp_f32.c     copy of riscv/accum_scalar_omp_f32.c
intel/accum_avx512_f32.c         riscv/accum_rvv_f32_m2.c ported to AVX-512
```

A kernel file defines `rvsp_accum_row`, then defines `RVSP_KERNEL_NAME` and
includes the driver, which emits both the numeric function and the one-shot
entry point. The OpenMP kernel is the exception: its parallel region lives in
the numeric function, so it writes that function directly.

The intel scalar files are the riscv ones under a new header comment, and
define the same symbols, which is why the two folders are never built
together. The AVX-512 kernel is a statement-for-statement port of the LMUL=2
RVV kernel. LMUL=2 at the X60's VLEN of 256 is 16 fp32 lanes, one zmm
register, so each strip covers the same nonzeros on both machines. vsetvl
becomes a lane mask, vle32 a masked load, vluxei32 and vsuxei32 a masked gather
and scatter on byte offsets, and vfmacc.vf an FMA.

Vector sources `#error` without their extension rather than falling back to
scalar, and `CMakeLists.txt` excludes them when the arch flags do not enable
it: the RVV files by the `-march` string, the AVX-512 file by asking the
compiler whether `__AVX512F__` is defined.
