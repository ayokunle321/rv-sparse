# rv-sparse SpGEMM kernels

Four fp32 CSR SpGEMM kernels using the same Gustavson driver and differing only
in the accumulation strategy.

## Files

```text
rvsp_spgemm.h             public API
rvsp_common.h             workspace and compiler helpers
rvsp_sort.h               shared column sort
gustavson_core_f32.inc    shared SpGEMM driver

rvsp_support.c            workspace query and CSR validation
rvsp_symbolic.c           symbolic count and fill

accum_scalar_f32.c        scalar accumulation
accum_rvv_f32.c           gather, FMA, scatter
accum_contig_f32.c        unit stride on contiguous runs
accum_adaptive_f32.c      contiguous runs, gather and scalar dispatch
```