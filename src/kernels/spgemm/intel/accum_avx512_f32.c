/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Gather-FMA-scatter accumulation on AVX-512, a one-to-one port of
 * riscv/accum_rvv_f32_m2.c.
 *
 * LMUL=2 is the variant ported because it is the RVSP_SPGEMM_ALGO_RVV alias,
 * and because at the X60's VLEN of 256 an e32m2 group holds 16 fp32 lanes,
 * exactly one zmm register. Each strip therefore covers the same nonzeros of
 * the B row on both machines, lane i taking nonzero p + i.
 *
 * Nothing is reordered or reformatted. Each statement below is the AVX-512
 * spelling of the RVV statement named in its comment, in the same order.
 */

#include "rvsp_common.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#else
#error "accum_avx512_f32.c requires AVX-512F"
#endif

/* VLMAX for e32m2 at VLEN=256, and the fp32 width of a zmm register. */
#define RVSP_AVX512_LANES 16

static inline void rvsp_accum_row(float a_val, int32_t b_nnz,
                                  const int32_t *RVSP_RESTRICT b_col_idx,
                                  const float *RVSP_RESTRICT b_values,
                                  float *RVSP_RESTRICT acc) {
  /* vfmacc.vf takes a_val as a scalar operand; x86 FMA has no scalar-operand
   * form, so the splat that RVV does inside the instruction happens here. */
  const __m512 va = _mm512_set1_ps(a_val);

  int32_t p = 0;

  while (p < b_nnz) {
    /* __riscv_vsetvl_e32m2: vl = min(remaining, VLMAX). AVX-512 has no vl
     * register, so vl becomes a mask of the low vl lanes and every memory
     * access below runs under it. As in RVV there is no separate tail loop:
     * the last strip is just a strip with a shorter vl. */
    const int32_t rem = b_nnz - p;
    const int32_t vl = rem < RVSP_AVX512_LANES ? rem : RVSP_AVX512_LANES;
    const __mmask16 m = (__mmask16)((1u << vl) - 1u);

    /* __riscv_vle32_v_f32m2 */
    const __m512 vb = _mm512_maskz_loadu_ps(m, &b_values[p]);

    /* __riscv_vle32_v_i32m2 */
    const __m512i vidx = _mm512_maskz_loadu_epi32(m, &b_col_idx[p]);

    /* canonical CSR has no repeated column in a row, so the gather and
     * scatter below cannot alias within one vl */

    /* __riscv_vsll_vx_i32m2(vidx, 2): column index to byte offset, kept as
     * an explicit shift so the gather and scatter take byte offsets (scale 1)
     * exactly like vluxei32 and vsuxei32. Offsets are signed here, so columns
     * must stay below 2^29; RVV's unsigned offsets allow 2^30. */
    const __m512i voff = _mm512_slli_epi32(vidx, 2);

    /* __riscv_vluxei32_v_f32m2: unordered indexed load */
    __m512 vacc = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), m, voff, acc, 1);

    /* __riscv_vfmacc_vf_f32m2: vacc += a_val * vb, fused. Lanes past vl hold
     * zeros and are never stored, as RVV leaves tail lanes unwritten. */
    vacc = _mm512_fmadd_ps(va, vb, vacc);

    /* __riscv_vsuxei32_v_f32m2: unordered indexed store */
    _mm512_mask_i32scatter_ps(acc, m, voff, vacc, 1);

    p += vl;
  }
}

#define RVSP_KERNEL_NAME rvsp_spgemm_avx512_f32
#include "gustavson_core_f32.inc"
