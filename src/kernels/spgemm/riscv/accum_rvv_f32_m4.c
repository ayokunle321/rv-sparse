/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Gather-FMA-scatter accumulation at LMUL=4.
 */

#include "rvsp_common.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#else
#error "accum_rvv_f32_m4.c requires the RISC-V V extension"
#endif

static inline void rvsp_accum_row(float a_val, int32_t b_nnz,
                                  const int32_t *RVSP_RESTRICT b_col_idx,
                                  const float *RVSP_RESTRICT b_values,
                                  float *RVSP_RESTRICT acc) {
  int32_t p = 0;

  while (p < b_nnz) {
    const size_t vl = __riscv_vsetvl_e32m4((size_t)(b_nnz - p));

    const vfloat32m4_t vb = __riscv_vle32_v_f32m4(&b_values[p], vl);

    const vint32m4_t vidx = __riscv_vle32_v_i32m4(&b_col_idx[p], vl);

    /* canonical CSR has no repeated column in a row, so the gather and
     * scatter below cannot alias within one vl */
    const vuint32m4_t voff =
        __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vsll_vx_i32m4(vidx, 2, vl));

    vfloat32m4_t vacc = __riscv_vluxei32_v_f32m4(acc, voff, vl);

    vacc = __riscv_vfmacc_vf_f32m4(vacc, a_val, vb, vl);

    __riscv_vsuxei32_v_f32m4(acc, voff, vacc, vl);

    p += (int32_t)vl;
  }
}

#define RVSP_KERNEL_NAME rvsp_spgemm_rvv_f32_m4
#include "gustavson_core_f32.inc"
