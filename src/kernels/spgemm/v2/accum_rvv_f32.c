/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Gather, FMA, scatter on every segment. No minimum length gate, deliberately.
 *
 * A gate does not just tune performance, it changes what the arm measures. The
 * behaviour on 2 element rows is the data point that locates the crossover.
 * Gating belongs in the adaptive kernel.
 *
 * The read modify write is safe only because no column index repeats inside one
 * row of B. Two lanes on the same index would both read the pre-update value
 * and the second scatter would win, dropping a contribution. Canonical CSR
 * rules that out, checked once at load.
 *
 * m2 keeps the gather at 16 fp32 per instruction on VLEN=256 with low register
 * pressure. Longer LMUL mostly lengthens the stall on a core that cracks
 * indexed accesses into per element micro-ops. Sweep RVSP_RVV_LMUL for the curve.
 */

#include "rvsp_v2_common.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#else
#error "accum_rvv_f32.c requires a -march with the V extension (rv64gcv)"
#endif

static inline void rvsp_accum_row(
    float a_val,
    int32_t b_nnz,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc)
{
    int32_t p = 0;

    while (p < b_nnz)
    {
        const size_t vl = __riscv_vsetvl_e32m2((size_t)(b_nnz - p));

        /* Both operand streams are unit stride; only the accumulator is not. */
        const vfloat32m2_t vb = __riscv_vle32_v_f32m2(&b_values[p], vl);
        const vint32m2_t vidx = __riscv_vle32_v_i32m2(&b_col_idx[p], vl);

        /* Column index -> byte offset for a 4 byte element. */
        const vuint32m2_t voff = __riscv_vreinterpret_v_i32m2_u32m2(
            __riscv_vsll_vx_i32m2(vidx, 2, vl));

        vfloat32m2_t vacc = __riscv_vluxei32_v_f32m2(acc, voff, vl);
        vacc = __riscv_vfmacc_vf_f32m2(vacc, a_val, vb, vl);
        __riscv_vsuxei32_v_f32m2(acc, voff, vacc, vl);

        p += (int32_t)vl;
    }
}

#define RVSP_KERNEL_NAME rvsp_spgemm_rvv_f32
#include "gustavson_core_f32.inc"
