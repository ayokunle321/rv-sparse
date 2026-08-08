/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Scalar accumulation with configurable loop unrolling.
 */

#include "rvsp_common.h"

/*
 * 0 emits a plain loop and leaves all instruction-level parallelism to the
 * compiler, which is what makes the baseline-vs-autovec pair measure the
 * compiler alone. 4 and 8 hand it independent FMA chains.
 */
#ifndef RVSP_SCALAR_UNROLL
#define RVSP_SCALAR_UNROLL 0
#endif

#if RVSP_SCALAR_UNROLL != 0 && RVSP_SCALAR_UNROLL != 4 && RVSP_SCALAR_UNROLL != 8
#error "RVSP_SCALAR_UNROLL must be 0, 4 or 8"
#endif

static inline void
rvsp_accum_row(
    float a_val,
    int32_t b_nnz,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc)
{
    int32_t p = 0;

#if RVSP_SCALAR_UNROLL == 8
    for (; p + 7 < b_nnz; p += 8)
    {
        const int32_t c0 = b_col_idx[p + 0];
        const int32_t c1 = b_col_idx[p + 1];
        const int32_t c2 = b_col_idx[p + 2];
        const int32_t c3 = b_col_idx[p + 3];
        const int32_t c4 = b_col_idx[p + 4];
        const int32_t c5 = b_col_idx[p + 5];
        const int32_t c6 = b_col_idx[p + 6];
        const int32_t c7 = b_col_idx[p + 7];

        acc[c0] += a_val * b_values[p + 0];
        acc[c1] += a_val * b_values[p + 1];
        acc[c2] += a_val * b_values[p + 2];
        acc[c3] += a_val * b_values[p + 3];
        acc[c4] += a_val * b_values[p + 4];
        acc[c5] += a_val * b_values[p + 5];
        acc[c6] += a_val * b_values[p + 6];
        acc[c7] += a_val * b_values[p + 7];
    }
#endif

#if RVSP_SCALAR_UNROLL >= 4
    for (; p + 3 < b_nnz; p += 4)
    {
        const int32_t c0 = b_col_idx[p + 0];
        const int32_t c1 = b_col_idx[p + 1];
        const int32_t c2 = b_col_idx[p + 2];
        const int32_t c3 = b_col_idx[p + 3];

        acc[c0] += a_val * b_values[p + 0];
        acc[c1] += a_val * b_values[p + 1];
        acc[c2] += a_val * b_values[p + 2];
        acc[c3] += a_val * b_values[p + 3];
    }
#endif

    for (; p < b_nnz; p++)
    {
        acc[b_col_idx[p]] += a_val * b_values[p];
    }
}

#define RVSP_KERNEL_NAME rvsp_spgemm_scalar_f32
#include "gustavson_core_f32.inc"