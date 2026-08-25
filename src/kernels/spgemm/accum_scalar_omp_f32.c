/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Scalar accumulation with the row loop parallelised over OpenMP threads.
 *
 * The parallel region lives in the numeric function rather than in a one-shot
 * entry point, because the benchmark times the numeric phase through
 * rvsp_spgemm_compute.
 */

#include "rvsp_common.h"

#if !defined(_OPENMP)
#error "accum_scalar_omp_f32.c requires OpenMP"
#endif

#include <omp.h>

static inline void
rvsp_accum_row(
    float a_val,
    int32_t n,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc)
{
    for (int32_t p = 0; p < n; p++)
    {
        acc[b_col_idx[p]] += a_val * b_values[p];
    }
}

/*
 * acc is nthreads consecutive accumulators of b_cols floats. Each thread takes
 * its own slice, so no two threads touch the same slot. Scheduling is static so
 * each thread owns a contiguous block of output rows, which matters because C
 * is written per row and the blocks must not overlap.
 */
void rvsp_spgemm_scalar_omp_f32_numeric(RVSP_NUMERIC_PARAMS)
{
    const size_t stride = (size_t)(b_cols > 0 ? b_cols : 1);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float *RVSP_RESTRICT acc_t = acc + (size_t)tid * stride;

#pragma omp for schedule(static)
        for (int32_t row = 0; row < a_rows; row++)
        {
            const int32_t a_start = a_row_ptr[row];
            const int32_t a_end = a_row_ptr[row + 1];

            for (int32_t ap = a_start; ap < a_end; ap++)
            {
                const int32_t k = a_col_idx[ap];
                const float a_val = a_values[ap];
                const int32_t b_start = b_row_ptr[k];
                const int32_t b_end = b_row_ptr[k + 1];

                rvsp_accum_row(
                    a_val,
                    b_end - b_start,
                    &b_col_idx[b_start],
                    &b_values[b_start],
                    acc_t);
            }

            const int32_t c_start = c_row_ptr[row];
            const int32_t c_end = c_row_ptr[row + 1];

            for (int32_t i = c_start; i < c_end; i++)
            {
                const int32_t col = c_col_idx[i];

                c_values[i] = acc_t[col];
                acc_t[col] = 0.0f;
            }
        }
    }
}
