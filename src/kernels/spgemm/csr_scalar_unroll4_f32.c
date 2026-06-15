/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file implements a scalar FP32 CSR SpGEMM kernel with manual
 * unrolling by a factor of 4 in the inner loop over rows of matrix B.
 *
 * Operation:
 *
 *     C = A * B
 *
 * Format:
 *
 *     CSR FP32 x CSR FP32 -> CSR FP32
 */

#include <stdlib.h>
#include "rv_sparse.h"
#include "csr_spgemm_kernels.h"

rvsp_status_t rvsp_spgemm_csr_scalar_unroll4_f32_raw(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *restrict a_row_ptr,
    const int32_t *restrict a_col_idx,
    const float *restrict a_values,
    const int32_t *restrict b_row_ptr,
    const int32_t *restrict b_col_idx,
    const float *restrict b_values,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    float **c_values_out,
    int32_t *c_nnz_out)
{
    if (!a_row_ptr || !a_col_idx || !a_values ||
        !b_row_ptr || !b_col_idx || !b_values ||
        !c_row_ptr_out || !c_col_idx_out || !c_values_out || !c_nnz_out)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (a_rows <= 0 || a_cols <= 0 || b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t max_nnz = a_rows * b_cols;

    int32_t *c_row_ptr = (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));
    int32_t *c_col_idx = (int32_t *)malloc((size_t)max_nnz * sizeof(int32_t));
    float *c_values = (float *)malloc((size_t)max_nnz * sizeof(float));

    if (!c_row_ptr || !c_col_idx || !c_values)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    int32_t nnz_count = 0;

    for (int32_t row = 0; row < a_rows; row++)
    {
        c_row_ptr[row] = nnz_count;

        float *acc = (float *)calloc((size_t)b_cols, sizeof(float));

        if (!acc)
        {
            free(c_row_ptr);
            free(c_col_idx);
            free(c_values);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }

        int32_t a_start = a_row_ptr[row];
        int32_t a_end = a_row_ptr[row + 1];

        for (int32_t a_pos = a_start; a_pos < a_end; a_pos++)
        {
            int32_t k = a_col_idx[a_pos];

            if (k < 0 || k >= a_cols)
            {
                free(acc);
                free(c_row_ptr);
                free(c_col_idx);
                free(c_values);
                return RVSP_ERROR_INVALID_CSR;
            }

            float a_val = a_values[a_pos];

            int32_t b_start = b_row_ptr[k];
            int32_t b_end = b_row_ptr[k + 1];

            int32_t b_pos = b_start;

            /*
             * Unrolled inner loop by 4.
             *
             * This processes four non-zero elements from row k of B
             * per iteration.
             */
            for (; b_pos + 3 < b_end; b_pos += 4)
            {
                int32_t col0 = b_col_idx[b_pos];
                int32_t col1 = b_col_idx[b_pos + 1];
                int32_t col2 = b_col_idx[b_pos + 2];
                int32_t col3 = b_col_idx[b_pos + 3];

                float val0 = b_values[b_pos];
                float val1 = b_values[b_pos + 1];
                float val2 = b_values[b_pos + 2];
                float val3 = b_values[b_pos + 3];

                acc[col0] += a_val * val0;
                acc[col1] += a_val * val1;
                acc[col2] += a_val * val2;
                acc[col3] += a_val * val3;
            }

            /*
             * Remainder loop.
             *
             * Handles 0 to 3 leftover non-zero elements.
             */
            for (; b_pos < b_end; b_pos++)
            {
                int32_t col = b_col_idx[b_pos];
                float val = b_values[b_pos];

                acc[col] += a_val * val;
            }
        }

        for (int32_t col = 0; col < b_cols; col++)
        {
            if (acc[col] != 0.0f)
            {
                c_col_idx[nnz_count] = col;
                c_values[nnz_count] = acc[col];
                nnz_count++;
            }
        }

        free(acc);
    }

    c_row_ptr[a_rows] = nnz_count;

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = nnz_count;

    return RVSP_SUCCESS;
}