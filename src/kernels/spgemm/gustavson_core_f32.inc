/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Gustavson SpGEMM driver. Each kernel supplies its own rvsp_accum_row and
 * includes this file to generate the numeric pass and the public entry point.
 */

#ifndef RVSP_KERNEL_NAME
#error "define RVSP_KERNEL_NAME before including gustavson_core_f32.inc"
#endif

#define RVSP_CAT_(a, b) a##b
#define RVSP_CAT(a, b) RVSP_CAT_(a, b)
#define RVSP_NUMERIC_FN RVSP_CAT(RVSP_KERNEL_NAME, _numeric)

void RVSP_NUMERIC_FN(
    int32_t a_rows,
    int32_t b_cols,
    const int32_t *RVSP_RESTRICT a_row_ptr,
    const int32_t *RVSP_RESTRICT a_col_idx,
    const float *RVSP_RESTRICT a_values,
    const int32_t *RVSP_RESTRICT b_row_ptr,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc,
    const int32_t *RVSP_RESTRICT c_row_ptr,
    const int32_t *RVSP_RESTRICT c_col_idx,
    float *RVSP_RESTRICT c_values)
{
    (void)b_cols;

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
                acc);
        }

        /* Drain the touched acc[] slots into C and clear them for the next
         * row, so acc stays zeroed without a full sweep. */
        const int32_t c_start = c_row_ptr[row];
        const int32_t c_end = c_row_ptr[row + 1];

        for (int32_t i = c_start; i < c_end; i++)
        {
            const int32_t col = c_col_idx[i];

            c_values[i] = acc[col];
            acc[col] = 0.0f;
        }
    }
}

rvsp_status_t RVSP_KERNEL_NAME(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *a_row_ptr,
    const int32_t *a_col_idx,
    const float *a_values,
    const int32_t *b_row_ptr,
    const int32_t *b_col_idx,
    const float *b_values,
    void *workspace,
    size_t workspace_bytes,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    float **c_values_out,
    int32_t *c_nnz_out,
    int64_t *op_counts_out)
{
    (void)a_cols;

    if (a_row_ptr == NULL || a_col_idx == NULL || a_values == NULL ||
        b_row_ptr == NULL || b_col_idx == NULL || b_values == NULL ||
        c_row_ptr_out == NULL || c_col_idx_out == NULL ||
        c_values_out == NULL || c_nnz_out == NULL || workspace == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (a_rows <= 0 || a_cols <= 0 || b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (workspace_bytes < rvsp_ws_bytes(b_cols))
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    *c_row_ptr_out = NULL;
    *c_col_idx_out = NULL;
    *c_values_out = NULL;
    *c_nnz_out = 0;

    rvsp_ws_t ws;
    rvsp_ws_bind(&ws, workspace, b_cols);

    memset(ws.acc, 0, (size_t)b_cols * sizeof(float));
    memset(ws.mark, 0, (size_t)b_cols * sizeof(uint8_t));

    int32_t *c_row_ptr =
        (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));

    if (c_row_ptr == NULL)
    {
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    int64_t total_nnz = 0;

    /* Symbolic pass first, so C can be sized exactly before the numeric pass. */
    rvsp_status_t status =
        rvsp_symbolic_count(
            a_rows,
            a_row_ptr,
            a_col_idx,
            b_row_ptr,
            b_col_idx,
            ws.mark,
            ws.touched,
            c_row_ptr,
            op_counts_out,
            &total_nnz);

    if (status != RVSP_SUCCESS)
    {
        free(c_row_ptr);
        return status;
    }

    /* malloc(0) is implementation-defined, so floor the allocation at 1. */
    const size_t alloc_nnz =
        total_nnz > 0 ? (size_t)total_nnz : 1;

    int32_t *c_col_idx =
        (int32_t *)malloc(alloc_nnz * sizeof(int32_t));

    float *c_values =
        (float *)malloc(alloc_nnz * sizeof(float));

    if (c_col_idx == NULL || c_values == NULL)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    rvsp_symbolic_fill(
        a_rows,
        b_cols,
        a_row_ptr,
        a_col_idx,
        b_row_ptr,
        b_col_idx,
        ws.mark,
        ws.scratch,
        c_row_ptr,
        c_col_idx);

    RVSP_NUMERIC_FN(
        a_rows,
        b_cols,
        a_row_ptr,
        a_col_idx,
        a_values,
        b_row_ptr,
        b_col_idx,
        b_values,
        ws.acc,
        c_row_ptr,
        c_col_idx,
        c_values);

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = c_row_ptr[a_rows];

    return RVSP_SUCCESS;
}