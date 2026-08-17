/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * OpenMP Gustavson driver. Same accumulate loop as the serial driver, but the
 * numeric pass runs the row loop across threads, each with its own accumulator.
 * The symbolic pass stays serial.
 */

#ifndef RVSP_KERNEL_NAME
#error "define RVSP_KERNEL_NAME before including gustavson_core_omp_f32.inc"
#endif

#include <omp.h>

#define RVSP_CAT_(a, b) a##b
#define RVSP_CAT(a, b) RVSP_CAT_(a, b)
#define RVSP_NUMERIC_FN RVSP_CAT(RVSP_KERNEL_NAME, _numeric)

void RVSP_NUMERIC_FN(
    int32_t a_rows,
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
    /* acc points at one thread's slice, bound by the caller before the region.
     * The serial numeric signature is kept so the kernel file is unchanged. */
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

    *c_row_ptr_out = NULL;
    *c_col_idx_out = NULL;
    *c_values_out = NULL;
    *c_nnz_out = 0;

    const int nthreads = omp_get_max_threads();
    const size_t acc_elems = (size_t)b_cols;

    /* The caller sizes and owns this. Allocating per-thread accumulators here
     * would put the allocation inside the timed region, which at high thread
     * counts dominates the measurement. */
    if (workspace_bytes < rvsp_omp_ws_bytes(b_cols, (int32_t)nthreads))
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    rvsp_omp_ws_t ws;
    rvsp_omp_ws_bind(&ws, workspace, b_cols, (int32_t)nthreads);

    float *acc_all = ws.acc;
    uint8_t *mark = ws.mark;
    int32_t *touched = ws.touched;
    int32_t *scratch = ws.scratch;

    /* The caller owns the buffer but not its contents. */
    memset(acc_all, 0, acc_elems * (size_t)nthreads * sizeof(float));
    memset(mark, 0, acc_elems * sizeof(uint8_t));

    int32_t *c_row_ptr =
        (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));

    if (c_row_ptr == NULL)
    {
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    int64_t total_nnz = 0;

    /* Symbolic pass, serial. */
    rvsp_status_t status =
        rvsp_symbolic_count(
            a_rows,
            a_row_ptr,
            a_col_idx,
            b_row_ptr,
            b_col_idx,
            mark,
            touched,
            c_row_ptr,
            op_counts_out,
            &total_nnz);

    if (status != RVSP_SUCCESS)
    {
        free(c_row_ptr);
        return status;
    }

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
        mark,
        scratch,
        c_row_ptr,
        c_col_idx);

    /* Numeric pass, parallel over rows. Each thread computes a contiguous block
     * of output rows into its own accumulator, so no two threads touch the same
     * acc slot. Static scheduling keeps each thread's rows contiguous, which
     * matters because C is written per row and the row blocks must not overlap. */
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float *acc = acc_all + (size_t)tid * acc_elems;

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
                    acc);
            }

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

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = c_row_ptr[a_rows];

    return RVSP_SUCCESS;
}