/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * Two-phase SpGEMM API that computes the structure once and reuses it across
 * numeric multiplies with the same sparsity pattern.
 */

#include <stdlib.h>
#include <string.h>

#include "rv_sparse.h"
#include "../kernels/spgemm/v2/rvsp_v2.h"
#include "../kernels/spgemm/v2/rvsp_v2_common.h"

struct rvsp_spgemm_descr
{
    rvsp_spgemm_algo_t algo;
    int estimated;

    /* Structure computed during work_estimation(). */
    int32_t *c_row_ptr;
    int32_t c_nnz;
    int64_t *op_counts;

    /* C->col_idx populated by the most recent symbolic fill. */
    const int32_t *filled_into;

    /* Input dimensions captured by work_estimation(). */
    int32_t a_rows, a_cols, a_nnz;
    int32_t b_rows, b_cols, b_nnz;

    size_t workspace_bytes;
};

/* ------------------------------------------------------------------ */
/* Algorithm table                                                     */
/* ------------------------------------------------------------------ */

typedef void (*rvsp_numeric_fn)(RVSP_NUMERIC_PARAMS);

/* Return the numeric implementation available for the selected algorithm. */
static rvsp_numeric_fn numeric_for(rvsp_spgemm_algo_t algo)
{
    switch (algo)
    {
    case RVSP_SPGEMM_ALGO_DEFAULT:
        return rvsp_spgemm_scalar_f32_numeric;

#if defined(__riscv_vector)
    case RVSP_SPGEMM_ALGO_RVV:
        return rvsp_spgemm_rvv_f32_numeric;

    case RVSP_SPGEMM_ALGO_CONTIG:
        return rvsp_spgemm_contig_f32_numeric;

    case RVSP_SPGEMM_ALGO_ADAPTIVE:
        return rvsp_spgemm_adaptive_f32_numeric;
#endif

    default:
        return NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

rvsp_status_t rvsp_spgemm_descr_create(rvsp_spgemm_descr_t *descr)
{
    if (descr == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    *descr = NULL;

    struct rvsp_spgemm_descr *d =
        (struct rvsp_spgemm_descr *)calloc(1, sizeof(*d));

    if (d == NULL)
    {
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    d->algo = RVSP_SPGEMM_ALGO_DEFAULT;

    *descr = d;

    return RVSP_SUCCESS;
}

void rvsp_spgemm_descr_destroy(rvsp_spgemm_descr_t descr)
{
    if (descr == NULL)
    {
        return;
    }

    free(descr->c_row_ptr);
    free(descr->op_counts);
    free(descr);
}

rvsp_status_t rvsp_spgemm_set_algo(rvsp_spgemm_descr_t descr,
                                   rvsp_spgemm_algo_t algo)
{
    if (descr == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    /* Algorithm selection must happen before structure estimation. */
    if (descr->estimated)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (numeric_for(algo) == NULL)
    {
        return RVSP_ERROR_UNSUPPORTED_BACKEND;
    }

    descr->algo = algo;

    return RVSP_SUCCESS;
}

void rvsp_spgemm_invalidate_structure(rvsp_spgemm_descr_t descr)
{
    if (descr != NULL)
    {
        descr->filled_into = NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Shared input checking                                               */
/* ------------------------------------------------------------------ */

static rvsp_status_t check_operand(const rvsp_csr_matrix_t *M)
{
    if (M == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (M->row_ptr == NULL || M->col_idx == NULL || M->values == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (M->dtype != RVSP_DTYPE_FP32)
    {
        return RVSP_ERROR_UNSUPPORTED_DTYPE;
    }

    if (M->rows <= 0 || M->cols <= 0 || M->nnz < 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    return RVSP_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Phase 1: structure                                                  */
/* ------------------------------------------------------------------ */

rvsp_status_t rvsp_spgemm_work_estimation(rvsp_spgemm_descr_t descr,
                                          const rvsp_csr_matrix_t *A,
                                          const rvsp_csr_matrix_t *B,
                                          size_t *workspace_bytes,
                                          int32_t *c_nnz_out)
{
    if (descr == NULL || workspace_bytes == NULL || c_nnz_out == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    rvsp_status_t status = check_operand(A);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    status = check_operand(B);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    if (A->cols != B->rows)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    /* A new estimation replaces the descriptor's previous structure. */
    free(descr->c_row_ptr);
    free(descr->op_counts);
    descr->c_row_ptr = NULL;
    descr->op_counts = NULL;
    descr->filled_into = NULL;
    descr->estimated = 0;
    descr->c_nnz = 0;

    void *count_ws = malloc(rvsp_count_ws_bytes(B->cols));
    int32_t *c_row_ptr =
        (int32_t *)calloc((size_t)A->rows + 1, sizeof(int32_t));
    int64_t *op_counts =
        (int64_t *)malloc((size_t)A->rows * sizeof(int64_t));

    if (count_ws == NULL || c_row_ptr == NULL || op_counts == NULL)
    {
        free(count_ws);
        free(c_row_ptr);
        free(op_counts);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    rvsp_ws_t ws;
    rvsp_count_ws_bind(&ws, count_ws, B->cols);

    /* symbolic_count expects cleared marks. */
    memset(ws.mark, 0, (size_t)B->cols * sizeof(uint8_t));

    int64_t total_nnz = 0;

    status = rvsp_symbolic_count(A->rows,
                                 A->row_ptr, A->col_idx,
                                 B->row_ptr, B->col_idx,
                                 ws.mark, ws.touched,
                                 c_row_ptr,
                                 op_counts,
                                 &total_nnz);

    free(count_ws);

    if (status != RVSP_SUCCESS)
    {
        free(c_row_ptr);
        free(op_counts);
        return status;
    }

    descr->c_row_ptr = c_row_ptr;
    descr->op_counts = op_counts;
    descr->c_nnz = (int32_t)total_nnz;

    descr->a_rows = A->rows;
    descr->a_cols = A->cols;
    descr->a_nnz = A->nnz;
    descr->b_rows = B->rows;
    descr->b_cols = B->cols;
    descr->b_nnz = B->nnz;

    descr->workspace_bytes = rvsp_compute_ws_bytes(B->cols);
    descr->estimated = 1;

    *workspace_bytes = descr->workspace_bytes;
    *c_nnz_out = descr->c_nnz;

    return RVSP_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Phase 2: values                                                     */
/* ------------------------------------------------------------------ */

rvsp_status_t rvsp_spgemm_compute(rvsp_spgemm_descr_t descr,
                                  const rvsp_csr_matrix_t *A,
                                  const rvsp_csr_matrix_t *B,
                                  rvsp_csr_matrix_t *C,
                                  void *workspace)
{
    if (descr == NULL || C == NULL || workspace == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (!descr->estimated)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    rvsp_status_t status = check_operand(A);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    status = check_operand(B);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    /* A and B must match the inputs used during structure estimation. */
    if (A->rows != descr->a_rows || A->cols != descr->a_cols ||
        A->nnz != descr->a_nnz ||
        B->rows != descr->b_rows || B->cols != descr->b_cols ||
        B->nnz != descr->b_nnz)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (C->row_ptr == NULL || C->col_idx == NULL || C->values == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (C->dtype != RVSP_DTYPE_FP32)
    {
        return RVSP_ERROR_UNSUPPORTED_DTYPE;
    }

    if (C->rows != descr->a_rows || C->cols != descr->b_cols)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (C->nnz < descr->c_nnz)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    const rvsp_numeric_fn numeric = numeric_for(descr->algo);

    if (numeric == NULL)
    {
        return RVSP_ERROR_UNSUPPORTED_BACKEND;
    }

    rvsp_ws_t ws;
    rvsp_compute_ws_bind(&ws, workspace, descr->b_cols);

    /* Restore the precomputed row structure for this output. */
    memcpy(C->row_ptr, descr->c_row_ptr,
           ((size_t)descr->a_rows + 1) * sizeof(int32_t));

    /*
     * Reuse C->col_idx when the caller is using the same output storage.
     * Call rvsp_spgemm_invalidate_structure() if that storage was replaced.
     */
    if (descr->filled_into != C->col_idx)
    {
        memset(ws.mark, 0, (size_t)descr->b_cols * sizeof(uint8_t));

        rvsp_symbolic_fill(descr->a_rows, descr->b_cols,
                           A->row_ptr, A->col_idx,
                           B->row_ptr, B->col_idx,
                           ws.mark, ws.scratch,
                           descr->c_row_ptr, C->col_idx);

        descr->filled_into = C->col_idx;
    }

    /* numeric requires a cleared accumulator. */
    memset(ws.acc, 0, (size_t)descr->b_cols * sizeof(float));

    numeric(descr->a_rows,
            A->row_ptr, A->col_idx, (const float *)A->values,
            B->row_ptr, B->col_idx, (const float *)B->values,
            ws.acc,
            descr->c_row_ptr, C->col_idx, (float *)C->values);

    C->nnz = descr->c_nnz;
    C->format = RVSP_FORMAT_CSR;

    return RVSP_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Diagnostics                                                         */
/* ------------------------------------------------------------------ */

rvsp_status_t rvsp_spgemm_get_op_counts(rvsp_spgemm_descr_t descr,
                                        int64_t *op_counts_out,
                                        int32_t n)
{
    if (descr == NULL || op_counts_out == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (!descr->estimated || descr->op_counts == NULL)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (n < descr->a_rows)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    memcpy(op_counts_out, descr->op_counts,
           (size_t)descr->a_rows * sizeof(int64_t));

    return RVSP_SUCCESS;
}
