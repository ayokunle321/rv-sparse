/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * Two phase SpGEMM built on the v2 kernel family.
 *
 * WHY THE PHASES SPLIT WHERE THEY DO
 * ----------------------------------
 * The obvious split is symbolic in work_estimation, numeric in compute. That
 * would mean work_estimation materialises C's column indices into descriptor
 * owned memory, and every compute copies them into the caller's C. The copy is
 * roughly 8 bytes per output nonzero against the numeric pass's 16 bytes per
 * intermediate product, so its share of traffic is 1 / (2 * Op/nnz_C + 2):
 * about 2% when many intermediate products merge into each output entry, but
 * around 25% when few do, which is exactly the power law graph case. Paying a
 * quarter of the reuse loop for a memcpy defeats the point of the API.
 *
 * So the fill moves into compute and writes straight into C->col_idx:
 *
 *   work_estimation   symbolic_count only -> c_row_ptr, exact c_nnz, Op_i
 *   compute, first    symbolic_fill into C->col_idx, then numeric
 *   compute, repeat   numeric only
 *
 * The reuse loop is then pure accumulate: no copy, no fill, no structural work.
 * Total cost for a single one shot multiply is unchanged, the work just moves
 * across the phase boundary. Two side effects, both good: the descriptor no
 * longer holds an nnz_C sized array (4 bytes per output nonzero it would have
 * pinned for its lifetime), and the caller's workspace drops from four regions
 * to three, since `touched` is only needed by the count.
 */

#include <stdlib.h>
#include <string.h>

#include "rv_sparse.h"
#include "../kernels/spgemm/v2/rvsp_v2.h"
#include "../kernels/spgemm/v2/rvsp_v2_common.h"

struct rvsp_spgemm_descr
{
    rvsp_spgemm_algo_t algo;
    int estimated; /* 0 until work_estimation() succeeds */

    /* C's row offsets, owned here. Column indices deliberately are not: they
     * are written straight into the caller's C by the first compute(). */
    int32_t *c_row_ptr; /* a_rows + 1 */
    int32_t c_nnz;

    int64_t *op_counts; /* a_rows, filled during the count */

    /*
     * The C->col_idx that compute() last populated. Used only to decide whether
     * the fill can be skipped. A different pointer always re-fills, so the
     * fast path is taken exactly in the reuse case and correctness never
     * depends on the guess being right.
     */
    const int32_t *filled_into;

    /*
     * Captured at work_estimation() time; compute() re-checks A and B against
     * these. It cannot cheaply prove the PATTERN is unchanged, so this catches
     * the affordable mistakes: wrong matrix, resized matrix, swapped arguments.
     */
    int32_t a_rows, a_cols, a_nnz;
    int32_t b_rows, b_cols, b_nnz;

    size_t workspace_bytes;
};

/* ------------------------------------------------------------------ */
/* Algorithm table                                                     */
/* ------------------------------------------------------------------ */

typedef void (*rvsp_numeric_fn)(RVSP_NUMERIC_PARAMS);

/*
 * Three of the four v2 strategies exist only in a build with the V extension;
 * their translation units #error out otherwise. So the table is conditional,
 * and an algorithm the current build cannot provide reports
 * RVSP_ERROR_UNSUPPORTED_BACKEND rather than failing to link.
 *
 * Only DEFAULT is wired for now, by design: the enum is the hook for algorithm
 * selection, not an invitation to build selection logic before there are
 * measurements to justify it.
 */
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
        /* Includes the three vector strategies in a build without V: their
         * translation units are gated out by the Makefile, so there is no
         * symbol to return. Reported as unsupported, never as a link error. */
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

    /* Changing it after the fact would leave the descriptor holding structure
     * computed for a different algorithm. Cheap to forbid, confusing to allow. */
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

    /* A re-run replaces whatever was here; drop it first so a failure part way
     * through cannot leave the descriptor holding a stale structure that later
     * looks valid. */
    free(descr->c_row_ptr);
    free(descr->op_counts);
    descr->c_row_ptr = NULL;
    descr->op_counts = NULL;
    descr->filled_into = NULL;
    descr->estimated = 0;
    descr->c_nnz = 0;

    /*
     * Decision (a): the count's scratch is allocated here rather than asked of
     * the caller. The signature reports the workspace size, so it cannot also
     * receive one, and this call runs once per sparsity pattern -- outside the
     * reuse loop the API exists to keep clean.
     */
    void *count_ws = malloc(rvsp_count_ws_bytes(B->cols));
    int32_t *c_row_ptr =
        (int32_t *)calloc((size_t)A->rows + 1, sizeof(int32_t));
    int64_t *op_counts = (int64_t *)malloc((size_t)A->rows * sizeof(int64_t));

    if (count_ws == NULL || c_row_ptr == NULL || op_counts == NULL)
    {
        free(count_ws);
        free(c_row_ptr);
        free(op_counts);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    rvsp_ws_t ws;
    rvsp_count_ws_bind(&ws, count_ws, B->cols);

    /* symbolic_count requires mark all zero on entry and restores it on exit */
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

    /* A and B must be the same operands the structure was computed from. */
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

    /* The bounds check the caller is trusting us to make. */
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

    /* O(rows), negligible beside the accumulate pass, and it keeps C->row_ptr
     * correct even if the caller reset C between calls. */
    memcpy(C->row_ptr, descr->c_row_ptr,
           ((size_t)descr->a_rows + 1) * sizeof(int32_t));

    /*
     * Fill only when C's column indices are not already the ones we wrote. A
     * mismatched pointer re-fills, which is correct but slow; a match skips,
     * which is the reuse case. The one blind spot is a C freed and reallocated
     * at the same address, which rvsp_spgemm_invalidate_structure() exists for.
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

    /* numeric requires acc all zero on entry and restores it on exit */
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
