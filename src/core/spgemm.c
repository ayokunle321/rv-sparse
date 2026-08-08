/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * rv-sparse is an experimental sparse linear algebra library focused on
 * portable sparse kernels and future RISC-V optimization.
 *
 * This source file implements part of the internal backend infrastructure
 * used by the public API.
 */

/*Note:
 * this is a basic dispacher, inspired in blas, from her the library decided what kernel use
 * under revision
 * todo things :  structure for add new kernels easily..
 */

#include <stdlib.h>

#include "rv_sparse.h"
#include "../kernels/spgemm/csr_spgemm_kernels.h"
#include "../kernels/spgemm/v2/rvsp_v2.h"

/*
 * ---------------------------------------------------------------------------
 * MIXED SEMANTICS, DELIBERATE
 * ---------------------------------------------------------------------------
 * fp32 runs on the v2 kernels; fp64 and int8 still run on v1. The two families
 * disagree about one thing, and callers need to know which one they are on:
 *
 *   v2 (fp32)      structural nonzeros are RETAINED even when they cancel to
 *                  exactly zero, matching cuSPARSE and rocSPARSE.
 *   v1 (fp64, i8)  entries that cancel to exactly zero are DROPPED.
 *
 * So for fp32, nnz(C) depends only on the patterns of A and B; for fp64 and
 * int8 it can also depend on their values. The split exists because v2 is
 * fp32 only today. When v2 grows fp64 and int8 kernels these arms move over
 * and the inconsistency goes away.
 *
 * fp32 also gains a precondition v1 did not have: A and B must be canonical
 * CSR. v2 does not check it and has no bounds checks in its hot loops, so the
 * one shot below verifies it explicitly rather than returning a silently wrong
 * answer. That check is O(nnz) and runs once per call; the descriptor API in
 * spgemm_descr.c is the path for callers who cannot afford it.
 */

/*
 * fp32 one shot, implemented on the two phase descriptor path so there is a
 * single implementation rather than two. Unlike the descriptor API this
 * allocates C itself and sets owns_data, preserving the contract the public
 * one shot has always had: the caller frees C with rvsp_csr_destroy().
 */
static rvsp_status_t spgemm_csr_v2_oneshot(const rvsp_csr_matrix_t *A,
                                           const rvsp_csr_matrix_t *B,
                                           rvsp_csr_matrix_t *C,
                                           rvsp_spgemm_algo_t algo)
{
    if (A->dtype != RVSP_DTYPE_FP32 || B->dtype != RVSP_DTYPE_FP32)
    {
        return RVSP_ERROR_UNSUPPORTED_DTYPE;
    }

    if (A->cols != B->rows)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    /* The precondition v2 relies on and does not verify for itself. */
    if (rvsp_csr_check(A->rows, A->cols, A->row_ptr, A->col_idx, NULL)
            != RVSP_CSR_OK ||
        rvsp_csr_check(B->rows, B->cols, B->row_ptr, B->col_idx, NULL)
            != RVSP_CSR_OK)
    {
        return RVSP_ERROR_INVALID_CSR;
    }

    rvsp_spgemm_descr_t descr = NULL;
    rvsp_status_t status = rvsp_spgemm_descr_create(&descr);

    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    status = rvsp_spgemm_set_algo(descr, algo);

    if (status != RVSP_SUCCESS)
    {
        rvsp_spgemm_descr_destroy(descr);
        return status;
    }

    size_t workspace_bytes = 0;
    int32_t c_nnz = 0;

    status = rvsp_spgemm_work_estimation(descr, A, B, &workspace_bytes, &c_nnz);

    if (status != RVSP_SUCCESS)
    {
        rvsp_spgemm_descr_destroy(descr);
        return status;
    }

    const size_t alloc_nnz = c_nnz > 0 ? (size_t)c_nnz : 1;

    int32_t *c_row_ptr =
        (int32_t *)malloc(((size_t)A->rows + 1) * sizeof(int32_t));
    int32_t *c_col_idx = (int32_t *)malloc(alloc_nnz * sizeof(int32_t));
    float *c_values = (float *)malloc(alloc_nnz * sizeof(float));
    void *workspace = malloc(workspace_bytes);

    if (!c_row_ptr || !c_col_idx || !c_values || !workspace)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        free(workspace);
        rvsp_spgemm_descr_destroy(descr);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    /* compute() validates C against the descriptor, so describe it first. */
    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = c_nnz;
    C->row_ptr = c_row_ptr;
    C->col_idx = c_col_idx;
    C->values = c_values;
    C->dtype = RVSP_DTYPE_FP32;
    C->format = RVSP_FORMAT_CSR;
    C->owns_data = 1;

    status = rvsp_spgemm_compute(descr, A, B, C, workspace);

    free(workspace);
    rvsp_spgemm_descr_destroy(descr);

    if (status != RVSP_SUCCESS)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);

        C->row_ptr = NULL;
        C->col_idx = NULL;
        C->values = NULL;
        C->nnz = 0;
        C->owns_data = 0;
    }

    return status;
}

rvsp_status_t rvsp_spgemm_csr(const rvsp_csr_matrix_t *A, const rvsp_csr_matrix_t *B,
                              rvsp_csr_matrix_t *C,
                              const rvsp_spgemm_options_t *options)
{
    rvsp_backend_t backend = RVSP_BACKEND_SCALAR;
    rvsp_dtype_t input_dtype = RVSP_DTYPE_FP32;
    rvsp_dtype_t output_dtype = RVSP_DTYPE_FP32;

    if (!A || !B || !C)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (options)
    {
        backend = options->backend;
        input_dtype = options->input_dtype;
        output_dtype = options->output_dtype;
    }

    /* ---------------------------------------------------------------- */
    /* fp32: v2. No v1 fp32 kernel is reachable from the public API.     */
    /* ---------------------------------------------------------------- */

    /*
     * SCALAR and SCALAR_UNROLL4 both land on the v2 scalar accumulator: it is
     * THE fp32 scalar backend. v2's unroll factor is RVSP_SCALAR_UNROLL, a
     * compile time macro, so there is nothing for a runtime backend enum to
     * select. SCALAR_UNROLL4 is kept as an accepted alias rather than an error
     * so existing callers keep working; the v1 unroll4 kernel is still built
     * and still benchmarked, just no longer publicly dispatchable.
     */
    if ((backend == RVSP_BACKEND_SCALAR ||
         backend == RVSP_BACKEND_SCALAR_UNROLL4) &&
        input_dtype == RVSP_DTYPE_FP32 &&
        output_dtype == RVSP_DTYPE_FP32)
    {
        return spgemm_csr_v2_oneshot(A, B, C, RVSP_SPGEMM_ALGO_DEFAULT);
    }

    /* Unavailable without the V extension: reported, not linked against. */
    if (backend == RVSP_BACKEND_RVV_INTRINSICS &&
        A->dtype == RVSP_DTYPE_FP32 &&
        B->dtype == RVSP_DTYPE_FP32)
    {
        return spgemm_csr_v2_oneshot(A, B, C, RVSP_SPGEMM_ALGO_RVV);
    }

    /* ---------------------------------------------------------------- */
    /* fp64 and int8: still v1. See the semantics note at the top.       */
    /* ---------------------------------------------------------------- */

    if (backend == RVSP_BACKEND_SCALAR &&
        input_dtype == RVSP_DTYPE_INT8 &&
        output_dtype == RVSP_DTYPE_INT32) // review this format of output
    {
        return rvsp_spgemm_csr_scalar_i8(A, B, C);
    }

    if (backend == RVSP_BACKEND_SCALAR &&
        input_dtype == RVSP_DTYPE_FP64 &&
        output_dtype == RVSP_DTYPE_FP64)
    {
        return rvsp_spgemm_csr_scalar_f64(A, B, C);
    }

    if (backend == RVSP_BACKEND_RVV_INTRINSICS &&
        A->dtype == RVSP_DTYPE_INT8 &&
        B->dtype == RVSP_DTYPE_INT8)
    {
        return rvsp_spgemm_csr_rvv_i8_indexed_marked(A, B, C);
    }

    if (backend == RVSP_BACKEND_RVV_INTRINSICS &&
        A->dtype == RVSP_DTYPE_FP64 &&
        B->dtype == RVSP_DTYPE_FP64)
    {
        return rvsp_spgemm_csr_rvv_f64_indexed_marked(A, B, C);
    }

    // default error
    return RVSP_ERROR_UNSUPPORTED_BACKEND;
}
