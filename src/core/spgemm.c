/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * Backend dispatch for the public SpGEMM API.
 */

#include <stdlib.h>

#include "rv_sparse.h"
#include "../kernels/spgemm/csr_spgemm_kernels.h"
#include "../kernels/spgemm/v2/rvsp_v2.h"

/*
 * fp32 uses the current kernel family. fp64 and int8 remain on the existing
 * kernels until corresponding implementations are available.
 */

/*
 * The one-shot fp32 path uses the descriptor API so structure analysis and
 * numeric execution share one implementation.
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

    /*
     * The kernel assumes canonical CSR, so validate it once before entering
     * the unchecked hot path.
     */
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

    status = rvsp_spgemm_work_estimation(descr, A, B,
                                         &workspace_bytes, &c_nnz);

    if (status != RVSP_SUCCESS)
    {
        rvsp_spgemm_descr_destroy(descr);
        return status;
    }

    const size_t alloc_nnz = c_nnz > 0 ? (size_t)c_nnz : 1;

    int32_t *c_row_ptr =
        (int32_t *)malloc(((size_t)A->rows + 1) * sizeof(int32_t));
    int32_t *c_col_idx =
        (int32_t *)malloc(alloc_nnz * sizeof(int32_t));
    float *c_values =
        (float *)malloc(alloc_nnz * sizeof(float));
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

    /*
     * Compute writes directly into the caller-owned C buffers.
     */
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

rvsp_status_t rvsp_spgemm_csr(const rvsp_csr_matrix_t *A,
                              const rvsp_csr_matrix_t *B,
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

    /*
     * The fp32 scalar backends use the same implementation. The unroll option
     * remains an alias for compatibility since unrolling is a compile-time setting.
     */
    if ((backend == RVSP_BACKEND_SCALAR ||
         backend == RVSP_BACKEND_SCALAR_UNROLL4) &&
        input_dtype == RVSP_DTYPE_FP32 &&
        output_dtype == RVSP_DTYPE_FP32)
    {
        return spgemm_csr_v2_oneshot(A, B, C, RVSP_SPGEMM_ALGO_DEFAULT);
    }

    /*
     * Vector dispatch is resolved by the build. The vector implementation is
     * only available when the binary was compiled with the V extension.
     */
    if (backend == RVSP_BACKEND_RVV_INTRINSICS &&
        A->dtype == RVSP_DTYPE_FP32 &&
        B->dtype == RVSP_DTYPE_FP32)
    {
        return spgemm_csr_v2_oneshot(A, B, C, RVSP_SPGEMM_ALGO_RVV);
    }

    /*
     * fp64 and int8 continue to use the existing implementations.
     */
    if (backend == RVSP_BACKEND_SCALAR &&
        input_dtype == RVSP_DTYPE_INT8 &&
        output_dtype == RVSP_DTYPE_INT32)
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

    return RVSP_ERROR_UNSUPPORTED_BACKEND;
}
