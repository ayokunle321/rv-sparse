/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * Public API for the rv-sparse library.
 */

#ifndef RV_SPARSE_H
#define RV_SPARSE_H

#include <stddef.h>

#include "rv_sparse_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define RVSP_VERSION_MAJOR 0
#define RVSP_VERSION_MINOR 1
#define RVSP_VERSION_PATCH 0

    const char *rvsp_get_version(void);

    const char *rvsp_status_to_string(rvsp_status_t status);

    /*
     * CSR matrix lifecycle.
     *
     * rvsp_csr_create() does not take ownership of user-provided arrays.
     * rvsp_spgemm_csr() may allocate output arrays for C.
     * rvsp_csr_destroy() frees internal arrays only when owns_data is set.
     */
    rvsp_status_t rvsp_csr_create(
        rvsp_csr_matrix_t *A,
        int32_t rows,
        int32_t cols,
        int32_t nnz,
        int32_t *row_ptr,
        int32_t *col_idx,
        void *values,
        rvsp_dtype_t dtype);

    rvsp_status_t rvsp_csr_validate(const rvsp_csr_matrix_t *A);

    void rvsp_csr_destroy(rvsp_csr_matrix_t *A);

    /*
     * Sparse matrix-matrix multiplication:
     *
     *     C = A * B
     *
     * Current implementation:
     *     FP32 x FP32 -> FP32 scalar backend.
     *
     * Planned next target:
     *     INT8 x INT8 -> INT32
     */
    rvsp_status_t rvsp_spgemm_csr(
        const rvsp_csr_matrix_t *A,
        const rvsp_csr_matrix_t *B,
        rvsp_csr_matrix_t *C,
        const rvsp_spgemm_options_t *options);

    /*
    * Two phase SpGEMM.
    *
    * Create a descriptor, optionally select an algorithm, perform structure
    * analysis, compute the result, then destroy the descriptor.
    *
    * A descriptor can be reused across compute() calls when the input sparsity
    * pattern is unchanged.
    *
    * A and B must be in canonical CSR form before calling these functions.
    */

    rvsp_status_t rvsp_spgemm_descr_create(rvsp_spgemm_descr_t *descr);

    /*
    * Destroys a descriptor and releases its internal resources.
    *
    * NULL is accepted.
    */
    void rvsp_spgemm_descr_destroy(rvsp_spgemm_descr_t descr);

    /*
    * Selects the algorithm used by the descriptor.
    *
    * Must be called before work_estimation().
    */
    rvsp_status_t rvsp_spgemm_set_algo(rvsp_spgemm_descr_t descr,
                                    rvsp_spgemm_algo_t algo);

    /*
    * Analyzes the structure of A and B and determines the structure of C.
    *
    * workspace_bytes reports the workspace required by compute().
    * c_nnz_out reports the exact number of structural nonzeros in C.
    *
    * Calling work_estimation() again replaces the structure stored in the
    * descriptor.
    */
    rvsp_status_t rvsp_spgemm_work_estimation(rvsp_spgemm_descr_t descr,
                                            const rvsp_csr_matrix_t *A,
                                            const rvsp_csr_matrix_t *B,
                                            size_t *workspace_bytes,
                                            int32_t *c_nnz_out);

    /*
    * Computes C = A * B using the structure produced by work_estimation().
    *
    * C must provide storage for its row pointers, column indices and values.
    * workspace must provide at least workspace_bytes bytes.
    *
    * Repeated calls with the same input structure reuse the analyzed structure.
    */
    rvsp_status_t rvsp_spgemm_compute(rvsp_spgemm_descr_t descr,
                                    const rvsp_csr_matrix_t *A,
                                    const rvsp_csr_matrix_t *B,
                                    rvsp_csr_matrix_t *C,
                                    void *workspace);

    /*
    * Returns the intermediate product count for each output row.
    *
    * The counts are valid until the next call to work_estimation().
    * n must be at least the number of rows analyzed by work_estimation().
    */
    rvsp_status_t rvsp_spgemm_get_op_counts(rvsp_spgemm_descr_t descr,
                                            int64_t *op_counts_out,
                                            int32_t n);

    /*
    * Invalidates the cached output structure while retaining the analysis.
    *
    * The next compute() repopulates C's column indices.
    */
    void rvsp_spgemm_invalidate_structure(rvsp_spgemm_descr_t descr);

#ifdef __cplusplus
}
#endif

#endif /* RV_SPARSE_H */
