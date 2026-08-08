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

#include <stddef.h> /* size_t, used by the two phase SpGEMM API below */

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
     * ------------------------------------------------------------------
     * Two phase SpGEMM:  C = A * B, CSR, fp32.
     * ------------------------------------------------------------------
     *
     *   descr_create -> [set_algo] -> work_estimation -> compute [-> compute ...]
     *                                                          -> descr_destroy
     *
     * work_estimation() computes C's structure and keeps it in the descriptor.
     * compute() fills the values. C's structure is value independent, because
     * structural nonzeros are retained even when they cancel to exactly zero
     * (matching cuSPARSE and rocSPARSE), so a caller whose values change but
     * whose sparsity pattern does not may call compute() repeatedly against one
     * descriptor and pay the symbolic cost once. That is the case iterative
     * solvers hit, and it is the reason C is caller allocated.
     *
     * PRECONDITION: A and B must be in canonical CSR form (columns ascending
     * within each row, no duplicates, indices in range). Neither call checks
     * this and the kernels contain no bounds checks in their hot loops. Verify
     * once at load time, outside any timed region.
     */

    rvsp_status_t rvsp_spgemm_descr_create(rvsp_spgemm_descr_t *descr);

    /* NULL safe. */
    void rvsp_spgemm_descr_destroy(rvsp_spgemm_descr_t descr);

    /*
     * Valid before work_estimation() only. Afterwards it returns
     * RVSP_ERROR_INVALID_ARGUMENT, so a descriptor's stored structure can never
     * belong to an algorithm other than the one that will consume it.
     */
    rvsp_status_t rvsp_spgemm_set_algo(rvsp_spgemm_descr_t descr,
                                       rvsp_spgemm_algo_t algo);

    /*
     * Structure phase.
     *
     *   workspace_bytes  bytes compute() needs. Allocate once and reuse.
     *   c_nnz_out        EXACT nnz of C, not an upper bound. Size C's col_idx
     *                    and values arrays from this.
     *
     * Re-running discards any structure already stored in the descriptor.
     */
    rvsp_status_t rvsp_spgemm_work_estimation(rvsp_spgemm_descr_t descr,
                                              const rvsp_csr_matrix_t *A,
                                              const rvsp_csr_matrix_t *B,
                                              size_t *workspace_bytes,
                                              int32_t *c_nnz_out);

    /*
     * Numeric phase. Writes C->row_ptr, C->col_idx and C->values, and sets
     * C->nnz to the value work_estimation() reported.
     *
     * C must be caller allocated with room for a_rows + 1 row pointers and at
     * least c_nnz entries; A and B must match the shapes work_estimation() saw.
     * C->col_idx is populated on the first compute against a given C and reused
     * unchanged on subsequent ones, so a reuse loop runs the accumulate pass and
     * nothing else.
     */
    rvsp_status_t rvsp_spgemm_compute(rvsp_spgemm_descr_t descr,
                                      const rvsp_csr_matrix_t *A,
                                      const rvsp_csr_matrix_t *B,
                                      rvsp_csr_matrix_t *C,
                                      void *workspace);

    /*
     * Op_i, the intermediate product count for each output row, that is
     * sum over k in A[i,:] of nnz(B[k,:]). Computed for free during
     * work_estimation() and valid until the next call to it.
     *
     * This is the standard dispatch metric in the SpGEMM literature and the
     * x axis for "which accumulator wins where" plots. `n` must be at least
     * the row count work_estimation() saw.
     */
    rvsp_status_t rvsp_spgemm_get_op_counts(rvsp_spgemm_descr_t descr,
                                            int64_t *op_counts_out,
                                            int32_t n);

    /*
     * Discard the structure stored in the descriptor without discarding the
     * analysis, forcing the next compute() to repopulate C->col_idx.
     *
     * Only needed when computing into a DIFFERENT C than the previous compute()
     * used. compute() detects that case on its own by pointer identity, so this
     * exists for the one situation identity cannot see: a C that was freed and
     * reallocated at the same address.
     */
    void rvsp_spgemm_invalidate_structure(rvsp_spgemm_descr_t descr);

#ifdef __cplusplus
}
#endif

#endif /* RV_SPARSE_H */