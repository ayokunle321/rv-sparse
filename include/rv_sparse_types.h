/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * Public API for the rv-sparse library.
 */

#ifndef RV_SPARSE_TYPES_H
#define RV_SPARSE_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        RVSP_SUCCESS = 0,
        RVSP_ERROR_NULL_POINTER = -1,
        RVSP_ERROR_INVALID_ARGUMENT = -2,
        RVSP_ERROR_INVALID_CSR = -3,
        RVSP_ERROR_UNSUPPORTED_DTYPE = -4,
        RVSP_ERROR_UNSUPPORTED_BACKEND = -5,
        RVSP_ERROR_ALLOCATION_FAILED = -6
    } rvsp_status_t;

    typedef enum
    {
        RVSP_DTYPE_INT8,
        RVSP_DTYPE_INT32,
        RVSP_DTYPE_BF16,
        RVSP_DTYPE_FP32,
        RVSP_DTYPE_FP64
    } rvsp_dtype_t;

    typedef enum
    {
        RVSP_FORMAT_CSR
    } rvsp_format_t;

    typedef enum
    {
        RVSP_BACKEND_SCALAR,
        RVSP_BACKEND_SCALAR_UNROLL4,
        RVSP_BACKEND_GCC_AUTOVEC,
        RVSP_BACKEND_RVV_INTRINSICS
    } rvsp_backend_t;

    typedef struct
    {
        int32_t rows;
        int32_t cols;
        int32_t nnz;

        int32_t *row_ptr;
        int32_t *col_idx;
        void *values;

        rvsp_dtype_t dtype;
        rvsp_format_t format;

        int owns_data;
    } rvsp_csr_matrix_t;

    typedef struct
    {
        rvsp_backend_t backend;
        rvsp_dtype_t input_dtype;
        rvsp_dtype_t output_dtype;
        int sort_output_indices;
    } rvsp_spgemm_options_t;

    /*
     * Opaque SpGEMM operation descriptor.
     *
     * Holds the selected algorithm and the computed structure of C between the
     * two phases. The workspace is NOT held here: it stays caller owned, which
     * is what keeps allocation out of the timed path.
     *
     * Note this is the only opaque type in the library. rvsp_csr_matrix_t above
     * remains transparent and caller allocated.
     */
    typedef struct rvsp_spgemm_descr *rvsp_spgemm_descr_t;

    /*
     * Accumulate strategy for the fp32 v2 kernels.
     *
     * DEFAULT is the scalar accumulator and is the only one available in a
     * build without the V extension. The other three are always present in this
     * enum so the header does not change shape with -march; availability is a
     * run time question, and selecting one the current build cannot provide
     * returns RVSP_ERROR_UNSUPPORTED_BACKEND.
     */
    typedef enum
    {
        RVSP_SPGEMM_ALGO_DEFAULT = 0, /* scalar                              */
        RVSP_SPGEMM_ALGO_RVV,         /* gather / FMA / scatter, needs V     */
        RVSP_SPGEMM_ALGO_CONTIG,      /* unit stride on runs, needs V        */
        RVSP_SPGEMM_ALGO_ADAPTIVE     /* runs + gather + scalar, needs V     */
    } rvsp_spgemm_algo_t;

#ifdef __cplusplus
}
#endif

#endif /*RV_SPARSE_TYPES_H */
