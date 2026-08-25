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
        RVSP_DTYPE_FP32
    } rvsp_dtype_t;

    typedef enum
    {
        RVSP_FORMAT_CSR
    } rvsp_format_t;

    typedef enum
    {
        RVSP_BACKEND_SCALAR,
        RVSP_BACKEND_SCALAR_UNROLL4,
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
    } rvsp_spgemm_options_t;

    /*
    * Opaque SpGEMM operation descriptor.
    *
    * Stores the selected algorithm and analyzed output structure between
    * the structure and numeric phases.
    */
    typedef struct rvsp_spgemm_descr *rvsp_spgemm_descr_t;

    /*
    * SpGEMM accumulation strategy.
    *
    * RVSP_SPGEMM_ALGO_DEFAULT uses the scalar implementation.
    * The RVV strategies require a build with the RISC-V Vector extension.
    */
    typedef enum
    {
        RVSP_SPGEMM_ALGO_DEFAULT = 0, /* scalar */
        RVSP_SPGEMM_ALGO_RVV_M1,      /* gather and scatter, LMUL=1 */
        RVSP_SPGEMM_ALGO_RVV_M2,      /* gather and scatter, LMUL=2 */
        RVSP_SPGEMM_ALGO_RVV_M4,      /* gather and scatter, LMUL=4 */
        RVSP_SPGEMM_ALGO_OMP,         /* scalar, row loop over OpenMP threads */

        /* Alias, kept last so it cannot renumber what follows. */
        RVSP_SPGEMM_ALGO_RVV = RVSP_SPGEMM_ALGO_RVV_M2
    } rvsp_spgemm_algo_t;

#ifdef __cplusplus
}
#endif

#endif /*RV_SPARSE_TYPES_H */
