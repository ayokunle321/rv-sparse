/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * SpGEMM kernels, fp32 CSR.
 *
 * Every kernel shares one Gustavson driver and differs only in the accumulate
 * loop, which each strategy inlines from gustavson_core_f32.inc. Keep it that
 * way or a timing difference between two kernels stops meaning anything.
 *
 * Inputs must be canonical CSR. Nothing here checks that and the hot loops
 * carry no bounds checks, so call rvsp_csr_check() once after loading.
 *
 * Nonzeros that cancel to exactly zero are kept, matching cuSPARSE and
 * rocSPARSE. Callers own the scratch buffer.
 */

#ifndef RVSP_H
#define RVSP_H

#include <stddef.h>
#include <stdint.h>

#include "rv_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Workspace */

/* Depends only on b_cols, so one buffer serves repeated products of one shape. */
rvsp_status_t rvsp_spgemm_buffer_size(int32_t b_cols, size_t *bytes_out);

/* Validation. Call once, outside any timed region. */

typedef enum
{
    RVSP_CSR_OK = 0,
    RVSP_CSR_BAD_ROW_PTR,       /* row_ptr[0] != 0 or not non-decreasing */
    RVSP_CSR_COL_OUT_OF_RANGE,  /* column index outside [0, n_cols)      */
    RVSP_CSR_UNSORTED,          /* columns not ascending within a row    */
    RVSP_CSR_DUPLICATE          /* repeated column within a row          */
} rvsp_csr_status_t;

/* O(nnz), no allocation. bad_row_out is optional and gets -1 if not row specific. */
rvsp_csr_status_t rvsp_csr_check(int32_t rows, int32_t cols,
                                       const int32_t *row_ptr,
                                       const int32_t *col_idx,
                                       int32_t *bad_row_out);

const char *rvsp_csr_status_string(rvsp_csr_status_t status);

/* Kernels */

/*
 * C = A * B. Output arrays are malloc'd by the callee, owned by the caller.
 * op_counts_out is optional, a_rows entries, and receives Op_i per output row.
 *
 *   scalar    no vector instructions
 *   rvv       gather, FMA, scatter on every segment
 */
#define RVSP_SPGEMM_PARAMS                                              \
    int32_t a_rows, int32_t a_cols, int32_t b_cols,                        \
    const int32_t *a_row_ptr, const int32_t *a_col_idx,                    \
    const float *a_values,                                                 \
    const int32_t *b_row_ptr, const int32_t *b_col_idx,                    \
    const float *b_values,                                                 \
    void *workspace, size_t workspace_bytes,                               \
    int32_t **c_row_ptr_out, int32_t **c_col_idx_out,                      \
    float **c_values_out, int32_t *c_nnz_out,                              \
    int64_t *op_counts_out

rvsp_status_t rvsp_spgemm_scalar_f32(RVSP_SPGEMM_PARAMS);
rvsp_status_t rvsp_spgemm_rvv_f32(RVSP_SPGEMM_PARAMS);

/* Phases, exposed separately for the descriptor API. */

/*
 * Driving the phases directly means zeroing mark before each symbolic call and
 * acc before numeric. Each leaves its array zeroed, so a buffer stays reusable.
 */

rvsp_status_t rvsp_symbolic_count(
    int32_t a_rows,
    const int32_t *a_row_ptr, const int32_t *a_col_idx,
    const int32_t *b_row_ptr, const int32_t *b_col_idx,
    uint8_t *mark, int32_t *touched,
    int32_t *c_row_ptr,          /* out, a_rows + 1, prefix summed */
    int64_t *op_counts_out,      /* optional, a_rows entries        */
    int64_t *total_nnz_out);

void rvsp_symbolic_fill(
    int32_t a_rows, int32_t b_cols,
    const int32_t *a_row_ptr, const int32_t *a_col_idx,
    const int32_t *b_row_ptr, const int32_t *b_col_idx,
    uint8_t *mark, int32_t *scratch,
    const int32_t *c_row_ptr,
    int32_t *c_col_idx);         /* out, sorted within each row */

#define RVSP_NUMERIC_PARAMS                                             \
    int32_t a_rows,                                                        \
    const int32_t *a_row_ptr, const int32_t *a_col_idx,                    \
    const float *a_values,                                                 \
    const int32_t *b_row_ptr, const int32_t *b_col_idx,                    \
    const float *b_values,                                                 \
    float *acc,                                                            \
    const int32_t *c_row_ptr, const int32_t *c_col_idx,                    \
    float *c_values

void rvsp_spgemm_scalar_f32_numeric(RVSP_NUMERIC_PARAMS);
void rvsp_spgemm_rvv_f32_numeric(RVSP_NUMERIC_PARAMS);

#ifdef __cplusplus
}
#endif

#endif /* RVSP_H */