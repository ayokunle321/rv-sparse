/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * CSR reordering. Produces a symmetric permutation that clusters nonzeros
 * toward the diagonal, so the accumulator accesses of a later SpGEMM fall in
 * a narrower column range. Preprocessing only, run once outside any timed
 * region.
 */

#ifndef RVSP_REORDER_H
#define RVSP_REORDER_H

#include <stdint.h>

#include "rv_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Computes a Reverse Cuthill-McKee permutation of a square canonical CSR
 * matrix. perm is filled with the new-to-old ordering: perm[i] is the original
 * index placed at position i. iperm, if non-NULL, receives the inverse
 * (old-to-new), which is what you need to un-permute results back to the
 * original order.
 *
 * perm and iperm must each hold n entries. Returns RVSP_SUCCESS or an error.
 */
rvsp_status_t rvsp_csr_rcm_order(
    int32_t n,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    int32_t *perm,
    int32_t *iperm);

/*
 * Applies a symmetric permutation to a square CSR matrix, producing a new
 * canonical CSR (columns sorted within each row). perm is new-to-old as
 * returned by rvsp_csr_rcm_order. The output arrays are allocated by the
 * callee and owned by the caller.
 */
rvsp_status_t rvsp_csr_permute(
    int32_t n,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const float *values,
    const int32_t *perm,
    const int32_t *iperm,
    int32_t **row_ptr_out,
    int32_t **col_idx_out,
    float **values_out);

#ifdef __cplusplus
}
#endif

#endif /* RVSP_REORDER_H */