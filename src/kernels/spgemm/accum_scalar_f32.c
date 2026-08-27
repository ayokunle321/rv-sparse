/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Scalar accumulation.
 */

#include "rvsp_common.h"

static inline void rvsp_accum_row(float a_val, int32_t b_nnz,
                                  const int32_t *RVSP_RESTRICT b_col_idx,
                                  const float *RVSP_RESTRICT b_values,
                                  float *RVSP_RESTRICT acc) {
  for (int32_t p = 0; p < b_nnz; p++) {
    acc[b_col_idx[p]] += a_val * b_values[p];
  }
}

#define RVSP_KERNEL_NAME rvsp_spgemm_scalar_f32
#include "gustavson_core_f32.inc"
