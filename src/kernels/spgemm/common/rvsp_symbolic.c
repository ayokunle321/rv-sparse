/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Symbolic phases shared by all kernels. These determine the structure of C
 * before the numeric pass computes values.
 */

#include "rvsp_common.h"

/* Counts the nonzeros per output row and prefix-sums them into c_row_ptr. */
rvsp_status_t
rvsp_symbolic_count(int32_t a_rows, const int32_t *RVSP_RESTRICT a_row_ptr,
                    const int32_t *RVSP_RESTRICT a_col_idx,
                    const int32_t *RVSP_RESTRICT b_row_ptr,
                    const int32_t *RVSP_RESTRICT b_col_idx,
                    uint8_t *RVSP_RESTRICT mark, int32_t *RVSP_RESTRICT touched,
                    int32_t *RVSP_RESTRICT c_row_ptr, int64_t *op_counts_out,
                    int64_t *total_nnz_out) {
  int64_t total_nnz = 0;

  for (int32_t row = 0; row < a_rows; row++) {
    const int32_t a_start = a_row_ptr[row];
    const int32_t a_end = a_row_ptr[row + 1];

    int32_t touched_count = 0;
    int64_t ops = 0;

    for (int32_t ap = a_start; ap < a_end; ap++) {
      const int32_t k = a_col_idx[ap];
      const int32_t b_start = b_row_ptr[k];
      const int32_t b_end = b_row_ptr[k + 1];

      ops += (int64_t)(b_end - b_start);

      for (int32_t bp = b_start; bp < b_end; bp++) {
        const int32_t col = b_col_idx[bp];

        if (mark[col] == 0) {
          mark[col] = 1;
          touched[touched_count++] = col;
        }
      }
    }

    /* Clear only the columns this row touched, so the sweep stays O(nnz). */
    for (int32_t i = 0; i < touched_count; i++) {
      mark[touched[i]] = 0;
    }

    if (op_counts_out != NULL) {
      op_counts_out[row] = ops;
    }

    c_row_ptr[row] = touched_count;
    total_nnz += touched_count;

    if (total_nnz > INT32_MAX) {
      return RVSP_ERROR_ALLOCATION_FAILED;
    }
  }

  int32_t running = 0;

  for (int32_t row = 0; row < a_rows; row++) {
    const int32_t count = c_row_ptr[row];
    c_row_ptr[row] = running;
    running += count;
  }

  c_row_ptr[a_rows] = running;
  *total_nnz_out = total_nnz;

  return RVSP_SUCCESS;
}

/* Writes each row's column indices into C and sorts them ascending. */
void rvsp_symbolic_fill(int32_t a_rows, int32_t b_cols,
                        const int32_t *RVSP_RESTRICT a_row_ptr,
                        const int32_t *RVSP_RESTRICT a_col_idx,
                        const int32_t *RVSP_RESTRICT b_row_ptr,
                        const int32_t *RVSP_RESTRICT b_col_idx,
                        uint8_t *RVSP_RESTRICT mark,
                        int32_t *RVSP_RESTRICT scratch,
                        const int32_t *RVSP_RESTRICT c_row_ptr,
                        int32_t *RVSP_RESTRICT c_col_idx) {
  const int32_t max_col = b_cols > 0 ? b_cols - 1 : 0;

  for (int32_t row = 0; row < a_rows; row++) {
    const int32_t a_start = a_row_ptr[row];
    const int32_t a_end = a_row_ptr[row + 1];

    int32_t *const out = &c_col_idx[c_row_ptr[row]];
    int32_t count = 0;

    for (int32_t ap = a_start; ap < a_end; ap++) {
      const int32_t k = a_col_idx[ap];
      const int32_t b_start = b_row_ptr[k];
      const int32_t b_end = b_row_ptr[k + 1];

      for (int32_t bp = b_start; bp < b_end; bp++) {
        const int32_t col = b_col_idx[bp];

        if (mark[col] == 0) {
          mark[col] = 1;
          out[count++] = col;
        }
      }
    }

    rvsp_sort_columns(out, scratch, count, max_col);

    for (int32_t i = 0; i < count; i++) {
      mark[out[i]] = 0;
    }
  }
}
