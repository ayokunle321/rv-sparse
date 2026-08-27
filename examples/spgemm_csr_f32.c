/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * One-shot SpGEMM: C = A * B in a single call.
 *
 * rvsp_spgemm_csr allocates C. Release it with rvsp_csr_destroy.
 * Input matrices are canonical CSR and owned by the caller.
 */

#include "rv_sparse.h"
#include <stdio.h>

static void print_csr(const rvsp_csr_matrix_t *A) {
  const float *values = (const float *)A->values;

  printf("rows = %d, cols = %d, nnz = %d\n", A->rows, A->cols, A->nnz);

  for (int32_t i = 0; i < A->rows; i++) {
    printf("row %d:", i);

    for (int32_t p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
      printf(" (%d, %.2f)", A->col_idx[p], values[p]);
    }

    printf("\n");
  }
}

int main(void) {
  /* A = [1 2; 0 3] */
  int32_t a_row_ptr[] = {0, 2, 3};
  int32_t a_col_idx[] = {0, 1, 1};
  float a_values[] = {1.0f, 2.0f, 3.0f};

  /* B = [4 0; 0 5] */
  int32_t b_row_ptr[] = {0, 1, 2};
  int32_t b_col_idx[] = {0, 1};
  float b_values[] = {4.0f, 5.0f};

  rvsp_csr_matrix_t A;
  rvsp_csr_matrix_t B;
  rvsp_csr_matrix_t C = {0};
  rvsp_status_t status;

  status = rvsp_csr_create(&A, 2, 2, 3, a_row_ptr, a_col_idx, a_values,
                           RVSP_DTYPE_FP32);
  if (status != RVSP_SUCCESS) {
    printf("create A: %s\n", rvsp_status_to_string(status));
    return 1;
  }

  status = rvsp_csr_create(&B, 2, 2, 2, b_row_ptr, b_col_idx, b_values,
                           RVSP_DTYPE_FP32);
  if (status != RVSP_SUCCESS) {
    printf("create B: %s\n", rvsp_status_to_string(status));
    return 1;
  }

  rvsp_spgemm_options_t options = {.backend = RVSP_BACKEND_SCALAR,
                                   .input_dtype = RVSP_DTYPE_FP32,
                                   .output_dtype = RVSP_DTYPE_FP32};

  status = rvsp_spgemm_csr(&A, &B, &C, &options);
  if (status != RVSP_SUCCESS) {
    printf("spgemm: %s\n", rvsp_status_to_string(status));
    return 1;
  }

  print_csr(&C);

  rvsp_csr_destroy(&A);
  rvsp_csr_destroy(&B);
  rvsp_csr_destroy(&C);

  return 0;
}
