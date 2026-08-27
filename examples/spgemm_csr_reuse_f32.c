/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Structure reuse: analyze once, compute twice.
 *
 * When A and B keep the same sparsity pattern and only their values change,
 * the analyzed structure is reused across compute calls, so work_estimation
 * runs only once.
 */

#include "rv_sparse.h"
#include <stdio.h>
#include <stdlib.h>

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
  /* A = [1 2; 0 3], same pattern reused with new values below */
  int32_t a_row_ptr[] = {0, 2, 3};
  int32_t a_col_idx[] = {0, 1, 1};
  float a_values[] = {1.0f, 2.0f, 3.0f};

  /* B = [4 0; 0 5] */
  int32_t b_row_ptr[] = {0, 1, 2};
  int32_t b_col_idx[] = {0, 1};
  float b_values[] = {4.0f, 5.0f};

  rvsp_csr_matrix_t A;
  rvsp_csr_matrix_t B;

  rvsp_csr_create(&A, 2, 2, 3, a_row_ptr, a_col_idx, a_values, RVSP_DTYPE_FP32);
  rvsp_csr_create(&B, 2, 2, 2, b_row_ptr, b_col_idx, b_values, RVSP_DTYPE_FP32);

  rvsp_spgemm_descr_t d;
  rvsp_spgemm_descr_create(&d);
  rvsp_spgemm_set_algo(d, RVSP_SPGEMM_ALGO_DEFAULT);

  size_t workspace_bytes = 0;
  int32_t c_nnz = 0;
  rvsp_spgemm_work_estimation(d, &A, &B, &workspace_bytes, &c_nnz);

  rvsp_csr_matrix_t C = {0};
  C.rows = A.rows;
  C.cols = B.cols;
  C.nnz = c_nnz;
  C.dtype = RVSP_DTYPE_FP32;
  C.row_ptr = malloc((size_t)(A.rows + 1) * sizeof(int32_t));
  C.col_idx = malloc((size_t)c_nnz * sizeof(int32_t));
  C.values = malloc((size_t)c_nnz * sizeof(float));
  void *workspace = malloc(workspace_bytes);

  /* First compute with the original values. */
  rvsp_spgemm_compute(d, &A, &B, &C, workspace);
  printf("first result:\n");
  print_csr(&C);

  /* Change only the values of A and B. The sparsity pattern is unchanged,
   * so the same descriptor and the same C allocation are reused, with no
   * second work_estimation call. */
  a_values[0] = 10.0f;
  a_values[1] = 20.0f;
  a_values[2] = 30.0f;
  b_values[0] = 40.0f;
  b_values[1] = 50.0f;

  rvsp_spgemm_compute(d, &A, &B, &C, workspace);
  printf("second result after value change:\n");
  print_csr(&C);

  free(C.row_ptr);
  free(C.col_idx);
  free(C.values);
  free(workspace);

  rvsp_spgemm_descr_destroy(d);
  rvsp_csr_destroy(&A);
  rvsp_csr_destroy(&B);

  return 0;
}
