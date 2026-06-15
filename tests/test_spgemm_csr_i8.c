/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Correctness test for INT8 CSR SpGEMM with INT32 output.
 */

#include <assert.h>
#include <stdio.h>

#include "rv_sparse.h"

int main(void)
{
    /*
     * A =
     * [1  2]
     * [0 -3]
     */
    int32_t a_row_ptr[] = {0, 2, 3};
    int32_t a_col_idx[] = {0, 1, 1};
    int8_t a_values[] = {1, 2, -3};

    /*
     * B =
     * [4  0]
     * [0  5]
     */
    int32_t b_row_ptr[] = {0, 1, 2};
    int32_t b_col_idx[] = {0, 1};
    int8_t b_values[] = {4, 5};

    rvsp_csr_matrix_t A;
    rvsp_csr_matrix_t B;
    rvsp_csr_matrix_t C = {0};

    rvsp_status_t status;

    status = rvsp_csr_create(
        &A,
        2,
        2,
        3,
        a_row_ptr,
        a_col_idx,
        a_values,
        RVSP_DTYPE_INT8);
    assert(status == RVSP_SUCCESS);

    status = rvsp_csr_create(
        &B,
        2,
        2,
        2,
        b_row_ptr,
        b_col_idx,
        b_values,
        RVSP_DTYPE_INT8);
    assert(status == RVSP_SUCCESS);

    rvsp_spgemm_options_t options = {
        .backend = RVSP_BACKEND_SCALAR,
        .input_dtype = RVSP_DTYPE_INT8,
        .output_dtype = RVSP_DTYPE_INT32,
        .sort_output_indices = 1};

    status = rvsp_spgemm_csr(&A, &B, &C, &options);
    assert(status == RVSP_SUCCESS);

    assert(C.rows == 2);
    assert(C.cols == 2);
    assert(C.nnz == 3);
    assert(C.dtype == RVSP_DTYPE_INT32);

    assert(C.row_ptr[0] == 0);
    assert(C.row_ptr[1] == 2);
    assert(C.row_ptr[2] == 3);

    assert(C.col_idx[0] == 0);
    assert(C.col_idx[1] == 1);
    assert(C.col_idx[2] == 1);

    int32_t *c_values = (int32_t *)C.values;

    /*
     * C =
     * [4  10]
     * [0 -15]
     */
    assert(c_values[0] == 4);
    assert(c_values[1] == 10);
    assert(c_values[2] == -15);

    rvsp_csr_destroy(&A);
    rvsp_csr_destroy(&B);
    rvsp_csr_destroy(&C);

    printf("test_spgemm_csr_i8: PASS\n");

    return 0;
}