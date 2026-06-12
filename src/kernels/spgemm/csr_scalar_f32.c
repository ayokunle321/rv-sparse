#include <stdlib.h>
#include "rv_sparse.h"

rvsp_status_t rvsp_spgemm_csr_scalar_f32(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C
) {
    rvsp_status_t status;

    if (!A || !B || !C) {
        return RVSP_ERROR_NULL_POINTER;
    }

    status = rvsp_csr_validate(A);
    if (status != RVSP_SUCCESS) {
        return status;
    }

    status = rvsp_csr_validate(B);
    if (status != RVSP_SUCCESS) {
        return status;
    }

    if (A->dtype != RVSP_DTYPE_FP32 || B->dtype != RVSP_DTYPE_FP32) {
        return RVSP_ERROR_UNSUPPORTED_DTYPE;
    }

    if (A->cols != B->rows) {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t max_nnz = A->rows * B->cols;

    int32_t *c_row_ptr = (int32_t *)calloc((size_t)A->rows + 1, sizeof(int32_t));
    int32_t *c_col_idx = (int32_t *)malloc((size_t)max_nnz * sizeof(int32_t));
    float   *c_values  = (float *)malloc((size_t)max_nnz * sizeof(float));

    if (!c_row_ptr || !c_col_idx || !c_values) {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    const float *a_values = (const float *)A->values;
    const float *b_values = (const float *)B->values;

    int32_t row_count = 0;

    for (int32_t row = 0; row < A->rows; row++) {
        c_row_ptr[row] = row_count;

        float *acc = (float *)calloc((size_t)B->cols, sizeof(float));
        if (!acc) {
            free(c_row_ptr);
            free(c_col_idx);
            free(c_values);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }

        for (int32_t a_pos = A->row_ptr[row]; a_pos < A->row_ptr[row + 1]; a_pos++) {
            int32_t a_col = A->col_idx[a_pos];
            float val_a = a_values[a_pos];

            for (int32_t b_pos = B->row_ptr[a_col]; b_pos < B->row_ptr[a_col + 1]; b_pos++) {
                int32_t b_col = B->col_idx[b_pos];
                float val_b = b_values[b_pos];

                acc[b_col] += val_a * val_b;
            }
        }

        for (int32_t col = 0; col < B->cols; col++) {
            if (acc[col] != 0.0f) {
                c_col_idx[row_count] = col;
                c_values[row_count] = acc[col];
                row_count++;
            }
        }

        free(acc);
    }

    c_row_ptr[A->rows] = row_count;

    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = row_count;
    C->row_ptr = c_row_ptr;
    C->col_idx = c_col_idx;
    C->values = c_values;
    C->dtype = RVSP_DTYPE_FP32;
    C->format = RVSP_FORMAT_CSR;
    C->owns_data = 1;

    return RVSP_SUCCESS;
}
