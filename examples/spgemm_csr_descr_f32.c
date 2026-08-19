/*
 * Descriptor SpGEMM: structure analysis then compute.
 *
 * The caller allocates C and the workspace from the sizes reported by
 * rvsp_spgemm_work_estimation, then calls rvsp_spgemm_compute.
 * Input matrices are canonical CSR and owned by the caller.
 */

#include <stdio.h>
#include <stdlib.h>
#include "rv_sparse.h"

static void print_csr(const rvsp_csr_matrix_t *A)
{
    const float *values = (const float *)A->values;

    printf("rows = %d, cols = %d, nnz = %d\n", A->rows, A->cols, A->nnz);

    for (int32_t i = 0; i < A->rows; i++)
    {
        printf("row %d:", i);

        for (int32_t p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++)
        {
            printf(" (%d, %.2f)", A->col_idx[p], values[p]);
        }

        printf("\n");
    }
}

int main(void)
{
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
    rvsp_status_t status;

    rvsp_csr_create(&A, 2, 2, 3,
                    a_row_ptr, a_col_idx, a_values, RVSP_DTYPE_FP32);
    rvsp_csr_create(&B, 2, 2, 2,
                    b_row_ptr, b_col_idx, b_values, RVSP_DTYPE_FP32);

    rvsp_spgemm_descr_t d;
    status = rvsp_spgemm_descr_create(&d);
    if (status != RVSP_SUCCESS)
    {
        printf("descr create: %s\n", rvsp_status_to_string(status));
        return 1;
    }

    status = rvsp_spgemm_set_algo(d, RVSP_SPGEMM_ALGO_DEFAULT);
    if (status != RVSP_SUCCESS)
    {
        printf("set algo: %s\n", rvsp_status_to_string(status));
        return 1;
    }

    size_t workspace_bytes = 0;
    int32_t c_nnz = 0;

    status = rvsp_spgemm_work_estimation(d, &A, &B, &workspace_bytes, &c_nnz);
    if (status != RVSP_SUCCESS)
    {
        printf("work estimation: %s\n", rvsp_status_to_string(status));
        return 1;
    }

    /* Caller allocates C's arrays from the reported sizes. C owns nothing the
     * library allocated, so these are freed by hand below, not by destroy. */
    rvsp_csr_matrix_t C = {0};
    C.rows = A.rows;
    C.cols = B.cols;
    C.nnz = c_nnz;
    C.dtype = RVSP_DTYPE_FP32;
    C.row_ptr = malloc((size_t)(A.rows + 1) * sizeof(int32_t));
    C.col_idx = malloc((size_t)c_nnz * sizeof(int32_t));
    C.values = malloc((size_t)c_nnz * sizeof(float));

    void *workspace = malloc(workspace_bytes);

    if (C.row_ptr == NULL || C.col_idx == NULL ||
        C.values == NULL || workspace == NULL)
    {
        printf("allocation failed\n");
        return 1;
    }

    status = rvsp_spgemm_compute(d, &A, &B, &C, workspace);
    if (status != RVSP_SUCCESS)
    {
        printf("compute: %s\n", rvsp_status_to_string(status));
        return 1;
    }

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