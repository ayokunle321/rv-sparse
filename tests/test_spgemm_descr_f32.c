/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Descriptor lifecycle, argument checking, and structure reuse.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "rv_sparse.h"
#include "csr_fixtures.h"

static int failures = 0;
static int checks = 0;

static void check(int cond, const char *what)
{
    checks++;
    printf("  [%s] %s\n", cond ? "ok  " : "FAIL", what);
    if (!cond)
    {
        failures++;
    }
}

static void describe(const rvsp_fx_csr_t *m, rvsp_csr_matrix_t *out)
{
    out->rows = m->rows;
    out->cols = m->cols;
    out->nnz = m->nnz;
    out->row_ptr = (int32_t *)m->row_ptr;
    out->col_idx = (int32_t *)m->col_idx;
    out->values = (void *)m->values;
    out->dtype = RVSP_DTYPE_FP32;
    out->format = RVSP_FORMAT_CSR;
    out->owns_data = 0;
}

int main(void)
{
    /* The basic fixture is enough; this file is about contracts, not numbers. */
    const rvsp_fx_pair_t *fx = &rvsp_fx_all()[0];

    rvsp_csr_matrix_t A;
    rvsp_csr_matrix_t B;
    describe(&fx->a, &A);
    describe(&fx->b, &B);

    /* Values are mutated later to prove reuse, so work on a copy. */
    float a_values[3];
    for (int i = 0; i < fx->a.nnz; i++)
    {
        a_values[i] = fx->a.values[i];
    }
    A.values = a_values;

    printf("lifecycle\n");

    rvsp_spgemm_descr_t descr = NULL;
    check(rvsp_spgemm_descr_create(&descr) == RVSP_SUCCESS && descr != NULL,
          "descr_create succeeds");
    check(rvsp_spgemm_descr_create(NULL) == RVSP_ERROR_NULL_POINTER,
          "descr_create rejects NULL out pointer");

    size_t ws_bytes = 0;
    int32_t c_nnz = 0;

    check(rvsp_spgemm_set_algo(descr, RVSP_SPGEMM_ALGO_DEFAULT)
              == RVSP_SUCCESS,
          "set_algo before work_estimation succeeds");
    check(rvsp_spgemm_set_algo(NULL, RVSP_SPGEMM_ALGO_DEFAULT)
              == RVSP_ERROR_NULL_POINTER,
          "set_algo rejects NULL descriptor");

    printf("\nbad arguments before analysis\n");

    {
        /* compute needs a C, so build a throwaway one to isolate the ordering
         * rule from the null checks. */
        int32_t rp[3];
        int32_t ci[3];
        float v[3];
        rvsp_csr_matrix_t C = {2, 2, 3, rp, ci, v,
                               RVSP_DTYPE_FP32, RVSP_FORMAT_CSR, 0};
        void *ws = malloc(4096);

        check(rvsp_spgemm_compute(descr, &A, &B, &C, ws)
                  == RVSP_ERROR_INVALID_ARGUMENT,
              "compute before work_estimation is rejected");

        int64_t ops[2];
        check(rvsp_spgemm_get_op_counts(descr, ops, 2)
                  == RVSP_ERROR_INVALID_ARGUMENT,
              "get_op_counts before work_estimation is rejected");

        free(ws);
    }

    printf("\nanalysis\n");

    check(rvsp_spgemm_work_estimation(descr, &A, &B, &ws_bytes, &c_nnz)
              == RVSP_SUCCESS,
          "work_estimation succeeds");
    check(c_nnz == fx->c.nnz, "work_estimation reports the exact nnz");
    check(ws_bytes > 0, "work_estimation reports a workspace size");
    check(rvsp_spgemm_work_estimation(descr, NULL, &B, &ws_bytes, &c_nnz)
              == RVSP_ERROR_NULL_POINTER,
          "work_estimation rejects NULL A");

    check(rvsp_spgemm_set_algo(descr, RVSP_SPGEMM_ALGO_DEFAULT)
              == RVSP_ERROR_INVALID_ARGUMENT,
          "set_algo after work_estimation is rejected");

    int64_t ops[2] = {0, 0};
    check(rvsp_spgemm_get_op_counts(descr, ops, 2) == RVSP_SUCCESS,
          "get_op_counts succeeds after analysis");
    check(ops[0] == 2 && ops[1] == 1,
          "op counts match the fixture");
    check(rvsp_spgemm_get_op_counts(descr, ops, 1)
              == RVSP_ERROR_INVALID_ARGUMENT,
          "get_op_counts rejects an undersized buffer");

    /* Caller-owned output and workspace. */
    int32_t *c_row_ptr = malloc(((size_t)A.rows + 1) * sizeof(int32_t));
    int32_t *c_col_idx = malloc((size_t)c_nnz * sizeof(int32_t));
    float *c_values = malloc((size_t)c_nnz * sizeof(float));
    void *ws = malloc(ws_bytes);

    rvsp_csr_matrix_t C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.nnz = c_nnz;
    C.row_ptr = c_row_ptr;
    C.col_idx = c_col_idx;
    C.values = c_values;
    C.dtype = RVSP_DTYPE_FP32;
    C.format = RVSP_FORMAT_CSR;
    C.owns_data = 0;

    printf("\nbad arguments at compute\n");

    check(rvsp_spgemm_compute(descr, &A, &B, &C, NULL)
              == RVSP_ERROR_NULL_POINTER,
          "compute rejects NULL workspace");
    check(rvsp_spgemm_compute(descr, &A, &B, NULL, ws)
              == RVSP_ERROR_NULL_POINTER,
          "compute rejects NULL C");

    {
        rvsp_csr_matrix_t small = C;
        small.nnz = c_nnz - 1;
        check(rvsp_spgemm_compute(descr, &A, &B, &small, ws)
                  == RVSP_ERROR_INVALID_ARGUMENT,
              "compute rejects an undersized C");
    }
    {
        rvsp_csr_matrix_t wrong = C;
        wrong.rows = C.rows + 1;
        check(rvsp_spgemm_compute(descr, &A, &B, &wrong, ws)
                  == RVSP_ERROR_INVALID_ARGUMENT,
              "compute rejects C with the wrong shape");
    }
    {
        rvsp_csr_matrix_t bad_a = A;
        bad_a.nnz = A.nnz - 1;
        check(rvsp_spgemm_compute(descr, &bad_a, &B, &C, ws)
                  == RVSP_ERROR_INVALID_ARGUMENT,
              "compute rejects an A that does not match the analysis");
    }

    printf("\ncompute and reuse\n");

    check(rvsp_spgemm_compute(descr, &A, &B, &C, ws) == RVSP_SUCCESS,
          "compute succeeds");
    check(C.nnz == fx->c.nnz, "compute sets C.nnz");

    int first_ok = 1;
    for (int32_t i = 0; i < fx->c.nnz; i++)
    {
        first_ok = first_ok && (c_col_idx[i] == fx->c.col_idx[i]) &&
                   fabsf(c_values[i] - fx->c.values[i]) < 1e-5f;
    }
    check(first_ok, "compute writes C's structure and values");

    /*
     * The reuse contract. Change only the values in A, keep the pattern, and
     * call compute again with the same descriptor and the same C. No
     * re-analysis, no invalidate.
     */
    for (int i = 0; i < fx->a.nnz; i++)
    {
        a_values[i] *= 2.0f;
    }

    check(rvsp_spgemm_compute(descr, &A, &B, &C, ws) == RVSP_SUCCESS,
          "second compute succeeds with no re-analysis");

    int doubled = 1;
    for (int32_t i = 0; i < fx->c.nnz; i++)
    {
        const float want = 2.0f * fx->c.values[i];
        doubled = doubled && fabsf(c_values[i] - want) < 1e-4f;
    }
    check(doubled, "values doubled, structure reused");

    int structure_held = 1;
    for (int32_t i = 0; i < fx->c.nnz; i++)
    {
        structure_held = structure_held && (c_col_idx[i] == fx->c.col_idx[i]);
    }
    check(structure_held, "column indices untouched by the second compute");

    printf("\nteardown\n");

    rvsp_spgemm_invalidate_structure(descr);
    check(rvsp_spgemm_compute(descr, &A, &B, &C, ws) == RVSP_SUCCESS,
          "compute succeeds after invalidate_structure");

    rvsp_spgemm_invalidate_structure(NULL);
    rvsp_spgemm_descr_destroy(NULL);
    check(1, "invalidate_structure and descr_destroy accept NULL");

    rvsp_spgemm_descr_destroy(descr);

    free(c_row_ptr);
    free(c_col_idx);
    free(c_values);
    free(ws);

    printf("\ntest_spgemm_descr_f32: %s  (%d checks, %d failed)\n",
           failures ? "FAIL" : "PASS", checks, failures);

    return failures ? 1 : 0;
}
