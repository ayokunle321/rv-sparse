/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Every available strategy against hand-computed products.
 *
 * Structure is compared exactly because it comes from the shared symbolic
 * phase and is deterministic. Values are compared with a tolerance because
 * vector accumulation reorders the additions.
 *
 * Strategies the build does not provide are reported as skipped, never as
 * passed.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "csr_fixtures.h"
#include "rv_sparse.h"

static int failures = 0;
static int checks = 0;
static int skipped = 0;

static void check(int cond, const char *what) {
  checks++;
  if (!cond) {
    failures++;
    printf("    FAIL  %s\n", what);
  }
}

static const struct {
  rvsp_spgemm_algo_t algo;
  const char *name;
} STRATEGIES[] = {
    {RVSP_SPGEMM_ALGO_DEFAULT, "scalar"},
    {RVSP_SPGEMM_ALGO_RVV, "rvv"},
};

static void describe(const rvsp_fx_csr_t *m, rvsp_csr_matrix_t *out) {
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

/*
 * Run one product through the descriptor path with the given strategy and
 * compare against the expected result. Returns 1 if the strategy ran, 0 if the
 * build does not provide it.
 */
static int run_case(const rvsp_fx_pair_t *fx, rvsp_spgemm_algo_t algo) {
  rvsp_csr_matrix_t A;
  rvsp_csr_matrix_t B;
  describe(&fx->a, &A);
  describe(&fx->b, &B);

  rvsp_spgemm_descr_t descr = NULL;
  if (rvsp_spgemm_descr_create(&descr) != RVSP_SUCCESS) {
    check(0, "descr_create");
    return 1;
  }

  if (rvsp_spgemm_set_algo(descr, algo) == RVSP_ERROR_UNSUPPORTED_BACKEND) {
    rvsp_spgemm_descr_destroy(descr);
    return 0;
  }

  size_t ws_bytes = 0;
  int32_t c_nnz = 0;

  if (rvsp_spgemm_work_estimation(descr, &A, &B, &ws_bytes, &c_nnz) !=
      RVSP_SUCCESS) {
    check(0, "work_estimation");
    rvsp_spgemm_descr_destroy(descr);
    return 1;
  }

  check(c_nnz == fx->c.nnz, "c_nnz matches expected");

  /* Caller owns C and the workspace on this path. */
  const size_t alloc = c_nnz > 0 ? (size_t)c_nnz : 1;
  int32_t *c_row_ptr = malloc(((size_t)fx->a.rows + 1) * sizeof(int32_t));
  int32_t *c_col_idx = malloc(alloc * sizeof(int32_t));
  float *c_values = malloc(alloc * sizeof(float));
  void *ws = malloc(ws_bytes);

  if (!c_row_ptr || !c_col_idx || !c_values || !ws) {
    check(0, "allocation");
    free(c_row_ptr);
    free(c_col_idx);
    free(c_values);
    free(ws);
    rvsp_spgemm_descr_destroy(descr);
    return 1;
  }

  rvsp_csr_matrix_t C;
  C.rows = fx->a.rows;
  C.cols = fx->b.cols;
  C.nnz = c_nnz;
  C.row_ptr = c_row_ptr;
  C.col_idx = c_col_idx;
  C.values = c_values;
  C.dtype = RVSP_DTYPE_FP32;
  C.format = RVSP_FORMAT_CSR;
  C.owns_data = 0;

  check(rvsp_spgemm_compute(descr, &A, &B, &C, ws) == RVSP_SUCCESS, "compute");

  /* Structure is integer and deterministic, so compare it exactly. */
  int structure_ok = (C.nnz == fx->c.nnz);
  for (int32_t i = 0; structure_ok && i <= fx->c.rows; i++) {
    structure_ok = (C.row_ptr[i] == fx->c.row_ptr[i]);
  }
  for (int32_t i = 0; structure_ok && i < fx->c.nnz; i++) {
    structure_ok = (C.col_idx[i] == fx->c.col_idx[i]);
  }
  check(structure_ok, "structure matches expected exactly");

  int values_ok = 1;
  for (int32_t i = 0; values_ok && i < fx->c.nnz; i++) {
    const float got = ((const float *)C.values)[i];
    const float want = fx->c.values[i];
    values_ok = fabsf(got - want) <= 1e-5f + 1e-4f * fabsf(want);
  }
  check(values_ok, "values match expected within tolerance");

  free(c_row_ptr);
  free(c_col_idx);
  free(c_values);
  free(ws);
  rvsp_spgemm_descr_destroy(descr);
  return 1;
}

int main(void) {
  const rvsp_fx_pair_t *fx = rvsp_fx_all();
  const int n_fx = rvsp_fx_count();
  const int n_st = (int)(sizeof(STRATEGIES) / sizeof(STRATEGIES[0]));

  for (int s = 0; s < n_st; s++) {
    /* Probe availability before announcing any fixture, so a skipped
     * strategy never looks like it ran one. */
    rvsp_spgemm_descr_t probe = NULL;
    int available = 0;

    if (rvsp_spgemm_descr_create(&probe) == RVSP_SUCCESS) {
      available = rvsp_spgemm_set_algo(probe, STRATEGIES[s].algo) !=
                  RVSP_ERROR_UNSUPPORTED_BACKEND;
      rvsp_spgemm_descr_destroy(probe);
    }

    if (!available) {
      skipped++;
      printf("  %-9s skipped, not provided by this build\n",
             STRATEGIES[s].name);
      continue;
    }

    for (int f = 0; f < n_fx; f++) {
      printf("  %-9s %-12s (%s)\n", STRATEGIES[s].name, fx[f].name,
             fx[f].targets);
      run_case(&fx[f], STRATEGIES[s].algo);
    }
  }

  printf("\ntest_spgemm_reference_f32: %s  (%d checks, %d failed, "
         "%d strategies skipped)\n",
         failures ? "FAIL" : "PASS", checks, failures, skipped);

  return failures ? 1 : 0;
}
