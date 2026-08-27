/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Shared CSR test matrices with hand-computed products.
 *
 * Each one targets a different branch of the accumulate kernels
 * and all inputs are canonical CSR.
 */

#ifndef RVSP_CSR_FIXTURES_H
#define RVSP_CSR_FIXTURES_H

#include <stdint.h>

typedef struct {
  int32_t rows;
  int32_t cols;
  int32_t nnz;
  const int32_t *row_ptr;
  const int32_t *col_idx;
  const float *values;
} rvsp_fx_csr_t;

typedef struct {
  const char *name;
  const char *targets;
  rvsp_fx_csr_t a;
  rvsp_fx_csr_t b;
  rvsp_fx_csr_t c; /* expected A * B */
} rvsp_fx_pair_t;

static inline int rvsp_fx_count(void);

/*
 * Data lives inside the accessor so that a translation unit including this
 * header without using every fixture does not trip -Wunused-const-variable.
 */
static inline const rvsp_fx_pair_t *rvsp_fx_all(void) {
  /* 1. basic product, two entries merging into one output row */
  static const int32_t a1_rp[] = {0, 2, 3};
  static const int32_t a1_ci[] = {0, 1, 1};
  static const float a1_v[] = {1.0f, 2.0f, 3.0f};
  static const int32_t b1_rp[] = {0, 1, 2};
  static const int32_t b1_ci[] = {0, 1};
  static const float b1_v[] = {4.0f, 5.0f};
  static const int32_t c1_rp[] = {0, 2, 3};
  static const int32_t c1_ci[] = {0, 1, 1};
  static const float c1_v[] = {4.0f, 10.0f, 15.0f};

  /* 2. exact cancellation. The entry is structural and must be retained. */
  static const int32_t a2_rp[] = {0, 2};
  static const int32_t a2_ci[] = {0, 1};
  static const float a2_v[] = {1.0f, 1.0f};
  static const int32_t b2_rp[] = {0, 1, 2};
  static const int32_t b2_ci[] = {0, 0};
  static const float b2_v[] = {1.0f, -1.0f};
  static const int32_t c2_rp[] = {0, 1};
  static const int32_t c2_ci[] = {0};
  static const float c2_v[] = {0.0f};

  /* 3. empty output row, and an empty row in B */
  static const int32_t a3_rp[] = {0, 0, 1};
  static const int32_t a3_ci[] = {0};
  static const float a3_v[] = {2.0f};
  static const int32_t b3_rp[] = {0, 2, 2};
  static const int32_t b3_ci[] = {0, 1};
  static const float b3_v[] = {3.0f, 4.0f};
  static const int32_t c3_rp[] = {0, 0, 2};
  static const int32_t c3_ci[] = {0, 1};
  static const float c3_v[] = {6.0f, 8.0f};

  /* 4. one 16 wide contiguous run, past the contig and gather thresholds */
  static const int32_t a4_rp[] = {0, 1};
  static const int32_t a4_ci[] = {0};
  static const float a4_v[] = {2.0f};
  static const int32_t b4_rp[] = {0, 16};
  static const int32_t b4_ci[] = {0, 1, 2,  3,  4,  5,  6,  7,
                                  8, 9, 10, 11, 12, 13, 14, 15};
  static const float b4_v[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                               7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                               13.0f, 14.0f, 15.0f, 16.0f};
  static const int32_t c4_rp[] = {0, 16};
  static const int32_t c4_ci[] = {0, 1, 2,  3,  4,  5,  6,  7,
                                  8, 9, 10, 11, 12, 13, 14, 15};
  static const float c4_v[] = {2.0f,  4.0f,  6.0f,  8.0f,  10.0f, 12.0f,
                               14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f,
                               26.0f, 28.0f, 30.0f, 32.0f};

  /* 5. fully scattered columns, no run long enough to vectorise */
  static const int32_t a5_rp[] = {0, 1};
  static const int32_t a5_ci[] = {0};
  static const float a5_v[] = {3.0f};
  static const int32_t b5_rp[] = {0, 4};
  static const int32_t b5_ci[] = {0, 5, 11, 19};
  static const float b5_v[] = {1.0f, 2.0f, 3.0f, 4.0f};
  static const int32_t c5_rp[] = {0, 4};
  static const int32_t c5_ci[] = {0, 5, 11, 19};
  static const float c5_v[] = {3.0f, 6.0f, 9.0f, 12.0f};

  /* 6. three A entries per row, two of them hitting the same columns */
  static const int32_t a6_rp[] = {0, 3};
  static const int32_t a6_ci[] = {0, 1, 2};
  static const float a6_v[] = {1.0f, 2.0f, 3.0f};
  static const int32_t b6_rp[] = {0, 2, 3, 5};
  static const int32_t b6_ci[] = {0, 2, 1, 0, 2};
  static const float b6_v[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  static const int32_t c6_rp[] = {0, 3};
  static const int32_t c6_ci[] = {0, 1, 2};
  static const float c6_v[] = {4.0f, 2.0f, 4.0f};

  static const rvsp_fx_pair_t pairs[] = {
      {"basic",
       "merge into one output row",
       {2, 2, 3, a1_rp, a1_ci, a1_v},
       {2, 2, 2, b1_rp, b1_ci, b1_v},
       {2, 2, 3, c1_rp, c1_ci, c1_v}},

      {"cancel",
       "structural zero retained",
       {1, 2, 2, a2_rp, a2_ci, a2_v},
       {2, 1, 2, b2_rp, b2_ci, b2_v},
       {1, 1, 1, c2_rp, c2_ci, c2_v}},

      {"empty-rows",
       "empty output row and empty B row",
       {2, 2, 1, a3_rp, a3_ci, a3_v},
       {2, 2, 2, b3_rp, b3_ci, b3_v},
       {2, 2, 2, c3_rp, c3_ci, c3_v}},

      {"contig-16",
       "contiguous run, vector path",
       {1, 1, 1, a4_rp, a4_ci, a4_v},
       {1, 16, 16, b4_rp, b4_ci, b4_v},
       {1, 16, 16, c4_rp, c4_ci, c4_v}},

      {"scattered",
       "no runs, gather or scalar path",
       {1, 1, 1, a5_rp, a5_ci, a5_v},
       {1, 20, 4, b5_rp, b5_ci, b5_v},
       {1, 20, 4, c5_rp, c5_ci, c5_v}},

      {"accumulate",
       "repeated columns accumulating",
       {1, 3, 3, a6_rp, a6_ci, a6_v},
       {3, 3, 5, b6_rp, b6_ci, b6_v},
       {1, 3, 3, c6_rp, c6_ci, c6_v}},
  };

  return pairs;
}

static inline int rvsp_fx_count(void) { return 6; }

#endif /* RVSP_CSR_FIXTURES_H */
