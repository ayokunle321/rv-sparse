/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Column sorting. Insertion sort for short rows, LSD radix sort for longer
 * ones.
 */

#ifndef RVSP_SORT_H
#define RVSP_SORT_H

#include <stdint.h>
#include <string.h>

/* Row length below which insertion sort beats radix. */
#ifndef RVSP_SORT_INSERTION_LIMIT
#define RVSP_SORT_INSERTION_LIMIT 40
#endif

#define RVSP_RADIX_BITS 8
#define RVSP_RADIX_BUCKETS (1 << RVSP_RADIX_BITS)
#define RVSP_RADIX_MASK (RVSP_RADIX_BUCKETS - 1)

static inline void rvsp_insertion_sort_i32(int32_t *x, int32_t n) {
  for (int32_t i = 1; i < n; i++) {
    const int32_t key = x[i];
    int32_t j = i - 1;

    while (j >= 0 && x[j] > key) {
      x[j + 1] = x[j];
      j--;
    }

    x[j + 1] = key;
  }
}

static inline void rvsp_radix_sort_i32(int32_t *keys, int32_t *tmp, int32_t n,
                                       int32_t max_val) {
  if (n <= 1) {
    return;
  }

  /* One pass per radix digit, capped at the number of digits max_val needs. */
  int32_t passes = 1;

  while (passes < 4 && (max_val >> (RVSP_RADIX_BITS * passes)) > 0) {
    passes++;
  }

  int32_t *src = keys;
  int32_t *dst = tmp;

  for (int32_t p = 0; p < passes; p++) {
    int32_t count[RVSP_RADIX_BUCKETS];
    const int shift = RVSP_RADIX_BITS * p;

    memset(count, 0, sizeof(count));

    for (int32_t i = 0; i < n; i++) {
      count[(src[i] >> shift) & RVSP_RADIX_MASK]++;
    }

    int32_t running = 0;

    for (int32_t b = 0; b < RVSP_RADIX_BUCKETS; b++) {
      const int32_t c = count[b];
      count[b] = running;
      running += c;
    }

    for (int32_t i = 0; i < n; i++) {
      dst[count[(src[i] >> shift) & RVSP_RADIX_MASK]++] = src[i];
    }

    int32_t *swap = src;
    src = dst;
    dst = swap;
  }

  /* Odd pass count leaves the result in tmp, so copy it back. */
  if (src != keys) {
    memcpy(keys, src, (size_t)n * sizeof(int32_t));
  }
}

static inline void rvsp_sort_columns(int32_t *cols, int32_t *tmp, int32_t n,
                                     int32_t max_col) {
  if (n <= 1) {
    return;
  }

  if (n <= RVSP_SORT_INSERTION_LIMIT) {
    rvsp_insertion_sort_i32(cols, n);
    return;
  }

  rvsp_radix_sort_i32(cols, tmp, n, max_col);
}

#endif
