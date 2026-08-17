/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Internal workspace layout and compiler helpers.
 */

#ifndef RVSP_COMMON_H
#define RVSP_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "rv_sparse.h"
#include "rvsp_spgemm.h"
#include "rvsp_sort.h"

#if defined(__GNUC__) || defined(__clang__)
#define RVSP_RESTRICT __restrict__
#define RVSP_LIKELY(x) __builtin_expect(!!(x), 1)
#define RVSP_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define RVSP_RESTRICT
#define RVSP_LIKELY(x) (x)
#define RVSP_UNLIKELY(x) (x)
#endif

#define RVSP_ALIGN_UP(x, a) \
    (((x) + ((a) - 1)) & ~((size_t)(a) - 1))

/*
 * Scratch shared across a call. acc is the dense accumulator, mark and
 * touched track which columns are live during the symbolic pass, and scratch
 * backs the column sort. Each region is 64-byte aligned in one buffer.
 */
typedef struct
{
    float *acc;
    uint8_t *mark;
    int32_t *touched;
    int32_t *scratch;
} rvsp_ws_t;

static inline size_t
rvsp_ws_bytes(int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    return RVSP_ALIGN_UP(n * sizeof(float), 64) +
           RVSP_ALIGN_UP(n * sizeof(uint8_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64);
}

static inline void
rvsp_ws_bind(
    rvsp_ws_t *ws,
    void *buffer,
    int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    unsigned char *p = (unsigned char *)buffer;

    ws->acc = (float *)p;
    p += RVSP_ALIGN_UP(n * sizeof(float), 64);

    ws->mark = (uint8_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(uint8_t), 64);

    ws->touched = (int32_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(int32_t), 64);

    ws->scratch = (int32_t *)p;
}

/* Symbolic count needs only mark and touched. */
static inline size_t
rvsp_count_ws_bytes(int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    return RVSP_ALIGN_UP(n * sizeof(uint8_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64);
}

static inline void
rvsp_count_ws_bind(
    rvsp_ws_t *ws,
    void *buffer,
    int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    unsigned char *p = (unsigned char *)buffer;

    ws->acc = NULL;

    ws->mark = (uint8_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(uint8_t), 64);

    ws->touched = (int32_t *)p;
    ws->scratch = NULL;
}

/* Numeric compute needs acc, mark, and scratch but not touched. */
static inline size_t
rvsp_compute_ws_bytes(int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    return RVSP_ALIGN_UP(n * sizeof(float), 64) +
           RVSP_ALIGN_UP(n * sizeof(uint8_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64);
}

static inline void
rvsp_compute_ws_bind(
    rvsp_ws_t *ws,
    void *buffer,
    int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    unsigned char *p = (unsigned char *)buffer;

    ws->acc = (float *)p;
    p += RVSP_ALIGN_UP(n * sizeof(float), 64);

    ws->mark = (uint8_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(uint8_t), 64);

    ws->scratch = (int32_t *)p;
    ws->touched = NULL;
}

/*
 * OpenMP scalar workspace.
 *
 * acc holds one full accumulator per thread, so the numeric pass can run rows
 * in parallel without two threads sharing a slot. mark, touched and scratch
 * serve the serial symbolic pass and are single copies.
 *
 * Sized by the caller because the thread count is not known to the kernel until
 * it runs, and allocating it inside would put tens of megabytes of calloc in
 * the timed region at high thread counts.
 */
typedef struct
{
    float *acc; /* nthreads * b_cols */
    uint8_t *mark;
    int32_t *touched;
    int32_t *scratch;
} rvsp_omp_ws_t;

static inline size_t
rvsp_omp_ws_bytes(int32_t b_cols, int32_t nthreads)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);
    const size_t t = (size_t)(nthreads > 0 ? nthreads : 1);

    return RVSP_ALIGN_UP(n * t * sizeof(float), 64) +
           RVSP_ALIGN_UP(n * sizeof(uint8_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64);
}

static inline void
rvsp_omp_ws_bind(rvsp_omp_ws_t *ws, void *buffer, int32_t b_cols,
                 int32_t nthreads)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);
    const size_t t = (size_t)(nthreads > 0 ? nthreads : 1);

    unsigned char *p = (unsigned char *)buffer;

    ws->acc = (float *)p;
    p += RVSP_ALIGN_UP(n * t * sizeof(float), 64);

    ws->mark = (uint8_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(uint8_t), 64);

    ws->touched = (int32_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(int32_t), 64);

    ws->scratch = (int32_t *)p;
}

/*
 * MAGNUS workspace.
 *
 * chunk_acc is one chunk-wide dense accumulator reused across chunks, sized to
 * stay L2 resident. counts, offsets and cursor are per-chunk histograms. The
 * bin buffers hold one row's products reordered by chunk, so they are sized by
 * the largest per-row intermediate product count, which the caller already
 * computes as op_max before the timed region.
 */
#ifndef RVSP_MAGNUS_CHUNK_LOG2
#define RVSP_MAGNUS_CHUNK_LOG2 16
#endif

#define RVSP_MAGNUS_CHUNK_WIDTH (1 << RVSP_MAGNUS_CHUNK_LOG2)

typedef struct
{
    uint8_t *mark;
    int32_t *touched;
    int32_t *scratch;
    float *chunk_acc;
    int32_t *counts;
    int32_t *offsets;
    int32_t *cursor;
    int32_t *bin_col;
    float *bin_val;
} rvsp_magnus_ws_t;

static inline int32_t
rvsp_magnus_n_chunks(int32_t b_cols)
{
    const int32_t n = b_cols > 0 ? b_cols : 1;
    return (n + RVSP_MAGNUS_CHUNK_WIDTH - 1) >> RVSP_MAGNUS_CHUNK_LOG2;
}

static inline size_t
rvsp_magnus_ws_bytes(int32_t b_cols, int32_t max_prod)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);
    const size_t c = (size_t)rvsp_magnus_n_chunks(b_cols);
    const size_t p = (size_t)(max_prod > 0 ? max_prod : 1);

    return RVSP_ALIGN_UP(n * sizeof(uint8_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64) +
           RVSP_ALIGN_UP((size_t)RVSP_MAGNUS_CHUNK_WIDTH * sizeof(float), 64) +
           RVSP_ALIGN_UP(c * sizeof(int32_t), 64) +
           RVSP_ALIGN_UP((c + 1) * sizeof(int32_t), 64) +
           RVSP_ALIGN_UP(c * sizeof(int32_t), 64) +
           RVSP_ALIGN_UP(p * sizeof(int32_t), 64) +
           RVSP_ALIGN_UP(p * sizeof(float), 64);
}

static inline void
rvsp_magnus_ws_bind(rvsp_magnus_ws_t *ws, void *buffer, int32_t b_cols,
                    int32_t max_prod)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);
    const size_t c = (size_t)rvsp_magnus_n_chunks(b_cols);
    const size_t p = (size_t)(max_prod > 0 ? max_prod : 1);

    unsigned char *q = (unsigned char *)buffer;

    ws->mark = (uint8_t *)q;
    q += RVSP_ALIGN_UP(n * sizeof(uint8_t), 64);

    ws->touched = (int32_t *)q;
    q += RVSP_ALIGN_UP(n * sizeof(int32_t), 64);

    ws->scratch = (int32_t *)q;
    q += RVSP_ALIGN_UP(n * sizeof(int32_t), 64);

    ws->chunk_acc = (float *)q;
    q += RVSP_ALIGN_UP((size_t)RVSP_MAGNUS_CHUNK_WIDTH * sizeof(float), 64);

    ws->counts = (int32_t *)q;
    q += RVSP_ALIGN_UP(c * sizeof(int32_t), 64);

    ws->offsets = (int32_t *)q;
    q += RVSP_ALIGN_UP((c + 1) * sizeof(int32_t), 64);

    ws->cursor = (int32_t *)q;
    q += RVSP_ALIGN_UP(c * sizeof(int32_t), 64);

    ws->bin_col = (int32_t *)q;
    q += RVSP_ALIGN_UP(p * sizeof(int32_t), 64);

    ws->bin_val = (float *)q;
}

#endif
