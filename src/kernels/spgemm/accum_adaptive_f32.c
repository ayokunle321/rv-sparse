/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Picks scalar, gather, or unit-stride execution per span based on the
 * column-index structure of each row.
 */

#include "rvsp_common.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#else
#error "accum_adaptive_f32.c requires the RISC-V V extension"
#endif

/* Shortest consecutive run worth handling with unit-stride vectors. */
#ifndef RVSP_CONTIG_MIN
#define RVSP_CONTIG_MIN 8
#endif

#ifndef RVSP_CONTIG_SCAN_AHEAD
#define RVSP_CONTIG_SCAN_AHEAD 256
#endif

/* Shortest irregular span worth a gather rather than scalar. */
#ifndef RVSP_GATHER_MIN
#define RVSP_GATHER_MIN 16
#endif

static inline int32_t
rvsp_min_i32(int32_t a, int32_t b)
{
    return a < b ? a : b;
}

static inline void
rvsp_scalar_span(
    float a_val,
    int32_t n,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc)
{
    int32_t p = 0;

    for (; p + 3 < n; p += 4)
    {
        const int32_t c0 = b_col_idx[p + 0];
        const int32_t c1 = b_col_idx[p + 1];
        const int32_t c2 = b_col_idx[p + 2];
        const int32_t c3 = b_col_idx[p + 3];

        acc[c0] += a_val * b_values[p + 0];
        acc[c1] += a_val * b_values[p + 1];
        acc[c2] += a_val * b_values[p + 2];
        acc[c3] += a_val * b_values[p + 3];
    }

    for (; p < n; p++)
    {
        acc[b_col_idx[p]] += a_val * b_values[p];
    }
}

/* A run lands in consecutive acc[] slots, so it needs only a unit-stride
 * load, FMA, and store rather than a gather. */
static inline void
rvsp_contig_span(
    float a_val,
    int32_t n,
    int32_t base_col,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc)
{
    int32_t q = 0;

    while (q < n)
    {
        const size_t vl =
            __riscv_vsetvl_e32m4((size_t)(n - q));

        const vfloat32m4_t vb =
            __riscv_vle32_v_f32m4(&b_values[q], vl);

        vfloat32m4_t vacc =
            __riscv_vle32_v_f32m4(&acc[base_col + q], vl);

        vacc = __riscv_vfmacc_vf_f32m4(vacc, a_val, vb, vl);

        __riscv_vse32_v_f32m4(
            &acc[base_col + q], vacc, vl);

        q += (int32_t)vl;
    }
}

/* Canonical CSR has no repeated column in a row, so the gather and scatter
 * cannot alias within one vl. */
static inline void
rvsp_gather_span(
    float a_val,
    int32_t n,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc)
{
    int32_t q = 0;

    while (q < n)
    {
        const size_t vl =
            __riscv_vsetvl_e32m2((size_t)(n - q));

        const vfloat32m2_t vb =
            __riscv_vle32_v_f32m2(&b_values[q], vl);

        const vint32m2_t vidx =
            __riscv_vle32_v_i32m2(&b_col_idx[q], vl);

        const vuint32m2_t voff =
            __riscv_vreinterpret_v_i32m2_u32m2(
                __riscv_vsll_vx_i32m2(vidx, 2, vl));

        vfloat32m2_t vacc =
            __riscv_vluxei32_v_f32m2(acc, voff, vl);

        vacc = __riscv_vfmacc_vf_f32m2(
            vacc, a_val, vb, vl);

        __riscv_vsuxei32_v_f32m2(
            acc, voff, vacc, vl);

        q += (int32_t)vl;
    }
}

/*
 * Returns the offset to the first run of at least min_run consecutive indices
 * and writes its length to run_out. If none is found within the scan-ahead
 * cap, returns the scanned length with run_out 0.
 */
static inline int32_t
rvsp_next_contig_run(
    const int32_t *RVSP_RESTRICT idx,
    int32_t n,
    int32_t min_run,
    int32_t *RVSP_RESTRICT run_out)
{
    *run_out = 0;

    if (n <= 0)
    {
        return 0;
    }

    const int32_t limit =
        rvsp_min_i32(n, RVSP_CONTIG_SCAN_AHEAD);

    if (limit < min_run)
    {
        return limit;
    }

    for (int32_t i = 0; i + min_run <= limit; i++)
    {
        if (idx[i + 1] != idx[i] + 1)
        {
            continue;
        }

        int32_t run = 2;

        while (i + run < n &&
               idx[i + run] == idx[i] + run)
        {
            run++;
        }

        if (run >= min_run)
        {
            *run_out = run;
            return i;
        }

        i += run - 1;
    }

    return limit;
}

static inline void
rvsp_irregular_span(
    float a_val,
    int32_t n,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc)
{
    if (n >= RVSP_GATHER_MIN)
    {
        rvsp_gather_span(
            a_val, n, b_col_idx, b_values, acc);
    }
    else
    {
        rvsp_scalar_span(
            a_val, n, b_col_idx, b_values, acc);
    }
}

static inline void
rvsp_accum_row(
    float a_val,
    int32_t b_nnz,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc)
{
    if (b_nnz < RVSP_CONTIG_MIN)
    {
        rvsp_irregular_span(
            a_val, b_nnz, b_col_idx, b_values, acc);
        return;
    }

    int32_t p = 0;

    while (p < b_nnz)
    {
        const int32_t remaining = b_nnz - p;

        int32_t run = 0;

        const int32_t offset =
            rvsp_next_contig_run(
                &b_col_idx[p],
                remaining,
                RVSP_CONTIG_MIN,
                &run);

        if (offset > 0)
        {
            rvsp_irregular_span(
                a_val,
                offset,
                &b_col_idx[p],
                &b_values[p],
                acc);

            p += offset;
            continue;
        }

        if (run >= RVSP_CONTIG_MIN)
        {
            rvsp_contig_span(
                a_val,
                run,
                b_col_idx[p],
                &b_values[p],
                acc);

            p += run;
            continue;
        }

        rvsp_irregular_span(
            a_val,
            remaining,
            &b_col_idx[p],
            &b_values[p],
            acc);

        p += remaining;
    }
}

#define RVSP_KERNEL_NAME rvsp_spgemm_adaptive_f32
#include "gustavson_core_f32.inc"