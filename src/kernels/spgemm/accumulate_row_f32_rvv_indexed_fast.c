#include <stdint.h>
#include <stddef.h>

#include "csr_spgemm_kernels.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#define RVSP_HAVE_RVV_INTRINSICS 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define RVSP_LIKELY(x) __builtin_expect(!!(x), 1)
#define RVSP_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define RVSP_LIKELY(x) (x)
#define RVSP_UNLIKELY(x) (x)
#endif

#ifndef RVSP_RVV_F32_MIN_B_NNZ
#define RVSP_RVV_F32_MIN_B_NNZ 512
#endif

#ifndef RVSP_RVV_F32_CONTIG_MIN
#define RVSP_RVV_F32_CONTIG_MIN 8
#endif

static inline void rvsp_accumulate_row_f32_scalar_unroll4(float a_val, int32_t b_nnz, const int32_t *b_col_idx, const float *b_values, float *acc)
{
    int32_t p = 0;

    for (; p + 3 < b_nnz; p += 4)
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

    for (; p < b_nnz; p++)
    {
        const int32_t col = b_col_idx[p];
        acc[col] += a_val * b_values[p];
    }
}

static inline int32_t rvsp_consecutive_run_i32(const int32_t *idx, int32_t n)
{
    if (RVSP_UNLIKELY(n <= 0))
    {
        return 0;
    }

    const int32_t base = idx[0];
    int32_t run = 1;

    while (run < n && idx[run] == base + run)
    {
        run++;
    }

    return run;
}

rvsp_status_t rvsp_accumulate_row_f32_rvv_indexed_fast(float a_val, int32_t b_nnz, const int32_t *b_col_idx, const float *b_values, float *acc)
{
    if (RVSP_UNLIKELY(b_nnz < 0 || b_col_idx == NULL || b_values == NULL || acc == NULL))
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (b_nnz == 0)
    {
        return RVSP_SUCCESS;
    }
    if (b_nnz < RVSP_RVV_F32_MIN_B_NNZ)
    {
        rvsp_accumulate_row_f32_scalar_unroll4(a_val, b_nnz, b_col_idx, b_values, acc);
        return RVSP_SUCCESS;
    }

#if defined(RVSP_HAVE_RVV_INTRINSICS)
    int32_t p = 0;

    while (p < b_nnz)
    {
        const int32_t remaining = b_nnz - p;
        const int32_t run =
            rvsp_consecutive_run_i32(&b_col_idx[p], remaining);

        if (run >= RVSP_RVV_F32_CONTIG_MIN)
        {
            const int32_t base_col = b_col_idx[p];
            int32_t q = 0;

            while (q < run)
            {
                const size_t vl =
                    __riscv_vsetvl_e32m4((size_t)(run - q));

                const vfloat32m4_t vb =
                    __riscv_vle32_v_f32m4(&b_values[p + q], vl);

                vfloat32m4_t vacc =
                    __riscv_vle32_v_f32m4(&acc[base_col + q], vl);

                vacc = __riscv_vfmacc_vf_f32m4(vacc, a_val, vb, vl);

                __riscv_vse32_v_f32m4(&acc[base_col + q], vacc, vl);

                q += (int32_t)vl;
            }

            p += run;
            continue;
        }

        {
            const size_t vl =
                __riscv_vsetvl_e32m2((size_t)remaining);

            const vfloat32m2_t vb =
                __riscv_vle32_v_f32m2(&b_values[p], vl);

            const vint32m2_t vidx_i32 =
                __riscv_vle32_v_i32m2(&b_col_idx[p], vl);

            const vint32m2_t voff_i32 =
                __riscv_vsll_vx_i32m2(vidx_i32, 2, vl);

            const vuint32m2_t voff_u32 =
                __riscv_vreinterpret_v_i32m2_u32m2(voff_i32);

            vfloat32m2_t vacc =
                __riscv_vluxei32_v_f32m2(acc, voff_u32, vl);

            vacc = __riscv_vfmacc_vf_f32m2(vacc, a_val, vb, vl);

            __riscv_vsuxei32_v_f32m2(acc, voff_u32, vacc, vl);

            p += (int32_t)vl;
        }
    }
#else
    rvsp_accumulate_row_f32_scalar_unroll4(a_val, b_nnz, b_col_idx, b_values, acc);
#endif

    return RVSP_SUCCESS;
}