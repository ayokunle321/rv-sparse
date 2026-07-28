#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

#include "csr_spgemm_kernels.h"

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

#ifndef RVSP_SORT_INSERTION_LIMIT
#define RVSP_SORT_INSERTION_LIMIT 32
#endif

static int compare_i32_rvv_f32(const void *a, const void *b)
{
    const int32_t ia = *(const int32_t *)a;
    const int32_t ib = *(const int32_t *)b;

    if (ia < ib)
    {
        return -1;
    }

    if (ia > ib)
    {
        return 1;
    }

    return 0;
}

static void insertion_sort_i32_rvv_f32(int32_t *x, int32_t n)
{
    for (int32_t i = 1; i < n; i++)
    {
        const int32_t key = x[i];
        int32_t j = i - 1;

        while (j >= 0 && x[j] > key)
        {
            x[j + 1] = x[j];
            j--;
        }

        x[j + 1] = key;
    }
}

static void sort_touched_columns_i32_rvv_f32(
    int32_t *touched,
    int32_t touched_count)
{
    if (touched_count <= 1)
    {
        return;
    }

    if (touched_count <= RVSP_SORT_INSERTION_LIMIT)
    {
        insertion_sort_i32_rvv_f32(touched, touched_count);
        return;
    }

    qsort(
        touched,
        (size_t)touched_count,
        sizeof(int32_t),
        compare_i32_rvv_f32);
}

static void clear_f32_workspace(float *acc,
                                uint8_t *mark,
                                const int32_t *touched,
                                int32_t touched_count)
{
    for (int32_t i = 0; i < touched_count; i++)
    {
        const int32_t col = touched[i];
        acc[col] = 0.0f;
        mark[col] = 0;
    }
}

static void clear_f32_marks_only(
    uint8_t *mark,
    const int32_t *touched,
    int32_t touched_count)
{
    for (int32_t i = 0; i < touched_count; i++)
    {
        mark[touched[i]] = 0;
    }
}

static rvsp_status_t mark_f32_row_columns(
    int32_t b_nnz,
    const int32_t *b_col_idx,
    int32_t b_cols,
    float *acc,
    uint8_t *mark,
    int32_t *touched,
    int32_t *touched_count)
{
    for (int32_t p = 0; p < b_nnz; p++)
    {
        const int32_t col = b_col_idx[p];

        if (RVSP_UNLIKELY(col < 0 || col >= b_cols))
        {
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        if (mark[col] == 0)
        {
            mark[col] = 1;
            touched[*touched_count] = col;
            (*touched_count)++;
            acc[col] = 0.0f;
        }
    }

    return RVSP_SUCCESS;
}

static rvsp_status_t mark_f32_row_columns_symbolic(int32_t b_nnz, const int32_t *b_col_idx, int32_t b_cols,
                                                   uint8_t *mark, int32_t *touched, int32_t *touched_count)
{
    for (int32_t p = 0; p < b_nnz; p++)
    {
        const int32_t col = b_col_idx[p];

        if (RVSP_UNLIKELY(col < 0 || col >= b_cols))
        {
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        if (mark[col] == 0)
        {
            mark[col] = 1;
            touched[*touched_count] = col;
            (*touched_count)++;
        }
    }

    return RVSP_SUCCESS;
}

static rvsp_status_t accumulate_f32_row_scalar_marked(float a_val,
                                                      int32_t b_nnz,
                                                      const int32_t *b_col_idx,
                                                      const float *b_values,
                                                      int32_t b_cols,
                                                      float *acc,
                                                      uint8_t *mark,
                                                      int32_t *touched,
                                                      int32_t *touched_count)
{
    int32_t p = 0;

    for (; p + 3 < b_nnz; p += 4)
    {
        const int32_t c0 = b_col_idx[p + 0];
        const int32_t c1 = b_col_idx[p + 1];
        const int32_t c2 = b_col_idx[p + 2];
        const int32_t c3 = b_col_idx[p + 3];

        if (RVSP_UNLIKELY(c0 < 0 || c0 >= b_cols ||
                          c1 < 0 || c1 >= b_cols ||
                          c2 < 0 || c2 >= b_cols ||
                          c3 < 0 || c3 >= b_cols))
        {
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        if (mark[c0] == 0)
        {
            mark[c0] = 1;
            touched[*touched_count] = c0;
            (*touched_count)++;
            acc[c0] = 0.0f;
        }

        if (mark[c1] == 0)
        {
            mark[c1] = 1;
            touched[*touched_count] = c1;
            (*touched_count)++;
            acc[c1] = 0.0f;
        }

        if (mark[c2] == 0)
        {
            mark[c2] = 1;
            touched[*touched_count] = c2;
            (*touched_count)++;
            acc[c2] = 0.0f;
        }

        if (mark[c3] == 0)
        {
            mark[c3] = 1;
            touched[*touched_count] = c3;
            (*touched_count)++;
            acc[c3] = 0.0f;
        }

        acc[c0] += a_val * b_values[p + 0];
        acc[c1] += a_val * b_values[p + 1];
        acc[c2] += a_val * b_values[p + 2];
        acc[c3] += a_val * b_values[p + 3];
    }

    for (; p < b_nnz; p++)
    {
        const int32_t col = b_col_idx[p];

        if (RVSP_UNLIKELY(col < 0 || col >= b_cols))
        {
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        if (mark[col] == 0)
        {
            mark[col] = 1;
            touched[*touched_count] = col;
            (*touched_count)++;
            acc[col] = 0.0f;
        }

        acc[col] += a_val * b_values[p];
    }

    return RVSP_SUCCESS;
}

static rvsp_status_t build_b_duplicate_flags_f32(int32_t b_rows, int32_t b_cols, const int32_t *b_row_ptr,
                                                 const int32_t *b_col_idx,
                                                 uint8_t *b_has_duplicates)
{
    uint8_t *seen = NULL;
    int32_t *seen_touched = NULL;
    rvsp_status_t status = RVSP_SUCCESS;

    if (b_cols > 0)
    {
        seen = (uint8_t *)calloc((size_t)b_cols, sizeof(uint8_t));
        seen_touched = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));

        if (seen == NULL || seen_touched == NULL)
        {
            status = RVSP_ERROR_ALLOCATION_FAILED;
            goto cleanup;
        }
    }

    for (int32_t row = 0; row < b_rows; row++)
    {
        const int32_t start = b_row_ptr[row];
        const int32_t end = b_row_ptr[row + 1];

        if (RVSP_UNLIKELY(start < 0 || end < start))
        {
            status = RVSP_ERROR_INVALID_ARGUMENT;
            goto cleanup;
        }

        if (end - start <= 1)
        {
            continue;
        }

        int32_t seen_count = 0;

        for (int32_t p = start; p < end; p++)
        {
            const int32_t col = b_col_idx[p];

            if (RVSP_UNLIKELY(col < 0 || col >= b_cols))
            {
                status = RVSP_ERROR_INVALID_ARGUMENT;
                break;
            }

            if (seen[col] != 0)
            {
                b_has_duplicates[row] = 1;
            }
            else
            {
                seen[col] = 1;
                seen_touched[seen_count] = col;
                seen_count++;
            }
        }

        for (int32_t i = 0; i < seen_count; i++)
        {
            seen[seen_touched[i]] = 0;
        }

        if (status != RVSP_SUCCESS)
        {
            goto cleanup;
        }
    }

cleanup:
    free(seen);
    free(seen_touched);
    return status;
}

static inline int should_use_rvv_f32(
    int32_t b_nnz,
    uint8_t has_duplicates)
{
    if (has_duplicates)
    {
        return 0;
    }

    if (b_nnz < RVSP_RVV_F32_MIN_B_NNZ)
    {
        return 0;
    }

    return 1;
}

static rvsp_status_t accumulate_f32_row_rvv_or_scalar(
    float a_val,
    int32_t b_nnz,
    const int32_t *b_col_idx,
    const float *b_values,
    int32_t b_cols,
    float *acc,
    uint8_t *mark,
    int32_t *touched,
    int32_t *touched_count,
    uint8_t has_duplicates)
{
    if (!should_use_rvv_f32(b_nnz, has_duplicates))
    {
        return accumulate_f32_row_scalar_marked(a_val, b_nnz, b_col_idx, b_values,
                                                b_cols,
                                                acc,
                                                mark,
                                                touched,
                                                touched_count);
    }

    rvsp_status_t status = mark_f32_row_columns(b_nnz, b_col_idx, b_cols, acc, mark, touched, touched_count);

    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    return rvsp_accumulate_row_f32_rvv_indexed_fast(a_val, b_nnz, b_col_idx, b_values, acc);
}

static rvsp_status_t symbolic_count_pass_f32(int32_t a_rows, int32_t a_cols, int32_t b_cols, const int32_t *a_row_ptr,
                                             const int32_t *a_col_idx,
                                             const int32_t *b_row_ptr, const int32_t *b_col_idx,
                                             uint8_t *mark, int32_t *touched, int32_t *c_row_ptr,
                                             int64_t *total_nnz_out)
{
    int64_t total_nnz = 0;

    for (int32_t row = 0; row < a_rows; row++)
    {
        const int32_t a_start = a_row_ptr[row];
        const int32_t a_end = a_row_ptr[row + 1];

        if (RVSP_UNLIKELY(a_start < 0 || a_end < a_start))
        {
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        int32_t touched_count = 0;

        for (int32_t ap = a_start; ap < a_end; ap++)
        {
            const int32_t a_col = a_col_idx[ap];

            if (RVSP_UNLIKELY(a_col < 0 || a_col >= a_cols))
            {
                clear_f32_marks_only(mark, touched, touched_count);
                return RVSP_ERROR_INVALID_ARGUMENT;
            }

            const int32_t b_start = b_row_ptr[a_col];
            const int32_t b_end = b_row_ptr[a_col + 1];

            if (RVSP_UNLIKELY(b_start < 0 || b_end < b_start))
            {
                clear_f32_marks_only(mark, touched, touched_count);
                return RVSP_ERROR_INVALID_ARGUMENT;
            }

            rvsp_status_t status = mark_f32_row_columns_symbolic(b_end - b_start, &b_col_idx[b_start],
                                                                 b_cols, mark, touched, &touched_count);

            if (status != RVSP_SUCCESS)
            {
                clear_f32_marks_only(mark, touched, touched_count);
                return status;
            }
        }

        c_row_ptr[row] = touched_count;
        total_nnz += touched_count;

        clear_f32_marks_only(mark, touched, touched_count);

        if (RVSP_UNLIKELY(total_nnz > INT32_MAX))
        {
            return RVSP_ERROR_ALLOCATION_FAILED;
        }
    }

    int32_t running = 0;

    for (int32_t row = 0; row < a_rows; row++)
    {
        const int32_t count = c_row_ptr[row];
        c_row_ptr[row] = running;
        running += count;
    }

    c_row_ptr[a_rows] = running;
    *total_nnz_out = total_nnz;

    return RVSP_SUCCESS;
}

static rvsp_status_t numeric_emit_pass_f32(int32_t a_rows, int32_t a_cols, int32_t b_cols, const int32_t *a_row_ptr,
                                           const int32_t *a_col_idx, const float *a_values, const int32_t *b_row_ptr,
                                           const int32_t *b_col_idx, const float *b_values, const uint8_t *b_has_duplicates,
                                           float *acc, uint8_t *mark, int32_t *touched,
                                           int32_t *c_row_ptr, int32_t *c_col_idx, float *c_values)
{
    (void)a_cols;

    for (int32_t row = 0; row < a_rows; row++)
    {
        const int32_t a_start = a_row_ptr[row];
        const int32_t a_end = a_row_ptr[row + 1];

        int32_t touched_count = 0;
        int32_t dst = c_row_ptr[row];
        const int32_t row_capacity_end = c_row_ptr[row + 1];

        for (int32_t ap = a_start; ap < a_end; ap++)
        {
            const int32_t a_col = a_col_idx[ap];
            const int32_t b_start = b_row_ptr[a_col];
            const int32_t b_end = b_row_ptr[a_col + 1];

            rvsp_status_t status = accumulate_f32_row_rvv_or_scalar(
                a_values[ap],
                b_end - b_start,
                &b_col_idx[b_start],
                &b_values[b_start],
                b_cols,
                acc,
                mark,
                touched,
                &touched_count,
                b_has_duplicates ? b_has_duplicates[a_col] : 0);

            if (status != RVSP_SUCCESS)
            {
                clear_f32_workspace(acc, mark, touched, touched_count);
                return status;
            }
        }

        sort_touched_columns_i32_rvv_f32(touched, touched_count);

        for (int32_t i = 0; i < touched_count; i++)
        {
            const int32_t col = touched[i];

            if (acc[col] != 0.0f)
            {
                if (RVSP_UNLIKELY(dst >= row_capacity_end))
                {
                    clear_f32_workspace(acc, mark, touched, touched_count);
                    return RVSP_ERROR_INVALID_ARGUMENT;
                }

                c_col_idx[dst] = col;
                c_values[dst] = acc[col];
                dst++;
            }
            mark[col] = 0;
        }

        c_row_ptr[row + 1] = dst;
    }

    return RVSP_SUCCESS;
}

rvsp_status_t rvsp_spgemm_csr_rvv_f32_indexed_marked_raw(int32_t a_rows, int32_t a_cols, int32_t b_cols,
                                                         const int32_t *a_row_ptr, const int32_t *a_col_idx,
                                                         const float *a_values, const int32_t *b_row_ptr,
                                                         const int32_t *b_col_idx, const float *b_values,
                                                         int32_t **c_row_ptr_out, int32_t **c_col_idx_out,
                                                         float **c_values_out, int32_t *c_nnz_out)
{
    rvsp_status_t status = RVSP_SUCCESS;

    int32_t *c_row_ptr = NULL;
    int32_t *c_col_idx = NULL;
    float *c_values = NULL;

    float *acc = NULL;
    int32_t *touched = NULL;
    uint8_t *mark = NULL;
    uint8_t *b_has_duplicates = NULL;

    int64_t total_nnz_upper = 0;

    if (RVSP_UNLIKELY(a_rows < 0 || a_cols < 0 || b_cols < 0 ||
                      a_row_ptr == NULL || a_col_idx == NULL ||
                      a_values == NULL ||
                      b_row_ptr == NULL || b_col_idx == NULL ||
                      b_values == NULL ||
                      c_row_ptr_out == NULL ||
                      c_col_idx_out == NULL ||
                      c_values_out == NULL ||
                      c_nnz_out == NULL))
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    *c_row_ptr_out = NULL;
    *c_col_idx_out = NULL;
    *c_values_out = NULL;
    *c_nnz_out = 0;

    c_row_ptr = (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));
    if (c_row_ptr == NULL)
    {
        status = RVSP_ERROR_ALLOCATION_FAILED;
        goto fail;
    }

    if (b_cols > 0)
    {
        acc = (float *)malloc((size_t)b_cols * sizeof(float));
        touched = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));
        mark = (uint8_t *)calloc((size_t)b_cols, sizeof(uint8_t));

        if (acc == NULL || touched == NULL || mark == NULL)
        {
            status = RVSP_ERROR_ALLOCATION_FAILED;
            goto fail;
        }
    }

    if (a_cols > 0)
    {
        b_has_duplicates = (uint8_t *)calloc((size_t)a_cols, sizeof(uint8_t));
        if (b_has_duplicates == NULL)
        {
            status = RVSP_ERROR_ALLOCATION_FAILED;
            goto fail;
        }

        status = build_b_duplicate_flags_f32(a_cols, b_cols, b_row_ptr, b_col_idx, b_has_duplicates);

        if (status != RVSP_SUCCESS)
        {
            goto fail;
        }
    }

    status = symbolic_count_pass_f32(a_rows, a_cols, b_cols, a_row_ptr, a_col_idx,
                                     b_row_ptr, b_col_idx, mark, touched, c_row_ptr,
                                     &total_nnz_upper);

    if (status != RVSP_SUCCESS)
    {
        goto fail;
    }

    {
        const size_t alloc_nnz =
            total_nnz_upper > 0 ? (size_t)total_nnz_upper : 1;

        c_col_idx = (int32_t *)malloc(alloc_nnz * sizeof(int32_t));
        c_values = (float *)malloc(alloc_nnz * sizeof(float));

        if (c_col_idx == NULL || c_values == NULL)
        {
            status = RVSP_ERROR_ALLOCATION_FAILED;
            goto fail;
        }
    }

    status = numeric_emit_pass_f32(a_rows, a_cols, b_cols, a_row_ptr, a_col_idx, a_values,
                                   b_row_ptr, b_col_idx, b_values, b_has_duplicates, acc,
                                   mark, touched, c_row_ptr, c_col_idx, c_values);

    if (status != RVSP_SUCCESS)
    {
        goto fail;
    }

    free(acc);
    free(touched);
    free(mark);
    free(b_has_duplicates);

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = c_row_ptr[a_rows];

    return RVSP_SUCCESS;

fail:
    free(c_row_ptr);
    free(c_col_idx);
    free(c_values);
    free(acc);
    free(touched);
    free(mark);
    free(b_has_duplicates);

    return status;
}