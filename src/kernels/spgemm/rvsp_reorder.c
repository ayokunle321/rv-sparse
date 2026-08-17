/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Reverse Cuthill-McKee reordering and symmetric CSR permutation. Used as a
 * preprocessing step to cluster column accesses before SpGEMM, so the dense
 * accumulator touches a narrower range and stays cache resident for longer.
 */

#include "rvsp_common.h"
#include "rvsp_reorder.h"

/*
 * Standard Cuthill-McKee: BFS from a low-degree seed, visiting neighbours in
 * increasing degree order, then reverse the result. Reversing is what turns
 * Cuthill-McKee into RCM and usually gives a tighter profile.
 *
 * Treats the matrix as the graph of its nonzero pattern. Assumes square,
 * canonical CSR. Disconnected components are handled by restarting BFS from
 * the next unvisited low-degree node.
 */
rvsp_status_t rvsp_csr_rcm_order(
    int32_t n,
    const int32_t *RVSP_RESTRICT row_ptr,
    const int32_t *RVSP_RESTRICT col_idx,
    int32_t *RVSP_RESTRICT perm,
    int32_t *RVSP_RESTRICT iperm)
{
    if (n < 0 || row_ptr == NULL || col_idx == NULL || perm == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (n == 0)
    {
        return RVSP_SUCCESS;
    }

    int32_t *degree = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    uint8_t *visited = (uint8_t *)calloc((size_t)n, sizeof(uint8_t));
    int32_t *queue = (int32_t *)malloc((size_t)n * sizeof(int32_t));

    if (degree == NULL || visited == NULL || queue == NULL)
    {
        free(degree);
        free(visited);
        free(queue);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    for (int32_t i = 0; i < n; i++)
    {
        degree[i] = row_ptr[i + 1] - row_ptr[i];
    }

    int32_t order_count = 0;

    /* Outer loop restarts BFS for each connected component. */
    for (int32_t start = 0; start < n; start++)
    {
        if (visited[start])
        {
            continue;
        }

        /* Seed each component at its lowest-degree unvisited node. */
        int32_t seed = start;
        for (int32_t i = start; i < n; i++)
        {
            if (!visited[i] && degree[i] < degree[seed])
            {
                seed = i;
            }
        }

        int32_t head = 0;
        int32_t tail = 0;

        visited[seed] = 1;
        queue[tail++] = seed;

        while (head < tail)
        {
            const int32_t node = queue[head++];
            perm[order_count++] = node;

            const int32_t r_start = row_ptr[node];
            const int32_t r_end = row_ptr[node + 1];

            /* Collect unvisited neighbours, then order them by ascending
             * degree before enqueueing. Insertion sort is fine here since a
             * row's neighbour count is small for sparse matrices. */
            const int32_t nbr_begin = tail;

            for (int32_t p = r_start; p < r_end; p++)
            {
                const int32_t nbr = col_idx[p];
                if (nbr != node && !visited[nbr])
                {
                    visited[nbr] = 1;
                    queue[tail++] = nbr;
                }
            }

            for (int32_t i = nbr_begin + 1; i < tail; i++)
            {
                const int32_t key = queue[i];
                const int32_t kd = degree[key];
                int32_t j = i - 1;
                while (j >= nbr_begin && degree[queue[j]] > kd)
                {
                    queue[j + 1] = queue[j];
                    j--;
                }
                queue[j + 1] = key;
            }
        }
    }

    /* Reverse Cuthill-McKee: reverse the ordering in place. */
    for (int32_t i = 0; i < order_count / 2; i++)
    {
        const int32_t t = perm[i];
        perm[i] = perm[order_count - 1 - i];
        perm[order_count - 1 - i] = t;
    }

    if (iperm != NULL)
    {
        for (int32_t i = 0; i < n; i++)
        {
            iperm[perm[i]] = i;
        }
    }

    free(degree);
    free(visited);
    free(queue);

    return RVSP_SUCCESS;
}

/*
 * Symmetric permutation B = P A P^T. Row r of B is old row perm[r], with each
 * old column c mapped to its new index iperm[c]. Columns are sorted within
 * each row so the result stays canonical.
 */
rvsp_status_t rvsp_csr_permute(
    int32_t n,
    const int32_t *RVSP_RESTRICT row_ptr,
    const int32_t *RVSP_RESTRICT col_idx,
    const float *RVSP_RESTRICT values,
    const int32_t *RVSP_RESTRICT perm,
    const int32_t *RVSP_RESTRICT iperm,
    int32_t **row_ptr_out,
    int32_t **col_idx_out,
    float **values_out)
{
    if (n < 0 || row_ptr == NULL || col_idx == NULL || values == NULL ||
        perm == NULL || iperm == NULL || row_ptr_out == NULL ||
        col_idx_out == NULL || values_out == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    *row_ptr_out = NULL;
    *col_idx_out = NULL;
    *values_out = NULL;

    const int32_t nnz = row_ptr[n];

    int32_t *out_row = (int32_t *)malloc(((size_t)n + 1) * sizeof(int32_t));
    int32_t *out_col = (int32_t *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int32_t));
    float *out_val = (float *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(float));
    int32_t *scratch = (int32_t *)malloc((size_t)n * sizeof(int32_t));

    if (out_row == NULL || out_col == NULL || out_val == NULL || scratch == NULL)
    {
        free(out_row);
        free(out_col);
        free(out_val);
        free(scratch);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    /* New row r has the same number of nonzeros as old row perm[r]. */
    out_row[0] = 0;
    for (int32_t r = 0; r < n; r++)
    {
        const int32_t old = perm[r];
        out_row[r + 1] = out_row[r] + (row_ptr[old + 1] - row_ptr[old]);
    }

    const int32_t max_col_val = n > 0 ? n - 1 : 0;

    for (int32_t r = 0; r < n; r++)
    {
        const int32_t old = perm[r];
        const int32_t o_start = row_ptr[old];
        const int32_t o_end = row_ptr[old + 1];

        int32_t *const dst_col = &out_col[out_row[r]];
        float *const dst_val = &out_val[out_row[r]];
        int32_t count = 0;

        for (int32_t p = o_start; p < o_end; p++)
        {
            dst_col[count] = iperm[col_idx[p]];
            dst_val[count] = values[p];
            count++;
        }

        /* Remapped columns are no longer sorted, so sort each row to keep the
         * output canonical. Values ride along by sorting index pairs. */
        for (int32_t i = 1; i < count; i++)
        {
            const int32_t kc = dst_col[i];
            const float kv = dst_val[i];
            int32_t j = i - 1;
            while (j >= 0 && dst_col[j] > kc)
            {
                dst_col[j + 1] = dst_col[j];
                dst_val[j + 1] = dst_val[j];
                j--;
            }
            dst_col[j + 1] = kc;
            dst_val[j + 1] = kv;
        }
    }

    (void)max_col_val;

    free(scratch);

    *row_ptr_out = out_row;
    *col_idx_out = out_col;
    *values_out = out_val;

    return RVSP_SUCCESS;
}