/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * MAGNUS fine-level driver. The numeric pass partitions the output column range
 * into fixed power-of-two chunks so the active accumulator slice stays L2
 * resident. Products are binned per chunk, then accumulated one chunk at a time.
 * Symbolic structure is built by the shared symbolic phase, so this holds
 * structure generation fixed and isolates the accumulation-locality effect.
 *
 * The reorder pass uses regular stores. The reference algorithm uses AVX-512
 * non-temporal streaming stores to keep the binned data out of L2, which
 * RISC-V lacks, so the reorder pass here competes with the accumulator for L2.
 */

#ifndef RVSP_KERNEL_NAME
#error "define RVSP_KERNEL_NAME before including magnus_core_f32.inc"
#endif

/* Chunk width in columns. A power of two so the chunk index is a shift.
 * 65536 columns is a 256 KB fp32 accumulator, resident in the 512 KB L2 with
 * room for the bin buffers. */
#ifndef RVSP_MAGNUS_CHUNK_LOG2
#define RVSP_MAGNUS_CHUNK_LOG2 16
#endif

#define RVSP_MAGNUS_CHUNK_WIDTH (1 << RVSP_MAGNUS_CHUNK_LOG2)

#define RVSP_CAT_(a, b) a##b
#define RVSP_CAT(a, b) RVSP_CAT_(a, b)
#define RVSP_NUMERIC_FN RVSP_CAT(RVSP_KERNEL_NAME, _numeric)

void RVSP_NUMERIC_FN(
    int32_t a_rows,
    const int32_t *RVSP_RESTRICT a_row_ptr,
    const int32_t *RVSP_RESTRICT a_col_idx,
    const float *RVSP_RESTRICT a_values,
    const int32_t *RVSP_RESTRICT b_row_ptr,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc,
    const int32_t *RVSP_RESTRICT c_row_ptr,
    const int32_t *RVSP_RESTRICT c_col_idx,
    float *RVSP_RESTRICT c_values)
{
    (void)acc;
    (void)a_rows;
    (void)a_row_ptr;
    (void)a_col_idx;
    (void)a_values;
    (void)b_row_ptr;
    (void)b_col_idx;
    (void)b_values;
    (void)c_row_ptr;
    (void)c_col_idx;
    (void)c_values;
    /* Not used. MAGNUS binds its own multi-buffer workspace and runs the
     * chunked numeric inside the public entry below, because it needs the
     * per-chunk bin arrays that the shared numeric signature does not carry. */
}

rvsp_status_t RVSP_KERNEL_NAME(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *a_row_ptr,
    const int32_t *a_col_idx,
    const float *a_values,
    const int32_t *b_row_ptr,
    const int32_t *b_col_idx,
    const float *b_values,
    void *workspace,
    size_t workspace_bytes,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    float **c_values_out,
    int32_t *c_nnz_out,
    int64_t *op_counts_out)
{
    (void)a_cols;

    if (a_row_ptr == NULL || a_col_idx == NULL || a_values == NULL ||
        b_row_ptr == NULL || b_col_idx == NULL || b_values == NULL ||
        c_row_ptr_out == NULL || c_col_idx_out == NULL ||
        c_values_out == NULL || c_nnz_out == NULL || workspace == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (a_rows <= 0 || a_cols <= 0 || b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    *c_row_ptr_out = NULL;
    *c_col_idx_out = NULL;
    *c_values_out = NULL;
    *c_nnz_out = 0;

    const int32_t n_chunks = rvsp_magnus_n_chunks(b_cols);

    /* Largest per-row intermediate product count. This sizes the bin buffers,
     * and it is the same quantity the harness reports as op_max, so the caller
     * can size the workspace without running the kernel. */
    int32_t max_prod = 0;
    for (int32_t row = 0; row < a_rows; row++)
    {
        int32_t prod = 0;
        for (int32_t ap = a_row_ptr[row]; ap < a_row_ptr[row + 1]; ap++)
        {
            const int32_t k = a_col_idx[ap];
            prod += b_row_ptr[k + 1] - b_row_ptr[k];
        }
        if (prod > max_prod)
        {
            max_prod = prod;
        }
    }

    /* The caller sizes and owns this. Allocating here would put every one of
     * these buffers inside the timed region. */
    if (workspace_bytes < rvsp_magnus_ws_bytes(b_cols, max_prod))
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    rvsp_magnus_ws_t ws;
    rvsp_magnus_ws_bind(&ws, workspace, b_cols, max_prod);

    uint8_t *mark = ws.mark;
    int32_t *touched = ws.touched;
    int32_t *scratch = ws.scratch;
    float *chunk_acc = ws.chunk_acc;
    int32_t *counts = ws.counts;
    int32_t *offsets = ws.offsets;
    int32_t *cursor = ws.cursor;
    int32_t *bin_col = ws.bin_col;
    float *bin_val = ws.bin_val;

    /* The caller owns the buffer but not its contents. */
    memset(mark, 0, (size_t)b_cols * sizeof(uint8_t));
    memset(chunk_acc, 0, (size_t)RVSP_MAGNUS_CHUNK_WIDTH * sizeof(float));

    int32_t *c_row_ptr =
        (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));

    if (c_row_ptr == NULL)
    {
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    int64_t total_nnz = 0;

    rvsp_status_t status =
        rvsp_symbolic_count(
            a_rows, a_row_ptr, a_col_idx, b_row_ptr, b_col_idx,
            mark, touched, c_row_ptr, op_counts_out, &total_nnz);

    if (status != RVSP_SUCCESS)
    {
        free(c_row_ptr);
        return status;
    }

    const size_t alloc_nnz = total_nnz > 0 ? (size_t)total_nnz : 1;

    int32_t *c_col_idx = (int32_t *)malloc(alloc_nnz * sizeof(int32_t));
    float *c_values = (float *)malloc(alloc_nnz * sizeof(float));

    if (c_col_idx == NULL || c_values == NULL)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    rvsp_symbolic_fill(
        a_rows, b_cols, a_row_ptr, a_col_idx, b_row_ptr, b_col_idx,
        mark, scratch, c_row_ptr, c_col_idx);

    /* Numeric pass, per row, chunked so the active accumulator stays L2 resident. */
    for (int32_t row = 0; row < a_rows; row++)
    {
        const int32_t a_start = a_row_ptr[row];
        const int32_t a_end = a_row_ptr[row + 1];

        /* Histogram: how many products fall in each chunk. */
        for (int32_t j = 0; j < n_chunks; j++)
        {
            counts[j] = 0;
        }

        for (int32_t ap = a_start; ap < a_end; ap++)
        {
            const int32_t k = a_col_idx[ap];
            const int32_t b_start = b_row_ptr[k];
            const int32_t b_end = b_row_ptr[k + 1];
            for (int32_t bp = b_start; bp < b_end; bp++)
            {
                counts[b_col_idx[bp] >> RVSP_MAGNUS_CHUNK_LOG2]++;
            }
        }

        /* Prefix sum: chunk offsets into the bin buffers. */
        offsets[0] = 0;
        for (int32_t j = 0; j < n_chunks; j++)
        {
            offsets[j + 1] = offsets[j] + counts[j];
            cursor[j] = offsets[j];
        }

        /* Reorder: bin every product by its chunk. */
        for (int32_t ap = a_start; ap < a_end; ap++)
        {
            const int32_t k = a_col_idx[ap];
            const float a_val = a_values[ap];
            const int32_t b_start = b_row_ptr[k];
            const int32_t b_end = b_row_ptr[k + 1];
            for (int32_t bp = b_start; bp < b_end; bp++)
            {
                const int32_t col = b_col_idx[bp];
                const int32_t chunk = col >> RVSP_MAGNUS_CHUNK_LOG2;
                const int32_t dst = cursor[chunk]++;
                bin_col[dst] = col;
                bin_val[dst] = a_val * b_values[bp];
            }
        }

        /* Accumulate one chunk at a time into the L2-resident accumulator,
         * then drain into C in the sorted order symbolic_fill produced. */
        const int32_t c_start = c_row_ptr[row];
        const int32_t c_end = c_row_ptr[row + 1];
        int32_t ci = c_start;

        for (int32_t j = 0; j < n_chunks; j++)
        {
            const int32_t lo = offsets[j];
            const int32_t hi = offsets[j + 1];
            if (lo == hi)
            {
                continue;
            }

            const int32_t base = j << RVSP_MAGNUS_CHUNK_LOG2;

            /* Scatter this chunk's products into the local accumulator. */
            for (int32_t p = lo; p < hi; p++)
            {
                chunk_acc[bin_col[p] - base] += bin_val[p];
            }

            /* Drain the C columns that fall in this chunk. c_col_idx is sorted,
             * so the columns for chunk j are a contiguous run starting at ci. */
            while (ci < c_end &&
                   (c_col_idx[ci] >> RVSP_MAGNUS_CHUNK_LOG2) == j)
            {
                const int32_t local = c_col_idx[ci] - base;
                c_values[ci] = chunk_acc[local];
                chunk_acc[local] = 0.0f;
                ci++;
            }
        }
    }

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = c_row_ptr[a_rows];

    return RVSP_SUCCESS;
}
