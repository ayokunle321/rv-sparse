/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Workspace size query and canonical CSR validation.
 */

#include "rvsp_v2_common.h"

rvsp_status_t
rvsp_spgemm_buffer_size(
    int32_t b_cols,
    size_t *bytes_out)
{
    if (bytes_out == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    *bytes_out = rvsp_ws_bytes(b_cols);

    return RVSP_SUCCESS;
}

rvsp_csr_status_t
rvsp_csr_check(
    int32_t rows,
    int32_t cols,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    int32_t *bad_row_out)
{
    if (bad_row_out != NULL)
    {
        *bad_row_out = -1;
    }

    if (rows < 0 || cols < 0 || row_ptr == NULL)
    {
        return RVSP_CSR_BAD_ROW_PTR;
    }

    if (row_ptr[0] != 0)
    {
        return RVSP_CSR_BAD_ROW_PTR;
    }

    for (int32_t row = 0; row < rows; row++)
    {
        const int32_t start = row_ptr[row];
        const int32_t end = row_ptr[row + 1];

        if (end < start)
        {
            if (bad_row_out != NULL)
            {
                *bad_row_out = row;
            }

            return RVSP_CSR_BAD_ROW_PTR;
        }

        if (end == start)
        {
            continue;
        }

        if (col_idx == NULL)
        {
            return RVSP_CSR_BAD_ROW_PTR;
        }

        int32_t prev = -1;

        for (int32_t p = start; p < end; p++)
        {
            const int32_t col = col_idx[p];

            if (col < 0 || col >= cols)
            {
                if (bad_row_out != NULL)
                {
                    *bad_row_out = row;
                }

                return RVSP_CSR_COL_OUT_OF_RANGE;
            }

            if (col == prev)
            {
                if (bad_row_out != NULL)
                {
                    *bad_row_out = row;
                }

                return RVSP_CSR_DUPLICATE;
            }

            if (col < prev)
            {
                if (bad_row_out != NULL)
                {
                    *bad_row_out = row;
                }

                return RVSP_CSR_UNSORTED;
            }

            prev = col;
        }
    }

    return RVSP_CSR_OK;
}

const char *
rvsp_csr_status_string(rvsp_csr_status_t status)
{
    switch (status)
    {
    case RVSP_CSR_OK:
        return "canonical CSR";

    case RVSP_CSR_BAD_ROW_PTR:
        return "malformed row pointer";

    case RVSP_CSR_COL_OUT_OF_RANGE:
        return "column index out of range";

    case RVSP_CSR_UNSORTED:
        return "column indices not ascending";

    case RVSP_CSR_DUPLICATE:
        return "duplicate column in row";

    default:
        return "unknown";
    }
}