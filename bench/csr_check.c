/*
 * csr_check.c — canonical CSR precondition check for the SpGEMM kernels.
 *
 * Checks each input matrix for canonical CSR and verifies that matrices used
 * with --mtx-sq have cols <= rows to avoid out-of-bounds accesses in A*A.
 *
 * Build:
 *   cc -O2 -Iinclude -Itools/include -Isrc/kernels/spgemm/v2 \
 *     bench/csr_check.c obj/<tag>/tools/[all].o -Llib/<tag> -lrvsparse -lm \
 *     -o bench/csr_check
 *
 * Usage:
 *   ./bench/csr_check matrices/star/star.mtx ...
 *   ./bench/csr_check matrices/ast/ast.mtx
 *
 * Exit status: 0 if every matrix is canonical and valid for A*A, 1 otherwise.
 */

#include "rv_sparse.h"
#include "mtx_to_csr_formatter.h"
#include "rvsp_v2.h"
#include "vec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Strip directories and the .mtx suffix for the output table. */
static void short_name(const char *path, char *out, size_t outsz) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    snprintf(out, outsz, "%s", base);

    char *dot = strrchr(out, '.');
    if (dot && strcmp(dot, ".mtx") == 0)
        *dot = '\0';
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s M1.mtx [M2.mtx ...]\n"
                "checks canonical CSR and A*A compatibility\n",
                argv[0]);
        return 2;
    }

    printf("%-20s %10s %10s %12s  %s\n",
           "matrix", "rows", "cols(inf)", "nnz", "status");
    printf("%-20s %10s %10s %12s  %s\n",
           "--------------------", "----------", "----------",
           "------------", "------");

    int failures = 0;
    int loaded = 0;

    for (int i = 1; i < argc; i++) {
        char name[128];
        short_name(argv[i], name, sizeof(name));

        struct CSR p = assemble_csr_matrix(argv[i]);

        if (!p.row_ptr || !p.col_ind) {
            printf("%-20s %10s %10s %12s  LOAD FAILED (%s)\n",
                   name, "-", "-", "-", argv[i]);

            failures++;
            vector_free(p.row_ptr);
            vector_free(p.col_ind);
            vector_free(p.val);
            continue;
        }

        loaded++;

        int32_t rows = (int32_t)vector_size(p.row_ptr) - 1;
        int32_t nnz  = (int32_t)vector_size(p.col_ind);

        const int32_t *row_ptr =
            (const int32_t *)p.row_ptr->data;
        const int32_t *col_idx =
            (const int32_t *)p.col_ind->data;

        /* Match bench.c's column inference. */
        int32_t cols = 0;
        for (int32_t k = 0; k < nnz; k++)
            if (col_idx[k] + 1 > cols)
                cols = col_idx[k] + 1;

        int32_t bad_row = -1;

        rvsp_csr_status_t st =
            rvsp_csr_check(rows, cols, row_ptr, col_idx, &bad_row);

        char detail[256];

        if (st == RVSP_CSR_OK) {
            snprintf(detail, sizeof(detail), "OK  (%s)",
                     rvsp_csr_status_string(st));
        } else if (bad_row >= 0) {
            snprintf(detail, sizeof(detail),
                     "FAIL %s (first bad row %d)",
                     rvsp_csr_status_string(st), bad_row);
            failures++;
        } else {
            snprintf(detail, sizeof(detail),
                     "FAIL %s",
                     rvsp_csr_status_string(st));
            failures++;
        }

        printf("%-20s %10d %10d %12d  %s\n",
               name, rows, cols, nnz, detail);

        /*
         * A*A uses each column index of A as a row index of B.
         * With B = A, cols > rows would access row_ptr past the end.
         */
        if (cols > rows) {
            printf("%-20s %10s %10s %12s  "
                   "FAIL not square for --mtx-sq: cols(%d) > rows(%d)\n",
                   "", "", "", "", cols, rows);
            failures++;
        }

        vector_free(p.row_ptr);
        vector_free(p.col_ind);
        vector_free(p.val);
    }

    printf("\n%d matrix/matrices loaded, %d problem(s) found.\n",
           loaded, failures);

    if (failures) {
        printf("\nInput validation failed. Fix the matrix loader/converter "
               "before benchmarking.\n");
    }

    return failures ? 1 : 0;
}