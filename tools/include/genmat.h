/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Random sparse matrix generation directly into CSR form.
 */

#ifndef GENMAT_H
#define GENMAT_H

/*
 * genmat: generates random sparse matrices directly into CSR form.
 *
 * Usage:
 *   genmat_params_t p = genmat_default_params(rows, cols);
 *   p.density = 0.01;          // override whatever you need
 *   p.is_symmetric = 1;
 *
 *   csr_matrix_t m = genmat_generate_csr(p);
 *   if (m.nrows < 0) { handle error }
 *   ... use m.row_ptr / m.col_idx / m.values ...
 *   genmat_free_csr(&m);
 */

typedef struct {
  int nrows;
  int ncols;
  long long nnz;
  int *row_ptr;   // size nrows + 1
  int *col_idx;   // size nnz, ascending within each row
  double *values; // size nnz, values[k] corresponds to col_idx[k]
} csr_matrix_t;

typedef struct {
  int row_cnt;
  int col_cnt;
  double density;    // nonzero ratio, default 0.001
  double cv;         // coefficient of variation of row degrees, default 0.5
  int min;           // minimum nonzeros per row, default 1
  double imbalance;  // (max-avg)/avg, default -1.0 means "unset, use ncols-1"
  int is_symmetric;  // 1 = force symmetric (row_cnt must equal col_cnt)
  int is_column;     // 1 = degree distribution is over columns, not rows
  int low_bandwidth; // default row_cnt - 1 (no banding)
  int up_bandwidth;  // default row_cnt - 1 (no banding, ignored if symmetric)
  int random_seed;   // default 1
} genmat_params_t;

// Fills in all the defaults the original CLI used, for the given shape.
genmat_params_t genmat_default_params(int row_cnt, int col_cnt);

// Generates the matrix and returns it directly as CSR. On error (currently
// only "symmetric requested but not square"), returns a struct with
// nrows == -1 and all pointers NULL; check this before using the result.
csr_matrix_t genmat_generate_csr(genmat_params_t params);

// save CSR to file in Matrix Market format
int genmat_save_csr_to_mtx(const csr_matrix_t *csr, const char *filename);

// Frees row_ptr/col_idx/values and zeroes the struct.
void genmat_free_csr(csr_matrix_t *m);

#endif // GENMAT_H
