# rv-sparse API Reference

C API for CSR SpGEMM, `C = A × B`, using FP32. Include `rv_sparse.h` and link
against the library.

There are two ways to compute SpGEMM. `rvsp_spgemm_csr` is a simple one-shot
interface that allocates the output. The descriptor API separates structure
analysis from computation, so the analysis can be reused when only the values
of A and B change.

All input matrices must use canonical CSR format. This means `row_ptr[0] == 0`,
`row_ptr` is non-decreasing, column indices are in range, sorted within each
row, and there are no duplicate columns in a row. The compute functions assume
these conditions and do not check them.

## Contents

- [Types](#types)
- [Matrix lifecycle](#matrix-lifecycle)
- [One-shot SpGEMM](#one-shot-spgemm)
- [Descriptor SpGEMM](#descriptor-spgemm)
- [Status codes](#status-codes)

## Types

### rvsp_csr_matrix_t

A CSR matrix. `values` is a `void *` so the same structure can be used with
different data types. For FP32 matrices it points to `float` values.

```c
int32_t  rows, cols, nnz;
int32_t *row_ptr;   /* length rows + 1 */
int32_t *col_idx;   /* length nnz      */
void    *values;    /* length nnz      */
rvsp_dtype_t dtype;
int owns_data;      /* set when the library allocated the arrays */
```

### rvsp_spgemm_algo_t

This selects which kernel the descriptor API runs.

```c
RVSP_SPGEMM_ALGO_DEFAULT    scalar
RVSP_SPGEMM_ALGO_RVV_M1     indexed gather and scatter, LMUL=1
RVSP_SPGEMM_ALGO_RVV_M2     indexed gather and scatter, LMUL=2
RVSP_SPGEMM_ALGO_RVV_M4     indexed gather and scatter, LMUL=4
RVSP_SPGEMM_ALGO_OMP        scalar, row loop over OpenMP threads
```

`RVSP_SPGEMM_ALGO_RVV` is an alias for `RVSP_SPGEMM_ALGO_RVV_M2`.

The RVV algorithms are compiled only into a vector build, and the OpenMP one
only when the library was built with `RVSP_OPENMP=ON`. Selecting an algorithm
the build does not contain returns `RVSP_ERROR_UNSUPPORTED_BACKEND`.

The OpenMP algorithm needs one accumulator per thread, so
`rvsp_spgemm_work_estimation` reports a larger workspace for it. Set
`OMP_NUM_THREADS` before the first call, because the thread count is read
during estimation.

Every algorithm produces the same output structure. Values can differ because
the vector kernels fuse the multiply into the accumulate and the scalar kernel
may not. On a well-conditioned product the difference is in the last bits; on
one whose intermediate products cancel heavily, the two results can diverge
without bound, and neither is meaningful in FP32.

## Matrix lifecycle

### rvsp_csr_create

```c
rvsp_status_t rvsp_csr_create(
    rvsp_csr_matrix_t *A,
    int32_t rows, int32_t cols, int32_t nnz,
    int32_t *row_ptr, int32_t *col_idx, void *values,
    rvsp_dtype_t dtype);
```

Initializes a matrix descriptor using arrays provided by the caller. It does
not allocate or copy any data.

`owns_data` is left unset, so `rvsp_csr_destroy` will not free these arrays.

* `A` `[out]` matrix descriptor to initialize
* `rows`, `cols`, `nnz` `[in]` matrix dimensions
* `row_ptr`, `col_idx`, `values` `[in]` CSR arrays owned by the caller
* `dtype` `[in]` element type. FP32 is the only supported value

Returns `RVSP_SUCCESS` on success, or `RVSP_ERROR_NULL_POINTER` or
`RVSP_ERROR_INVALID_ARGUMENT` on failure.

### rvsp_csr_validate

```c
rvsp_status_t rvsp_csr_validate(const rvsp_csr_matrix_t *A);
```

Checks the basic structure of a CSR matrix. It checks that the arrays are not
null, the dimensions are valid, `row_ptr[0] == 0`,
`row_ptr[rows] == nnz`, `row_ptr` is non-decreasing, and column indices are
in range.

It does not check whether column indices are sorted or whether a row contains
duplicate columns.

* `A` `[in]` matrix to validate

Returns `RVSP_SUCCESS` if the matrix is valid or an error status otherwise.

### rvsp_csr_destroy

```c
void rvsp_csr_destroy(rvsp_csr_matrix_t *A);
```

Resets the descriptor and frees the CSR arrays when `owns_data` is set.

For matrices created with `rvsp_csr_create`, the arrays are not freed. For
matrices returned by `rvsp_spgemm_csr`, the library owns the arrays and frees
them.

Passing NULL is allowed.

## One-shot SpGEMM

### rvsp_spgemm_csr

```c
rvsp_status_t rvsp_spgemm_csr(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C,
    const rvsp_spgemm_options_t *options);
```

Computes `C = A × B` and allocates the arrays for C.

The function sets `owns_data` for C, so release C with `rvsp_csr_destroy`.

* `A`, `B` `[in]` input matrices in canonical CSR format using FP32
* `C` `[out]` output matrix allocated by the library
* `options` `[in]` backend and data type options

The available backends are `RVSP_BACKEND_SCALAR` and
`RVSP_BACKEND_RVV_INTRINSICS`. `RVSP_BACKEND_RVV_INTRINSICS` runs the LMUL=2
kernel; use the descriptor API to select a different width.

FP32 is the only supported data type.

Returns `RVSP_SUCCESS` on success. It returns `RVSP_ERROR_UNSUPPORTED_DTYPE`
when either data type is not FP32, `RVSP_ERROR_UNSUPPORTED_BACKEND` when the
requested backend is not in this build, or `RVSP_ERROR_INVALID_CSR` when an
input matrix is not in canonical CSR format.

## Descriptor SpGEMM

The descriptor API separates structure analysis from computation.

The usual sequence is to create a descriptor, select an algorithm, estimate
the work, allocate C and the workspace, and then compute.

The descriptor can be reused when the sparsity pattern of A and B stays the
same.

### rvsp_spgemm_descr_create

```c
rvsp_status_t rvsp_spgemm_descr_create(rvsp_spgemm_descr_t *descr);
```

Creates a SpGEMM descriptor.

* `descr` `[out]` descriptor handle

### rvsp_spgemm_descr_destroy

```c
void rvsp_spgemm_descr_destroy(rvsp_spgemm_descr_t descr);
```

Destroys the descriptor and releases its internal resources.

Passing NULL is allowed.

### rvsp_spgemm_set_algo

```c
rvsp_status_t rvsp_spgemm_set_algo(rvsp_spgemm_descr_t descr,
                                   rvsp_spgemm_algo_t algo);
```

Selects which kernel the descriptor runs.

Call this before `rvsp_spgemm_work_estimation` because the selected algorithm
is part of the analysis and determines the workspace size.

* `descr` `[in]` descriptor
* `algo` `[in]` algorithm

Returns `RVSP_SUCCESS` on success.

It returns `RVSP_ERROR_INVALID_ARGUMENT` if called after work estimation, or
`RVSP_ERROR_UNSUPPORTED_BACKEND` if the algorithm is not in this build.

### rvsp_spgemm_work_estimation

```c
rvsp_status_t rvsp_spgemm_work_estimation(rvsp_spgemm_descr_t descr,
                                          const rvsp_csr_matrix_t *A,
                                          const rvsp_csr_matrix_t *B,
                                          size_t *workspace_bytes,
                                          int32_t *c_nnz_out);
```

Analyzes A and B and determines the structure of C.

Calling it again replaces the previous structure stored in the descriptor.

* `descr` `[in]` descriptor
* `A`, `B` `[in]` input matrices in canonical CSR format
* `workspace_bytes` `[out]` workspace required by `rvsp_spgemm_compute`
* `c_nnz_out` `[out]` exact number of nonzeros in C

Returns `RVSP_SUCCESS` on success or an error status otherwise.

### rvsp_spgemm_compute

```c
rvsp_status_t rvsp_spgemm_compute(rvsp_spgemm_descr_t descr,
                                  const rvsp_csr_matrix_t *A,
                                  const rvsp_csr_matrix_t *B,
                                  rvsp_csr_matrix_t *C,
                                  void *workspace);
```

Computes `C = A × B` using the structure from
`rvsp_spgemm_work_estimation`.

The caller provides storage for C. The `row_ptr`, `col_idx`, and `values`
arrays must be large enough for `c_nnz_out` nonzeros.

The caller also provides a workspace with at least `workspace_bytes` bytes.

When A and B have the same sparsity pattern, repeated calls can reuse the
previous analysis.

* `descr` `[in]` descriptor with completed work estimation
* `A`, `B` `[in]` input matrices
* `C` `[in,out]` result matrix with caller allocated storage
* `workspace` `[in]` scratch space

Returns `RVSP_SUCCESS` on success or an error status otherwise.

### rvsp_spgemm_get_op_counts

```c
rvsp_status_t rvsp_spgemm_get_op_counts(rvsp_spgemm_descr_t descr,
                                        int64_t *op_counts_out, int32_t n);
```

Returns the number of intermediate products for each output row from the last
work estimation.

The values remain valid until the next call to
`rvsp_spgemm_work_estimation`.

* `descr` `[in]` descriptor
* `op_counts_out` `[out]` array receiving the per-row counts
* `n` `[in]` length of `op_counts_out`. Must be at least the number of
  analyzed rows

### rvsp_spgemm_invalidate_structure

```c
void rvsp_spgemm_invalidate_structure(rvsp_spgemm_descr_t descr);
```

Discards the cached output structure while keeping the rest of the analysis.

The next call to `rvsp_spgemm_compute` will regenerate the column indices.

* `descr` `[in]` descriptor

## Status codes

* `RVSP_SUCCESS` means the operation completed successfully
* `RVSP_ERROR_NULL_POINTER` means a required pointer was NULL
* `RVSP_ERROR_INVALID_ARGUMENT` means an argument was invalid or the API was
  used in the wrong order
* `RVSP_ERROR_UNSUPPORTED_BACKEND` means the requested backend or algorithm is
  not in this build
* `RVSP_ERROR_UNSUPPORTED_DTYPE` means the data type is not FP32
* `RVSP_ERROR_INVALID_CSR` means the input matrix does not satisfy the required
  CSR format
* `RVSP_ERROR_ALLOCATION_FAILED` means a memory allocation failed

Use `rvsp_status_to_string` for a human readable description of a status code.
