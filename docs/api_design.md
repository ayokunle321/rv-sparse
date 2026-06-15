# rv-sparse API Design

## Purpose

The initial `rv-sparse` API provides a clean C interface for sparse matrix operations while keeping the low-level kernel implementations separated from the public interface.

The current implementation focuses on CSR sparse matrix-matrix multiplication using a scalar FP32 baseline kernel.

## Current Supported Operation

The currently implemented operation is:

```text
C = A * B
```

where `A`, `B`, and `C` are sparse matrices stored in Compressed Sparse Row (CSR) format.

## Current Supported Data Type

The implemented data type combination is:

```text
FP32 x FP32 -> FP32
```

This means that:

* `A.values` must contain `float` values.
* `B.values` must contain `float` values.
* `C.values` is allocated and returned as `float` values.

## Current Supported Backend

The currently implemented backend is:

```text
RVSP_BACKEND_SCALAR
```

The API defines additional backend identifiers for future work, but only the scalar FP32 path is currently implemented in the dispatcher.

## Architecture

The library currently follows a layered architecture composed of four main components.

### 1. Public API layer

The public API is exposed through:

```text
include/rv_sparse.h
include/rv_sparse_types.h
```

This layer defines the public matrix descriptors, status codes, data types, backend options, and user-facing functions.

Users interact with the library through structs such as:

```c
rvsp_csr_matrix_t
rvsp_spgemm_options_t
```

### 2. Dispatch layer

The dispatch layer is implemented in:

```text
src/core/spgemm.c
```

This layer selects the correct implementation according to:

* selected backend
* input data type
* output data type

Currently, the dispatcher supports the scalar FP32 CSR SpGEMM path.

### 3. Internal wrapper layer

The wrapper layer is implemented in:

```text
src/kernels/spgemm/csr_spgemm_wrappers.c
```

This layer connects the public API with the raw kernel implementation.

The wrapper is responsible for:

* validating input pointers
* validating CSR matrices
* checking matrix dimensions
* checking data types
* casting `void *values` to the correct typed pointer
* calling the raw pointer kernel
* filling the output `rvsp_csr_matrix_t`

### 4. Raw kernel layer

The raw kernel is implemented in:

```text
src/kernels/spgemm/csr_scalar_f32.c
```

This layer contains the actual computational kernel.

The current raw kernel uses plain pointers and dimensions instead of public structs. This design keeps kernels simple, portable, and easier to optimize in future RISC-V implementations.

## Main Public Types

### `rvsp_csr_matrix_t`

Represents a sparse matrix in CSR format.

Main fields:

```c
int32_t rows;
int32_t cols;
int32_t nnz;

int32_t *row_ptr;
int32_t *col_idx;
void    *values;

rvsp_dtype_t dtype;
rvsp_format_t format;

int owns_data;
```

CSR layout:

```text
row_ptr size: rows + 1
col_idx size: nnz
values  size: nnz
```

The `values` field is stored as `void *` so the same matrix descriptor can be reused for different data types. In the current implementation, FP32 matrices use `float *` values.

### `rvsp_spgemm_options_t`

Controls the backend and data type selection for SpGEMM.

```c
typedef struct {
    rvsp_backend_t backend;
    rvsp_dtype_t input_dtype;
    rvsp_dtype_t output_dtype;
    int sort_output_indices;
} rvsp_spgemm_options_t;
```

Current supported configuration:

```c
rvsp_spgemm_options_t options = {
    .backend = RVSP_BACKEND_SCALAR,
    .input_dtype = RVSP_DTYPE_FP32,
    .output_dtype = RVSP_DTYPE_FP32,
    .sort_output_indices = 1
};
```

The current scalar FP32 implementation writes output column indices in increasing order because it scans the temporary accumulator from column `0` to `B->cols - 1`.

## Public Functions

### `rvsp_csr_create()`

Initializes a CSR matrix descriptor.

```c
rvsp_status_t rvsp_csr_create(
    rvsp_csr_matrix_t *A,
    int32_t rows,
    int32_t cols,
    int32_t nnz,
    int32_t *row_ptr,
    int32_t *col_idx,
    void *values,
    rvsp_dtype_t dtype
);
```

This function does not allocate matrix data. It only initializes the descriptor using arrays provided by the caller.

### `rvsp_csr_validate()`

Validates the structure of a CSR matrix.

```c
rvsp_status_t rvsp_csr_validate(const rvsp_csr_matrix_t *A);
```

The validation checks:

* non-null matrix pointer
* non-null `row_ptr`, `col_idx`, and `values`
* valid dimensions
* `row_ptr[0] == 0`
* `row_ptr[rows] == nnz`
* non-decreasing `row_ptr`
* column indices within matrix bounds

### `rvsp_csr_destroy()`

Resets a CSR matrix descriptor and frees owned memory when applicable.

```c
void rvsp_csr_destroy(rvsp_csr_matrix_t *A);
```

Ownership behavior:

* If `owns_data == 0`, the arrays are not freed.
* If `owns_data == 1`, `row_ptr`, `col_idx`, and `values` are freed.

Matrices created with `rvsp_csr_create()` do not own their input arrays.

Matrices returned by `rvsp_spgemm_csr()` currently own their output arrays.

### `rvsp_spgemm_csr()`

Computes sparse matrix-matrix multiplication in CSR format.

```c
rvsp_status_t rvsp_spgemm_csr(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C,
    const rvsp_spgemm_options_t *options
);
```

Current supported operation:

```text
CSR FP32 x CSR FP32 -> CSR FP32
```

Current supported backend:

```text
RVSP_BACKEND_SCALAR
```

If an unsupported backend or data type combination is requested, the function returns an error status.

## Ownership Model

The current ownership model is intentionally simple.

Input matrices:

* The caller owns `row_ptr`, `col_idx`, and `values`.
* `rvsp_csr_create()` does not copy or free those arrays.
* Input matrix descriptors have `owns_data = 0`.

Output matrix:

* `rvsp_spgemm_csr()` allocates the CSR arrays for `C`.
* The output matrix has `owns_data = 1`.
* The caller must release the output matrix with `rvsp_csr_destroy(&C)`.

## Current Example

The current public API example is located at:

```text
examples/spgemm_csr_f32.c
```

It demonstrates how to:

* create CSR matrix descriptors
* configure scalar FP32 SpGEMM options
* call `rvsp_spgemm_csr()`
* print the resulting CSR matrix
* destroy matrix descriptors

## Current Test

The current correctness test is located at:

```text
tests/test_spgemm_csr_f32.c
```

It verifies that the scalar FP32 CSR SpGEMM implementation produces the expected CSR output.

## Current Design Rule

The public API should remain stable as new kernels are added.

New implementations should be added internally by providing:

1. a raw pointer kernel
2. an internal wrapper
3. a dispatcher branch
4. an example
5. a correctness test

User code should continue to call the same public function:

```c
rvsp_spgemm_csr(&A, &B, &C, &options);
```
