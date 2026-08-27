# rv-sparse

Sparse linear algebra for RISC-V. rv-sparse computes CSR sparse matrix-matrix
multiplication (SpGEMM), `C = A × B`, in FP32.

Five kernels share one Gustavson driver and differ only in how they accumulate.
A scalar kernel, a hand-written RISC-V Vector (RVV) kernel using indexed
gather and scatter at LMUL 1, 2 and 4, and a scalar kernel with the row loop
parallelised over OpenMP threads.

## Build

CMake is the only build system.

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

Select the target with `RVSP_ARCH_FLAGS`:

```bash
cmake -S . -B build -DRVSP_ARCH_FLAGS="-march=rv64gcv -mabi=lp64d"  # scalar and RVV
cmake -S . -B build -DRVSP_ARCH_FLAGS="-march=rv64gc -mabi=lp64d"   # scalar only
```

The RVV kernels are compiled only when that string enables the vector
extension, so a scalar build contains no vector code.

| Option | Default | Meaning |
| --- | --- | --- |
| `RVSP_ARCH_FLAGS` | `-march=native` | target ISA flags |
| `RVSP_OPENMP` | `OFF` | build the OpenMP kernel |
| `RVSP_LTO` | `ON` | link-time optimisation in release builds |
| `RVSP_BUILD_TESTS` | `ON` | build and register the tests |
| `RVSP_BUILD_EXAMPLES` | `ON` | build the examples |

## Usage

```c
#include "rv_sparse.h"

rvsp_csr_matrix_t A, B, C;

rvsp_csr_create(
    &A,
    M, K, A_nnz,
    A_row_ptr,
    A_col_idx,
    A_values,
    RVSP_DTYPE_FP32
);

rvsp_csr_create(
    &B,
    K, N, B_nnz,
    B_row_ptr,
    B_col_idx,
    B_values,
    RVSP_DTYPE_FP32
);

rvsp_spgemm_options_t options = {
    .backend = RVSP_BACKEND_SCALAR,
    .input_dtype = RVSP_DTYPE_FP32,
    .output_dtype = RVSP_DTYPE_FP32,
};

rvsp_spgemm_csr(&A, &B, &C, &options);

rvsp_csr_destroy(&C);
```

`rvsp_spgemm_csr` allocates the output matrix C. The caller is responsible for
releasing it with `rvsp_csr_destroy`.

For repeated multiplies with the same sparsity pattern, use the descriptor API
to reuse the structure analysis. It also selects which kernel runs, through
`rvsp_spgemm_set_algo`.

## Documentation

* [docs/api_design.md](docs/api_design.md) for the API reference
* [docs/directory_structure.md](docs/directory_structure.md) for the layout
* [examples/](examples/) for worked examples
* [bench/README.md](bench/README.md) for the benchmark harness

## License

GPL-3.0-only. See [LICENSE](LICENSE).
