# Contributing to rv-sparse

Thanks for your interest in rv-sparse. Pull requests, bug reports,
documentation, and new kernels are all welcome.

## Issues

Use the GitHub issue tracker for bugs and feature requests. Search existing
issues first. If an issue already covers your case, add details there instead
of opening a duplicate.

For anything larger than a bug fix, open an issue to discuss the approach
before writing code. This helps make sure the change fits the project before
implementation starts.

## Workflow

Fork the repository and work on a branch. Do not commit directly to `main`.

```bash
git checkout -b my-change
# work, commit
git push -u origin my-change
```

Open a pull request against `main`. Keep each pull request focused on a single
fix or feature. Avoid reformatting or refactoring unrelated code in the same
change.

The pull request description should explain what the change does, how it was
tested, and whether it changes the public API.

## Building and testing

Before opening a pull request, build and run the tests:

```bash
make
make test
```

For a RISC-V vector build, set `TARGET_ARCH=riscv`. This uses the
`riscv64-linux-gnu` cross-toolchain, enables the vector extension, and runs the
tests under `qemu-riscv64-static`. Physical hardware is not required.

```bash
make test TARGET_ARCH=riscv
```

Other build options include:

```bash
make BUILD_TYPE=debug    # unoptimized build with debug symbols
make OPENMP=1            # enable OpenMP
make print-config        # print the resolved toolchain and flags
```

Tests are discovered automatically. A new `.c` file at the top level of
`tests/` is picked up without any Makefile changes. Tests in `tests/legacy/`
cover superseded kernels and are not run.

See [docs/directory_structure.md](docs/directory_structure.md) for the
repository layout.

A change that does not build or breaks a test will not be merged. If you fix a
bug, add a test that fails before the fix and passes after it.

## Adding a strategy

The SpGEMM strategies shipped today share one Gustavson driver and differ only
in how they accumulate. Adding another strategy that follows this design uses
the same pattern:

1. Add `src/kernels/spgemm/accum_<name>_f32.c`. Define `RVSP_KERNEL_NAME` and
   include `gustavson_core_f32.inc`. The file provides only `rvsp_accum_row`.
   The driver and symbolic phase are shared.
2. Declare `rvsp_spgemm_<name>_f32` and
   `rvsp_spgemm_<name>_f32_numeric` in
   `src/kernels/spgemm/rvsp_spgemm.h`. The `.inc` file emits both functions,
   and `numeric_for()` needs the second one declared.
3. Add the strategy to `rvsp_spgemm_algo_t` in
   `include/rv_sparse_types.h`.
4. Add the case to `numeric_for()` in `src/core/spgemm_descr.c`. Guard it with
   `__riscv_vector` if it uses vector intrinsics.
5. Add the source to `VECTOR_SRCS` in the Makefile if it must not be compiled
   for scalar targets.

The symbolic phase is shared, so a strategy following this design changes how
values are accumulated but does not change the output structure.

If a contribution needs a different dataflow or a different approach to
accumulation, open an issue first so we can decide how it should fit into the
library.

## Input contract

The compute kernels assume canonical CSR input. This means `row_ptr[0]` is 0,
`row_ptr` is non-decreasing, column indices are in range, column indices are
sorted within each row, and no column repeats within a row.

The kernels do not check these conditions in the hot loop. Validate or
canonicalize the input once before calling a kernel, and keep that work outside
any timed region.

## Vector code

Vector kernels require the RISC-V V extension. If the extension is not
available, the kernel should fail to compile rather than silently falling back
to scalar code.

```c
#if defined(__riscv_vector)
#include <riscv_vector.h>
#else
#error "accum_<name>_f32.c requires the RISC-V V extension"
#endif
```

The build system controls whether vector sources are compiled. Add the source
to `VECTOR_SRCS` so the Makefile excludes it when `-march` does not enable V.

For code that uses RVV intrinsics, add comments explaining the vector type,
the memory access pattern, and any assumptions about index validity. Follow
the style of the existing kernels.

## Public API

`include/rv_sparse.h` and `include/rv_sparse_types.h` are the public API.

Keep the public API stable. Discuss any API change in an issue before
implementing it.

## License

rv-sparse is licensed under GPL-3.0-only, and contributions are accepted under
the same license.

Add the SPDX header to new files:

```c
/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 */
```
