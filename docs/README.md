# rv-sparse Documentation

This directory contains the core design documentation, architectural decisions, and API guidelines for the `rv-sparse` project.

## Current Focus

The active development and documentation efforts are currently centered around the following areas:

| Focus Area | Status | Comments / Next Steps |
| :--- | :--- | :--- |
| **Public API Design** | Work in Progress | Defining a stable and predictable interface for users (`rv_sparse.h`). |
| **Directory Structure** | Proposed | Scalable layout established (OpenBLAS/MKL style). |
| **CSR SpGEMM Interface** | Work in Progress | Standardizing the initial baseline kernels for sparse matrix multiplication. |
| **Backend Selection Model** | Pending | Designing the dispatch mechanism (Scalar vs. GCC auto-vectorized vs. RVV). |
| **Project Timeline** | Planning | Mapping out upcoming milestones and feature integrations. |

## Library Architecture

The proposed library structure ensures a strict separation of concerns. Below is the visual representation of the architecture followed by its detailed components:

```text
+-------------------------------------------------------------+
|                   rv-sparse Architecture                    |
+-------------------------------------------------------------+
|                                                             |
|  [rv-sparse/]                                               |
|   |                                                         |
|   +-- [include/]    <-- Public API Headers                  |
|   |                                                         |
|   +-- [src/]                                                |
|   |    +-- [core/]     <-- Logic, Context & Error Handling  |
|   |    +-- [formats/]  <-- Data Structures (CSR, COO)       |
|   |    `-- [kernels/]  <-- Compute Engines (Scalar, RVV)    |
|   |                                                         |
|   +-- [benchmarks/] <-- Performance Measurement             |
|   +-- [examples/]   <-- API Usage Samples                   |
|   +-- [tests/]      <-- Correctness Validation              |
|   `-- [docs/]       <-- Architecture Specs & Guidelines     |
|                                                             |
+-------------------------------------------------------------+
```
