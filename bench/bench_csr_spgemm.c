/*
 * bench_spgemm.c
 *
 * Unified benchmark harness for rv-sparse raw CSR SpGEMM kernels.
 *
 * One source, three binaries. Select the kernel at compile time:
 *
 *   -DKERNEL_F32       -> rvsp_spgemm_csr_scalar_f32_raw     
 *   -DKERNEL_UNROLL4   -> rvsp_spgemm_csr_scalar_unroll4_f32_raw 
 *   -DKERNEL_I8        -> rvsp_spgemm_csr_scalar_i8_raw
 *
 * Build (scalar baseline), from repo root:
 *   riscv64-linux-gnu-gcc -O3 -march=rv64gc -static -Iinclude -DKERNEL_F32 \
 *       bench/bench_spgemm.c src/kernels/spgemm/csr_scalar_f32.c -o bench_f32
 *   riscv64-linux-gnu-gcc -O3 -march=rv64gc -static -Iinclude -DKERNEL_UNROLL4 \
 *       bench/bench_spgemm.c src/kernels/spgemm/csr_scalar_unroll4_f32.c -o bench_unroll4
 *   riscv64-linux-gnu-gcc -O3 -march=rv64gc -static -Iinclude -DKERNEL_I8 \
 *       bench/bench_spgemm.c src/kernels/spgemm/csr_scalar_i8.c -o bench_i8
 *
 * Run:
 *   qemu-riscv64 ./bench_f32     [M K N density reps]
 *   qemu-riscv64 ./bench_unroll4 [M K N density reps]
 *   qemu-riscv64 ./bench_i8      [M K N density reps]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "rv_sparse_types.h"

/* ------------------------------------------------------------------ */
/* Per-kernel configuration                                            */
/* ------------------------------------------------------------------ */
#if defined(KERNEL_F32)
  #define KERNEL_NAME       "csr_scalar_f32"
  typedef float   val_t;            /* input value type  */
  typedef float   out_t;            /* output value type */
  #define VAL_IS_FLOAT      1
  #define BYTES_PER_MADD    16.0    /* B idx(4) + B val(4) + acc rmw(8) */
  #define OP_LABEL          "flops"
  #define INTENSITY_LABEL   "FLOP/byte"
  rvsp_status_t rvsp_spgemm_csr_scalar_f32_raw(
      int32_t,int32_t,int32_t,
      const int32_t*,const int32_t*,const float*,
      const int32_t*,const int32_t*,const float*,
      int32_t**,int32_t**,float**,int32_t*);
  #define CALL_KERNEL(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz) \
      rvsp_spgemm_csr_scalar_f32_raw(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz)

#elif defined(KERNEL_UNROLL4)
  #define KERNEL_NAME       "csr_scalar_unroll4_f32"
  typedef float   val_t;
  typedef float   out_t;
  #define VAL_IS_FLOAT      1
  #define BYTES_PER_MADD    16.0
  #define OP_LABEL          "flops"
  #define INTENSITY_LABEL   "FLOP/byte"
  rvsp_status_t rvsp_spgemm_csr_scalar_unroll4_f32_raw(
      int32_t,int32_t,int32_t,
      const int32_t*,const int32_t*,const float*,
      const int32_t*,const int32_t*,const float*,
      int32_t**,int32_t**,float**,int32_t*);
  #define CALL_KERNEL(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz) \
      rvsp_spgemm_csr_scalar_unroll4_f32_raw(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz)

#elif defined(KERNEL_I8)
  #define KERNEL_NAME       "csr_scalar_i8"
  typedef int8_t  val_t;            /* int8 inputs */
  typedef int32_t out_t;            /* int32 output */
  #define VAL_IS_FLOAT      0
  /* i8 byte model: B idx(4) + B val(1) + acc int32 rmw(4+4=8) = 13 bytes */
  #define BYTES_PER_MADD    13.0
  #define OP_LABEL          "iops"
  #define INTENSITY_LABEL   "IOP/byte"
  rvsp_status_t rvsp_spgemm_csr_scalar_i8_raw(
      int32_t,int32_t,int32_t,
      const int32_t*,const int32_t*,const int8_t*,
      const int32_t*,const int32_t*,const int8_t*,
      int32_t**,int32_t**,int32_t**,int32_t*);
  #define CALL_KERNEL(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz) \
      rvsp_spgemm_csr_scalar_i8_raw(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz)

#else
  #error "Define one of KERNEL_F32, KERNEL_UNROLL4, KERNEL_I8"
#endif

/* ------------------------------------------------------------------ */
/* Synthetic CSR generator (unstructured sparsity)    */
/* ------------------------------------------------------------------ */
static int32_t gen_csr(int32_t rows, int32_t cols, double density, unsigned seed,
                       int32_t **row_ptr, int32_t **col_idx, val_t **values)
{
    srand(seed);
    int32_t cap = (int32_t)((double)rows * cols * density) + rows + 16;
    int32_t *rp = (int32_t *)malloc((size_t)(rows + 1) * sizeof(int32_t));
    int32_t *ci = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    val_t   *vv = (val_t *)malloc((size_t)cap * sizeof(val_t));
    int32_t nnz = 0;

    for (int32_t r = 0; r < rows; r++) {
        rp[r] = nnz;
        for (int32_t c = 0; c < cols; c++) {
            if ((double)rand() / (double)RAND_MAX < density) {
                if (nnz >= cap) {
                    cap *= 2;
                    ci = (int32_t *)realloc(ci, (size_t)cap * sizeof(int32_t));
                    vv = (val_t *)realloc(vv, (size_t)cap * sizeof(val_t));
                }
                ci[nnz] = c;
#if VAL_IS_FLOAT
                vv[nnz] = (val_t)(((rand() % 20) - 10) * 0.5f + 0.1f);
#else
                /* nonzero int8 in [-9, 9], avoiding 0 */
                { int t = (rand() % 19) - 9; if (t == 0) t = 1; vv[nnz] = (val_t)t; }
#endif
                nnz++;
            }
        }
    }
    rp[rows] = nnz;
    *row_ptr = rp; *col_idx = ci; *values = vv;
    return nnz;
}

int main(int argc, char **argv)
{
    int32_t M       = argc > 1 ? atoi(argv[1]) : 256;
    int32_t K       = argc > 2 ? atoi(argv[2]) : 256;
    int32_t N       = argc > 3 ? atoi(argv[3]) : 256;
    double  density = argc > 4 ? atof(argv[4]) : 0.05;
    int     reps    = argc > 5 ? atoi(argv[5]) : 20;

    int32_t *a_rp, *a_ci, *b_rp, *b_ci;
    val_t   *a_v, *b_v;
    int32_t a_nnz = gen_csr(M, K, density, 1u, &a_rp, &a_ci, &a_v);
    int32_t b_nnz = gen_csr(K, N, density, 2u, &b_rp, &b_ci, &b_v);

    fprintf(stderr,
            "KERNEL=%s  A:%dx%d nnz=%d  B:%dx%d nnz=%d  density=%.3f reps=%d\n",
            KERNEL_NAME, M, K, a_nnz, K, N, b_nnz, density, reps);

    /* Analytical work: multiply-add pairs = sum over A-nonzeros of nnz(B row k) */
    long long madds = 0;
    for (int32_t row = 0; row < M; row++)
        for (int32_t ap = a_rp[row]; ap < a_rp[row + 1]; ap++) {
            int32_t k = a_ci[ap];
            madds += (long long)(b_rp[k + 1] - b_rp[k]);
        }
    double ops   = 2.0 * (double)madds;              /* 1 mul + 1 add per madd */
    double bytes = BYTES_PER_MADD * (double)madds;
    double ai    = bytes > 0 ? ops / bytes : 0.0;

    fprintf(stderr,
            "WORK: madd_pairs=%lld  %s=%.0f  bytes=%.0f  AI=%.4f %s\n",
            madds, OP_LABEL, ops, bytes, ai, INTENSITY_LABEL);

    int32_t c_nnz = 0;
    volatile double sink = 0.0;
    for (int it = 0; it < reps; it++) {
        int32_t *c_rp = NULL, *c_ci = NULL;
        out_t   *c_v = NULL;
        rvsp_status_t st = CALL_KERNEL(M, K, N,
            a_rp, a_ci, a_v, b_rp, b_ci, b_v,
            &c_rp, &c_ci, &c_v, &c_nnz);
        if (st != RVSP_SUCCESS) {
            fprintf(stderr, "kernel error %d on iter %d\n", (int)st, it);
            return 1;
        }
        if (c_nnz > 0) sink += (double)c_v[0] + (double)c_v[c_nnz - 1];
        free(c_rp); free(c_ci); free(c_v);
    }

    fprintf(stderr, "C nnz=%d  sink=%.4f  (done)\n", c_nnz, (double)sink);
    free(a_rp); free(a_ci); free(a_v);
    free(b_rp); free(b_ci); free(b_v);
    return 0;
}