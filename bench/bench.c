/*
 * SpGEMM benchmark harness for rv-sparse.
 *
 * Benchmarks a single kernel on one A × B multiplication, validates the
 * result against the scalar reference implementation, and prints one CSV
 * row containing the timing results for each run.
 *
 * Usage:
 *   ./bench --kernel NAME
 *           (--gen R C DENSITY SEED | --mtx A B | --mtx-sq M)
 *           [--runs N] [--warmup W]
 *           [--label TAG] [--header]
 */

#define _POSIX_C_SOURCE 199309L   /* for clock_gettime / CLOCK_MONOTONIC under -std=c11 */
#define _GNU_SOURCE               /* for syscall() used by perf_event_open */

#include "rv_sparse.h"
#include "genmat.h"
#include "mtx_to_csr_formatter.h"
#include "vec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* Optional hardware counters via perf_event_open, enabled with -DUSE_PERF.
 * Counts only the kernel call. Needs perf_event_paranoid <= 2. Degrades
 * to -1 if the counters can't be opened. */
#ifdef USE_PERF
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <errno.h>

static int perf_fd_cycles = -1;
static int perf_fd_insns  = -1;

static int perf_open_one(uint32_t type, uint64_t config, int group_fd) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type           = type;
    pe.size           = sizeof(pe);
    pe.config         = config;
    pe.disabled       = 1;
    pe.exclude_kernel = 1;   /* user-space only, matches perf's ":u", works at paranoid=2 */
    pe.exclude_hv     = 1;
    long fd = syscall(__NR_perf_event_open, &pe, 0 /*this thread*/, -1 /*any cpu*/, group_fd, 0);
    return (int)fd;
}

static void perf_init(void) {
    perf_fd_cycles = perf_open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1);
    perf_fd_insns  = perf_open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1);
    if (perf_fd_cycles < 0 || perf_fd_insns < 0)
        fprintf(stderr, "warning: perf counters unavailable (%s); "
                        "cycles/instructions will be blank\n", strerror(errno));
}

static void perf_start(void) {
    if (perf_fd_cycles >= 0) {
        ioctl(perf_fd_cycles, PERF_EVENT_IOC_RESET, 0);
        ioctl(perf_fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    }
    if (perf_fd_insns >= 0) {
        ioctl(perf_fd_insns, PERF_EVENT_IOC_RESET, 0);
        ioctl(perf_fd_insns, PERF_EVENT_IOC_ENABLE, 0);
    }
}

static void perf_stop(long long *cycles, long long *insns) {
    *cycles = -1; *insns = -1;
    if (perf_fd_cycles >= 0) {
        ioctl(perf_fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
        if (read(perf_fd_cycles, cycles, sizeof(*cycles)) != sizeof(*cycles)) *cycles = -1;
    }
    if (perf_fd_insns >= 0) {
        ioctl(perf_fd_insns, PERF_EVENT_IOC_DISABLE, 0);
        if (read(perf_fd_insns, insns, sizeof(*insns)) != sizeof(*insns)) *insns = -1;
    }
}

static void perf_close(void) {
    if (perf_fd_cycles >= 0) close(perf_fd_cycles);
    if (perf_fd_insns  >= 0) close(perf_fd_insns);
}
#else
static void perf_init(void) {}
static void perf_start(void) {}
static void perf_stop(long long *cycles, long long *insns) { *cycles = -1; *insns = -1; }
static void perf_close(void) {}
#endif

/* kernel registry, uniform signature, iterate over these */
typedef rvsp_status_t (*spgemm_fn)(const rvsp_csr_matrix_t *,
                                   const rvsp_csr_matrix_t *,
                                   rvsp_csr_matrix_t *);

/* declared in csr_spgemm_kernels.h, but declare here so bench.c is self-contained */
rvsp_status_t rvsp_spgemm_csr_scalar_f32(const rvsp_csr_matrix_t*, const rvsp_csr_matrix_t*, rvsp_csr_matrix_t*);
rvsp_status_t rvsp_spgemm_csr_scalar_f64(const rvsp_csr_matrix_t*, const rvsp_csr_matrix_t*, rvsp_csr_matrix_t*);
rvsp_status_t rvsp_spgemm_csr_scalar_i8(const rvsp_csr_matrix_t*, const rvsp_csr_matrix_t*, rvsp_csr_matrix_t*);
rvsp_status_t rvsp_spgemm_csr_scalar_unroll4_f32(const rvsp_csr_matrix_t*, const rvsp_csr_matrix_t*, rvsp_csr_matrix_t*);
rvsp_status_t rvsp_spgemm_csr_rvv_f32_indexed_marked(const rvsp_csr_matrix_t*, const rvsp_csr_matrix_t*, rvsp_csr_matrix_t*);
rvsp_status_t rvsp_spgemm_csr_rvv_f64_indexed_marked(const rvsp_csr_matrix_t*, const rvsp_csr_matrix_t*, rvsp_csr_matrix_t*);
rvsp_status_t rvsp_spgemm_csr_rvv_i8_indexed_marked(const rvsp_csr_matrix_t*, const rvsp_csr_matrix_t*, rvsp_csr_matrix_t*);

typedef struct {
    const char   *name;      /* cli name, e.g. "rvv_f32" */
    spgemm_fn     fn;
    rvsp_dtype_t  dtype;     /* input dtype */
    const char   *dtype_str; /* "f32"/"f64"/"i8" for csv */
    spgemm_fn     ref;       /* scalar reference of same dtype for validation */
    const char   *ref_name;
} kernel_entry_t;

/* reference fns filled in below */
static const kernel_entry_t KERNELS[] = {
    { "scalar_f32",  rvsp_spgemm_csr_scalar_f32,          RVSP_DTYPE_FP32, "f32",
      rvsp_spgemm_csr_scalar_f32, "scalar_f32" },
    { "scalar_f64",  rvsp_spgemm_csr_scalar_f64,          RVSP_DTYPE_FP64, "f64",
      rvsp_spgemm_csr_scalar_f64, "scalar_f64" },
    { "scalar_i8",   rvsp_spgemm_csr_scalar_i8,           RVSP_DTYPE_INT8, "i8",
      rvsp_spgemm_csr_scalar_i8,  "scalar_i8" },
    { "unroll4_f32", rvsp_spgemm_csr_scalar_unroll4_f32,  RVSP_DTYPE_FP32, "f32",
      rvsp_spgemm_csr_scalar_f32, "scalar_f32" },
    { "rvv_f32",     rvsp_spgemm_csr_rvv_f32_indexed_marked, RVSP_DTYPE_FP32, "f32",
      rvsp_spgemm_csr_scalar_f32, "scalar_f32" },
    { "rvv_f64",     rvsp_spgemm_csr_rvv_f64_indexed_marked, RVSP_DTYPE_FP64, "f64",
      rvsp_spgemm_csr_scalar_f64, "scalar_f64" },
    { "rvv_i8",      rvsp_spgemm_csr_rvv_i8_indexed_marked,  RVSP_DTYPE_INT8, "i8",
      rvsp_spgemm_csr_scalar_i8,  "scalar_i8" },
};
static const int N_KERNELS = (int)(sizeof(KERNELS)/sizeof(KERNELS[0]));

static const kernel_entry_t *find_kernel(const char *name) {
    for (int i = 0; i < N_KERNELS; i++)
        if (strcmp(KERNELS[i].name, name) == 0) return &KERNELS[i];
    return NULL;
}

/* timing */
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* op count is 2 * intermediate products. For C = A*B, work is the sum over
 * each nonzero (i,k) of A of nnz(row k of B). */
static double intermediate_products(const rvsp_csr_matrix_t *A,
                                    const rvsp_csr_matrix_t *B) {
    double prods = 0.0;
    for (int32_t i = 0; i < A->rows; i++) {
        for (int32_t p = A->row_ptr[i]; p < A->row_ptr[i+1]; p++) {
            int32_t k = A->col_idx[p];       /* A column = B row */
            if (k >= 0 && k < B->rows)
                prods += (double)(B->row_ptr[k+1] - B->row_ptr[k]);
        }
    }
    return prods;
}

/* Validation runs after the timed region, so it never touches the hot path.
 * Order independent within a row, row_ptr must match but columns are sorted
 * before comparing, so a kernel is not forced to sort inside the timed loop
 * just to pass a positional check. Values use a combined relative and
 * absolute tolerance. */
typedef struct { int32_t col; int32_t pos; } colref_t;

static int cmp_colref(const void *a, const void *b) {
    int32_t ca = ((const colref_t*)a)->col;
    int32_t cb = ((const colref_t*)b)->col;
    return (ca > cb) - (ca < cb);
}

static int vals_close(const void *xv, const void *yv,
                      int32_t px, int32_t py, rvsp_dtype_t dtype) {
    if (dtype == RVSP_DTYPE_FP32) {
        float a = ((const float*)xv)[px], b = ((const float*)yv)[py];
        return fabsf(a - b) <= 1e-5f + 1e-4f * fabsf(b);
    } else if (dtype == RVSP_DTYPE_FP64) {
        double a = ((const double*)xv)[px], b = ((const double*)yv)[py];
        return fabs(a - b) <= 1e-12 + 1e-9 * fabs(b);
    } else { /* i8 -> i32 output, exact */
        int32_t a = ((const int32_t*)xv)[px], b = ((const int32_t*)yv)[py];
        return a == b;
    }
}

static int csr_equal(const rvsp_csr_matrix_t *X, const rvsp_csr_matrix_t *Y,
                     rvsp_dtype_t dtype) {
    if (X->rows != Y->rows || X->cols != Y->cols || X->nnz != Y->nnz) return 0;
    for (int32_t i = 0; i <= X->rows; i++)
        if (X->row_ptr[i] != Y->row_ptr[i]) return 0;   /* same per-row counts */

    int32_t maxlen = 0;
    for (int32_t i = 0; i < X->rows; i++) {
        int32_t len = X->row_ptr[i+1] - X->row_ptr[i];
        if (len > maxlen) maxlen = len;
    }
    if (maxlen == 0) return 1;

    colref_t *rx = malloc((size_t)maxlen * sizeof(colref_t));
    colref_t *ry = malloc((size_t)maxlen * sizeof(colref_t));
    if (!rx || !ry) { free(rx); free(ry); return 0; }  /* OOM, fail closed */

    int ok = 1;
    for (int32_t i = 0; i < X->rows && ok; i++) {
        int32_t sx = X->row_ptr[i], sy = Y->row_ptr[i];
        int32_t len = X->row_ptr[i+1] - sx;
        for (int32_t t = 0; t < len; t++) {
            rx[t].col = X->col_idx[sx+t]; rx[t].pos = sx+t;
            ry[t].col = Y->col_idx[sy+t]; ry[t].pos = sy+t;
        }
        qsort(rx, (size_t)len, sizeof(colref_t), cmp_colref);
        qsort(ry, (size_t)len, sizeof(colref_t), cmp_colref);
        for (int32_t t = 0; t < len; t++) {
            if (rx[t].col != ry[t].col) { ok = 0; break; }
            if (!vals_close(X->values, Y->values, rx[t].pos, ry[t].pos, dtype)) { ok = 0; break; }
        }
    }
    free(rx); free(ry);
    return ok;
}

/* matrix acquisition, all produce int32 row_ptr/col_idx and typed values
 * wrapped in rvsp_csr_matrix_t. owns_data stays 0, we free the raw arrays
 * ourselves. */

/* holds the raw arrays we allocate so we can free them after */
typedef struct {
    int32_t *row_ptr;
    int32_t *col_idx;
    void    *values;
    int      free_via_genmat;   /* if set, values came from genmat (double*) we converted */
} raw_csr_t;

/* Convert a genmat csr_matrix_t (double values, int arrays) into typed rvsp arrays. */
static int build_from_genmat(csr_matrix_t *g, rvsp_dtype_t dtype,
                             rvsp_csr_matrix_t *out, raw_csr_t *raw) {
    int32_t rows = (int32_t)g->nrows;
    int32_t cols = (int32_t)g->ncols;
    int32_t nnz  = (int32_t)g->nnz;

    raw->row_ptr = malloc((size_t)(rows+1) * sizeof(int32_t));
    raw->col_idx = malloc((size_t)nnz * sizeof(int32_t));
    if (!raw->row_ptr || !raw->col_idx) return -1;

    for (int32_t i = 0; i <= rows; i++) raw->row_ptr[i] = (int32_t)g->row_ptr[i];
    for (int32_t p = 0; p < nnz; p++)   raw->col_idx[p] = (int32_t)g->col_idx[p];

    if (dtype == RVSP_DTYPE_FP32) {
        float *v = malloc((size_t)nnz * sizeof(float));
        if (!v) return -1;
        for (int32_t p = 0; p < nnz; p++) v[p] = (float)g->values[p];
        raw->values = v;
    } else if (dtype == RVSP_DTYPE_FP64) {
        double *v = malloc((size_t)nnz * sizeof(double));
        if (!v) return -1;
        for (int32_t p = 0; p < nnz; p++) v[p] = g->values[p];
        raw->values = v;
    } else { /* i8, clamp generated values into int8 range */
        int8_t *v = malloc((size_t)nnz * sizeof(int8_t));
        if (!v) return -1;
        for (int32_t p = 0; p < nnz; p++) {
            double d = g->values[p];
            int iv = (int)d % 127; if (iv == 0) iv = 1;
            v[p] = (int8_t)iv;
        }
        raw->values = v;
    }
    return rvsp_csr_create(out, rows, cols, nnz,
                           raw->row_ptr, raw->col_idx, raw->values, dtype);
}

/* Convert an assemble_csr_matrix() struct CSR into typed rvsp arrays. */
static int build_from_mtx(struct CSR *parsed, rvsp_dtype_t dtype,
                          rvsp_csr_matrix_t *out, raw_csr_t *raw) {
    int32_t rows = (int32_t)vector_size(parsed->row_ptr) - 1;
    int32_t nnz  = (int32_t)vector_size(parsed->col_ind);
    /* infer cols from max col index + 1 */
    const int32_t *cidx = (const int32_t*)parsed->col_ind->data;
    int32_t cols = 0;
    for (int32_t p = 0; p < nnz; p++) if (cidx[p]+1 > cols) cols = cidx[p]+1;

    raw->row_ptr = malloc((size_t)(rows+1) * sizeof(int32_t));
    raw->col_idx = malloc((size_t)nnz * sizeof(int32_t));
    if (!raw->row_ptr || !raw->col_idx) return -1;
    memcpy(raw->row_ptr, parsed->row_ptr->data, (size_t)(rows+1)*sizeof(int32_t));
    memcpy(raw->col_idx, parsed->col_ind->data, (size_t)nnz*sizeof(int32_t));

    const float *fv = (const float*)parsed->val->data;
    if (dtype == RVSP_DTYPE_FP32) {
        float *v = malloc((size_t)nnz * sizeof(float));
        if (!v) return -1;
        memcpy(v, fv, (size_t)nnz*sizeof(float));
        raw->values = v;
    } else if (dtype == RVSP_DTYPE_FP64) {
        double *v = malloc((size_t)nnz * sizeof(double));
        if (!v) return -1;
        for (int32_t p = 0; p < nnz; p++) v[p] = (double)fv[p];
        raw->values = v;
    } else {
        int8_t *v = malloc((size_t)nnz * sizeof(int8_t));
        if (!v) return -1;
        for (int32_t p = 0; p < nnz; p++) {
            int iv = (int)fv[p] % 127; if (iv == 0) iv = 1;
            v[p] = (int8_t)iv;
        }
        raw->values = v;
    }
    return rvsp_csr_create(out, rows, cols, nnz,
                           raw->row_ptr, raw->col_idx, raw->values, dtype);
}

static void free_raw(raw_csr_t *raw) {
    free(raw->row_ptr); free(raw->col_idx); free(raw->values);
    raw->row_ptr = NULL; raw->col_idx = NULL; raw->values = NULL;
}

/* main */
static void usage(const char *prog) {
    fprintf(stderr,
      "usage: %s --kernel NAME (--gen R C DENSITY SEED | --mtx A.mtx B.mtx | --mtx-sq M.mtx)\n"
      "          [--runs N] [--warmup W] [--label TAG] [--header]\n"
      "kernels: scalar_f32 scalar_f64 scalar_i8 unroll4_f32 rvv_f32 rvv_f64 rvv_i8\n", prog);
}

int main(int argc, char **argv) {
    const char *kernel_name = NULL;
    const char *label = "-";
    int runs = 10, warmup = 3, header = 0;

    enum { SRC_NONE, SRC_GEN, SRC_MTX, SRC_MTXSQ } src = SRC_NONE;
    int gen_r=0, gen_c=0, gen_seed=0; double gen_density=0.0;
    const char *mtx_a=NULL, *mtx_b=NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--kernel") && i+1<argc) kernel_name = argv[++i];
        else if (!strcmp(argv[i],"--runs") && i+1<argc) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--warmup") && i+1<argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--label") && i+1<argc) label = argv[++i];
        else if (!strcmp(argv[i],"--header")) header = 1;
        else if (!strcmp(argv[i],"--gen") && i+4<argc) {
            src = SRC_GEN; gen_r=atoi(argv[++i]); gen_c=atoi(argv[++i]);
            gen_density=atof(argv[++i]); gen_seed=atoi(argv[++i]);
        }
        else if (!strcmp(argv[i],"--mtx") && i+2<argc) {
            src = SRC_MTX; mtx_a=argv[++i]; mtx_b=argv[++i];
        }
        else if (!strcmp(argv[i],"--mtx-sq") && i+1<argc) {
            src = SRC_MTXSQ; mtx_a=argv[++i];
        }
        else { fprintf(stderr,"unknown/incomplete arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }

    if (header)
        printf("label,kernel,dtype,src,rows,cols,nnz_a,nnz_b,nnz_c,flops,run,time_s,gops,correct,cycles,instructions\n");

    if (!kernel_name || src == SRC_NONE) { usage(argv[0]); return 2; }
    const kernel_entry_t *K = find_kernel(kernel_name);
    if (!K) { fprintf(stderr,"no such kernel: %s\n", kernel_name); return 2; }

    /* build A and B, untimed */
    rvsp_csr_matrix_t A={0}, B={0};
    raw_csr_t rawA={0}, rawB={0};
    csr_matrix_t gA={0}, gB={0};
    struct CSR pA={0}, pB={0};
    const char *src_str = "?";

    if (src == SRC_GEN) {
        src_str = "gen";
        genmat_params_t p = genmat_default_params(gen_r, gen_c);
        p.density = gen_density; p.random_seed = gen_seed; p.cv = 0.5; p.min = 1;
        gA = genmat_generate_csr(p);
        genmat_params_t p2 = genmat_default_params(gen_c, gen_r); /* B is cols x rows so A*B is valid */
        p2.density = gen_density; p2.random_seed = gen_seed+1; p2.cv = 0.5; p2.min = 1;
        gB = genmat_generate_csr(p2);
        if (gA.nrows < 0 || gB.nrows < 0) { fprintf(stderr,"genmat failed\n"); return 1; }
        if (build_from_genmat(&gA, K->dtype, &A, &rawA) != RVSP_SUCCESS ||
            build_from_genmat(&gB, K->dtype, &B, &rawB) != RVSP_SUCCESS) {
            fprintf(stderr,"build_from_genmat failed\n"); return 1; }
    } else if (src == SRC_MTX) {
        src_str = "mtx";
        pA = assemble_csr_matrix(mtx_a);
        pB = assemble_csr_matrix(mtx_b);
        if (!pA.row_ptr || !pB.row_ptr) { fprintf(stderr,"mtx load failed\n"); return 1; }
        if (build_from_mtx(&pA, K->dtype, &A, &rawA) != RVSP_SUCCESS ||
            build_from_mtx(&pB, K->dtype, &B, &rawB) != RVSP_SUCCESS) {
            fprintf(stderr,"build_from_mtx failed\n"); return 1; }
    } else { /* SRC_MTXSQ, C = A*A */
        src_str = "mtx_sq";
        pA = assemble_csr_matrix(mtx_a);
        if (!pA.row_ptr) { fprintf(stderr,"mtx load failed\n"); return 1; }
        if (build_from_mtx(&pA, K->dtype, &A, &rawA) != RVSP_SUCCESS) {
            fprintf(stderr,"build_from_mtx failed\n"); return 1; }
        B = A; /* square, B aliases A, read only in kernels */
    }

    if (rvsp_csr_validate(&A) != RVSP_SUCCESS ||
        rvsp_csr_validate(&B) != RVSP_SUCCESS) {
        fprintf(stderr,"csr validate failed (A or B malformed)\n"); return 1;
    }

    double flops = 2.0 * intermediate_products(&A, &B);

    /* reference result for validation, untimed */
    rvsp_csr_matrix_t Cref = {0};
    int have_ref = (K->ref(&A, &B, &Cref) == RVSP_SUCCESS);

    /* warmup, untimed */
    for (int w = 0; w < warmup; w++) {
        rvsp_csr_matrix_t Cw = {0};
        if (K->fn(&A, &B, &Cw) == RVSP_SUCCESS) rvsp_csr_destroy(&Cw);
    }

    /* timed runs */
    perf_init();
    for (int r = 0; r < runs; r++) {
        rvsp_csr_matrix_t C = {0};
        long long cycles = -1, insns = -1;

        perf_start();
        double t0 = now_seconds();
        rvsp_status_t st = K->fn(&A, &B, &C);
        double t1 = now_seconds();
        perf_stop(&cycles, &insns);


        double dt = t1 - t0;
        int correct = 0;
        int32_t nnz_c = 0;
        if (st == RVSP_SUCCESS) {
            nnz_c = C.nnz;
            /* validation is outside the timed region above */
            correct = have_ref ? csr_equal(&C, &Cref, K->dtype) : -1;
        }
        double gops = (dt > 0.0) ? (flops / dt) / 1e9 : 0.0;

        printf("%s,%s,%s,%s,%d,%d,%d,%d,%d,%.0f,%d,%.9f,%.4f,%d,",
               label, K->name, K->dtype_str, src_str,
               A.rows, A.cols, A.nnz, B.nnz, nnz_c,
               flops, r, dt, gops, correct);
        /* blank rather than -1 when counters unavailable, so analyze.py skips them */
        if (cycles >= 0) printf("%lld,", cycles); else printf(",");
        if (insns  >= 0) printf("%lld\n", insns); else printf("\n");
        fflush(stdout);

        if (st == RVSP_SUCCESS) rvsp_csr_destroy(&C);
    }
    perf_close();

    /* teardown */
    if (have_ref) rvsp_csr_destroy(&Cref);
    rvsp_csr_destroy(&A);
    if (src != SRC_MTXSQ) rvsp_csr_destroy(&B);  /* don't double-free aliased square B */
    free_raw(&rawA);
    if (src != SRC_MTXSQ) free_raw(&rawB);

    if (src == SRC_GEN) { genmat_free_csr(&gA); genmat_free_csr(&gB); }
    if (src == SRC_MTX || src == SRC_MTXSQ) {
        vector_free(pA.row_ptr); vector_free(pA.col_ind); vector_free(pA.val);
        if (src == SRC_MTX) { vector_free(pB.row_ptr); vector_free(pB.col_ind); vector_free(pB.val); }
    }
    return 0;
}
