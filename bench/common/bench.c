/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * SpGEMM benchmark harness.
 *
 * Builds A and B, validates the selected kernel, runs warmups and timed
 * iterations, and emits one CSV row per run. Setup and validation are untimed.
 *
 * Usage:
 *   ./bench --kernel NAME \
 *     (--gen R C DENSITY SEED | --mtx A.mtx B.mtx | --mtx-sq M.mtx) \
 *     [--runs N] [--warmup W] [--label TAG] [--header] \
 *     [--arm ARM] [--build TAG] [--march FLAGS] [--cflags FLAGS] \
 *     [--cc-version VER] [--pinning DESC] [--git-commit REV]
 *     [--verify-pinning]
 *
 * Input modes:
 *   --gen R C DENSITY SEED   Generate synthetic CSR matrices.
 *   --mtx A.mtx B.mtx       Load two Matrix Market matrices.
 *   --mtx-sq M.mtx          Benchmark M * M.
 *
 * Build metadata is supplied by the caller and recorded in the CSV so
 * results from different binaries can share the same dataset. --pinning
 * describes how the caller meant to pin the process and its threads;
 * --verify-pinning makes the harness check that intent against what the OS
 * and OpenMP actually did, and refuse to run if they differ.
 *
 * Exit status 3 means a precondition for a trustworthy result failed: input
 * not canonical CSR, no reference, the reference failing its float64 check,
 * or pinning not as intended. Drivers stop the sweep on it.
 */

#define _POSIX_C_SOURCE 199309L /* clock_gettime / CLOCK_MONOTONIC */
#define _GNU_SOURCE             /* syscall() for perf_event_open */

#include "genmat.h"
#include "mtx_to_csr_formatter.h"
#include "rv_sparse.h"
#include "rvsp_spgemm.h"
#include "vec.h"

#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Exit status for a failed precondition; drivers stop the sweep on it. */
#define EXIT_PRECONDITION 3

/*
 * Optional hardware counters. When enabled, counters cover only the
 * kernel call and are left blank when unavailable. perf_note says which, so a
 * blank counter column is never ambiguous.
 */
static char perf_note[160] = "off";
#ifdef USE_PERF
#include <errno.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

/*
 * Three counters on a PMU with few slots will usually be multiplexed, so each
 * is opened with TOTAL_TIME_ENABLED and TOTAL_TIME_RUNNING. A caller scales a
 * raw count back to a full-window estimate with enabled/running; when the two
 * are equal the counter ran the whole time and no scaling is needed.
 */
struct perf_read {
  uint64_t value;
  uint64_t time_enabled;
  uint64_t time_running;
};

static int perf_fd_cycles = -1;
static int perf_fd_insns = -1;
static int perf_fd_stalls = -1;

static int perf_open_one(uint32_t type, uint64_t config, int group_fd) {
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(pe));
  pe.type = type;
  pe.size = sizeof(pe);
  pe.config = config;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  pe.read_format =
      PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

  long fd = syscall(__NR_perf_event_open, &pe, 0, -1, group_fd, 0);
  return (int)fd;
}

static void perf_init(void) {
  perf_fd_cycles =
      perf_open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1);
  perf_fd_insns =
      perf_open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1);
  perf_fd_stalls = perf_open_one(PERF_TYPE_HARDWARE,
                                 PERF_COUNT_HW_STALLED_CYCLES_BACKEND, -1);

  const int err = errno;

  if (perf_fd_cycles < 0 || perf_fd_insns < 0)
    fprintf(stderr,
            "warning: perf counters unavailable (%s); "
            "cycles/instructions will be blank\n",
            strerror(err));

  if (perf_fd_stalls < 0)
    fprintf(stderr, "warning: stalled-cycles-backend unavailable on this PMU; "
                    "the column will be blank\n");

  if (perf_fd_cycles < 0 || perf_fd_insns < 0) {
    int paranoid = -99;
    FILE *f = fopen("/proc/sys/kernel/perf_event_paranoid", "r");

    if (f) {
      if (fscanf(f, "%d", &paranoid) != 1)
        paranoid = -99;
      fclose(f);
    }

    if (paranoid != -99)
      snprintf(perf_note, sizeof(perf_note),
               "unavailable: %s (perf_event_paranoid=%d)", strerror(err),
               paranoid);
    else
      snprintf(perf_note, sizeof(perf_note), "unavailable: %s",
               strerror(err));
  } else if (perf_fd_stalls < 0) {
    snprintf(perf_note, sizeof(perf_note),
             "partial: stalls_backend unsupported by PMU");
  } else {
    snprintf(perf_note, sizeof(perf_note), "ok");
  }

  /* The note goes into a CSV field. */
  for (char *c = perf_note; *c; c++)
    if (*c == ',')
      *c = ';';
}

static void perf_enable_one(int fd) {
  if (fd >= 0) {
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  }
}

static void perf_start(void) {
  perf_enable_one(perf_fd_cycles);
  perf_enable_one(perf_fd_insns);
  perf_enable_one(perf_fd_stalls);
}

/* Returns the raw count, and reports the counter's enabled and running time
 * so the caller can tell whether it was multiplexed. */
static long long perf_read_one(int fd, uint64_t *enabled, uint64_t *running) {
  if (fd < 0)
    return -1;

  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

  struct perf_read pr = {0};

  if (read(fd, &pr, sizeof(pr)) != (ssize_t)sizeof(pr))
    return -1;

  if (enabled)
    *enabled = pr.time_enabled;

  if (running)
    *running = pr.time_running;

  return (long long)pr.value;
}

static void perf_stop(long long *cycles, long long *insns, long long *stalls,
                      double *mux) {
  uint64_t enabled = 0;
  uint64_t running = 0;

  *cycles = perf_read_one(perf_fd_cycles, NULL, NULL);
  *insns = perf_read_one(perf_fd_insns, NULL, NULL);
  *stalls = perf_read_one(perf_fd_stalls, &enabled, &running);

  /* Fraction of the measured window the stall counter was actually on. */
  *mux = (enabled > 0) ? (double)running / (double)enabled : -1.0;
}

static void perf_close(void) {
  if (perf_fd_cycles >= 0)
    close(perf_fd_cycles);
  if (perf_fd_insns >= 0)
    close(perf_fd_insns);
  if (perf_fd_stalls >= 0)
    close(perf_fd_stalls);
}
#else
static void perf_init(void) {}
static void perf_start(void) {}
static void perf_stop(long long *cycles, long long *insns, long long *stalls,
                      double *mux) {
  *cycles = -1;
  *insns = -1;
  *stalls = -1;
  *mux = -1.0;
}
static void perf_close(void) {}
#endif

/* Uniform harness signature; adapters hide kernel-specific arguments. */
typedef rvsp_status_t (*spgemm_fn)(const rvsp_csr_matrix_t *A,
                                   const rvsp_csr_matrix_t *B,
                                   rvsp_csr_matrix_t *C, void *workspace,
                                   size_t workspace_bytes,
                                   int64_t *op_counts_out);

/* Kernel declarations used by the benchmark registry. */
/*
 * Adapt the raw kernel API to rvsp_csr_matrix_t.
 * Workspace is supplied by the harness and reused across runs.
 */
#define KERNEL_WRAP(wrapper, kernel)                                           \
  static rvsp_status_t wrapper(                                                \
      const rvsp_csr_matrix_t *A, const rvsp_csr_matrix_t *B,                  \
      rvsp_csr_matrix_t *C, void *ws, size_t wsb, int64_t *ops) {              \
    int32_t *c_row_ptr = NULL, *c_col_idx = NULL, c_nnz = 0;                   \
    float *c_values = NULL;                                                    \
    rvsp_status_t st =                                                         \
        kernel(A->rows, A->cols, B->cols, A->row_ptr, A->col_idx,              \
               (const float *)A->values, B->row_ptr, B->col_idx,               \
               (const float *)B->values, ws, wsb, &c_row_ptr, &c_col_idx,      \
               &c_values, &c_nnz, ops);                                        \
    if (st != RVSP_SUCCESS)                                                    \
      return st;                                                               \
    C->rows = A->rows;                                                         \
    C->cols = B->cols;                                                         \
    C->nnz = c_nnz;                                                            \
    C->row_ptr = c_row_ptr;                                                    \
    C->col_idx = c_col_idx;                                                    \
    C->values = c_values;                                                      \
    C->dtype = RVSP_DTYPE_FP32;                                                \
    C->format = RVSP_FORMAT_CSR;                                               \
    C->owns_data = 1;                                                          \
    return RVSP_SUCCESS;                                                       \
  }

KERNEL_WRAP(scalar_f32_w, rvsp_spgemm_scalar_f32)

#if defined(__riscv_vector)
KERNEL_WRAP(rvv_f32_m1_w, rvsp_spgemm_rvv_f32_m1)
KERNEL_WRAP(rvv_f32_m2_w, rvsp_spgemm_rvv_f32_m2)
KERNEL_WRAP(rvv_f32_m4_w, rvsp_spgemm_rvv_f32_m4)
#endif

#if defined(__AVX512F__)
KERNEL_WRAP(avx512_f32_w, rvsp_spgemm_avx512_f32)
#endif

typedef struct {
  const char *name;
  spgemm_fn fn;
  rvsp_spgemm_algo_t algo;
  int lmul; /* 0 when the kernel is not RVV */
  rvsp_dtype_t dtype;
  const char *dtype_str;
  spgemm_fn ref;
  const char *ref_name;
} kernel_entry_t;

/* Each current kernel is validated against the matching scalar reference. */
static const kernel_entry_t KERNELS[] = {
    {"scalar_f32", scalar_f32_w, RVSP_SPGEMM_ALGO_DEFAULT, 0, RVSP_DTYPE_FP32,
     "f32", scalar_f32_w, "scalar_f32"},

#if defined(__riscv_vector)
    {"rvv_f32_m1", rvv_f32_m1_w, RVSP_SPGEMM_ALGO_RVV_M1, 1, RVSP_DTYPE_FP32,
     "f32", scalar_f32_w, "scalar_f32"},
    {"rvv_f32_m2", rvv_f32_m2_w, RVSP_SPGEMM_ALGO_RVV_M2, 2, RVSP_DTYPE_FP32,
     "f32", scalar_f32_w, "scalar_f32"},
    {"rvv_f32_m4", rvv_f32_m4_w, RVSP_SPGEMM_ALGO_RVV_M4, 4, RVSP_DTYPE_FP32,
     "f32", scalar_f32_w, "scalar_f32"},
#endif

#if defined(__AVX512F__)
    {"avx512_f32", avx512_f32_w, RVSP_SPGEMM_ALGO_AVX512, 0, RVSP_DTYPE_FP32,
     "f32", scalar_f32_w, "scalar_f32"},
#endif

#if defined(_OPENMP)
    {"scalar_omp_f32", scalar_f32_w, RVSP_SPGEMM_ALGO_OMP, 0, RVSP_DTYPE_FP32,
     "f32", scalar_f32_w, "scalar_f32"},
#endif
};

static const int N_KERNELS = (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));

static const kernel_entry_t *find_kernel(const char *name) {
  for (int i = 0; i < N_KERNELS; i++)
    if (strcmp(KERNELS[i].name, name) == 0)
      return &KERNELS[i];
  return NULL;
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Compute the intermediate products for A*B and record Op_i per output row. */
static double intermediate_products(const rvsp_csr_matrix_t *A,
                                    const rvsp_csr_matrix_t *B,
                                    int64_t *op_counts) {

  double prods = 0.0;

  for (int32_t i = 0; i < A->rows; i++) {
    int64_t row_ops = 0;

    for (int32_t p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
      int32_t k = A->col_idx[p]; /* A column = B row */

      if (k >= 0 && k < B->rows)
        row_ops += (int64_t)(B->row_ptr[k + 1] - B->row_ptr[k]);
    }

    if (op_counts)
      op_counts[i] = row_ops;

    prods += (double)row_ops;
  }

  return prods;
}

/* Mean, maximum, and population variance of Op_i over the output rows. */
static void op_stats(const int64_t *op, int32_t n, double *mean_out,
                     double *max_out, double *var_out) {

  *mean_out = 0.0;
  *max_out = 0.0;
  *var_out = 0.0;

  if (!op || n <= 0)
    return;

  double sum = 0.0;
  int64_t mx = 0;

  for (int32_t i = 0; i < n; i++) {
    sum += (double)op[i];
    if (op[i] > mx)
      mx = op[i];
  }

  const double mean = sum / (double)n;

  double acc = 0.0;
  for (int32_t i = 0; i < n; i++) {
    const double d = (double)op[i] - mean;
    acc += d * d;
  }

  *mean_out = mean;
  *max_out = (double)mx;
  *var_out = acc / (double)n;
}

/*
 * Compare CSR outputs independent of column ordering within each row.
 * Values use dtype-specific tolerances.
 */
typedef struct {
  int32_t col;
  int32_t pos;
} colref_t;

static int cmp_colref(const void *a, const void *b) {
  int32_t ca = ((const colref_t *)a)->col;
  int32_t cb = ((const colref_t *)b)->col;
  return (ca > cb) - (ca < cb);
}

static int vals_close(const void *xv, const void *yv, int32_t px, int32_t py,
                      rvsp_dtype_t dtype) {

  (void)dtype;

  float a = ((const float *)xv)[px];
  float b = ((const float *)yv)[py];

  return fabsf(a - b) <= 1e-5f + 1e-4f * fabsf(b);
}

static int csr_equal(const rvsp_csr_matrix_t *X, const rvsp_csr_matrix_t *Y,
                     rvsp_dtype_t dtype) {

  if (X->rows != Y->rows || X->cols != Y->cols || X->nnz != Y->nnz)
    return 0;

  for (int32_t i = 0; i <= X->rows; i++)
    if (X->row_ptr[i] != Y->row_ptr[i])
      return 0;

  int32_t maxlen = 0;

  for (int32_t i = 0; i < X->rows; i++) {
    int32_t len = X->row_ptr[i + 1] - X->row_ptr[i];
    if (len > maxlen)
      maxlen = len;
  }

  if (maxlen == 0)
    return 1;

  colref_t *rx = malloc((size_t)maxlen * sizeof(colref_t));
  colref_t *ry = malloc((size_t)maxlen * sizeof(colref_t));

  if (!rx || !ry) {
    free(rx);
    free(ry);
    return 0;
  }

  int ok = 1;

  for (int32_t i = 0; i < X->rows && ok; i++) {
    int32_t sx = X->row_ptr[i];
    int32_t sy = Y->row_ptr[i];
    int32_t len = X->row_ptr[i + 1] - sx;

    for (int32_t t = 0; t < len; t++) {
      rx[t].col = X->col_idx[sx + t];
      rx[t].pos = sx + t;
      ry[t].col = Y->col_idx[sy + t];
      ry[t].pos = sy + t;
    }

    qsort(rx, (size_t)len, sizeof(colref_t), cmp_colref);
    qsort(ry, (size_t)len, sizeof(colref_t), cmp_colref);

    for (int32_t t = 0; t < len; t++) {
      if (rx[t].col != ry[t].col ||
          !vals_close(X->values, Y->values, rx[t].pos, ry[t].pos, dtype)) {
        ok = 0;
        break;
      }
    }
  }

  free(rx);
  free(ry);
  return ok;
}

/*
 * Error against the reference, beyond the pass/fail tolerance above.
 *
 * Two float32 sums of the same k products, in any order and with or without
 * FMA, can each be off from the exact value by about (k+1) * 2^-24 times the
 * sum of the products' magnitudes, so they can differ by twice that. An entry
 * further apart than 4 (k+1) 2^-24 sum|a*b|, double the bound for slack, is
 * not rounding: something computed a different value. k is taken as the A
 * row length, an upper bound on the products per entry.
 *
 *   exact       every value bitwise equal
 *   rounding    values differ, all within the rounding bound
 *   value       same structure, some value beyond the rounding bound or NaN
 *   structural  nnz, row_ptr or col_idx differ
 *   failed      the kernel call returned an error
 *   noref       no reference could be built
 *
 * Relative error is |c - ref| / |ref| over entries with ref != 0; entries
 * where ref == 0 are still covered by the bound.
 */
typedef struct {
  long tol_fail;
  double max_rel;
  double mean_rel;
  const char *cls;
} err_stats_t;

/*
 * sum |a_ik * b_kj| for every entry of R, in R's order, and an independent
 * check of R itself: the same product accumulated in float64, against which R
 * must sit within the rounding bound. Without it the reference is only
 * compared with itself, since the baseline kernel and the reference are the
 * same scalar code. *ref_bad receives the number of R entries that fail.
 * NULL on allocation failure.
 */
static float *term_magnitudes(const rvsp_csr_matrix_t *A,
                              const rvsp_csr_matrix_t *B,
                              const rvsp_csr_matrix_t *R, long *ref_bad) {
  const size_t ncols = (size_t)(B->cols > 0 ? B->cols : 1);
  double *acc = calloc(ncols, sizeof(double));
  double *exact = calloc(ncols, sizeof(double));
  float *mag = malloc((size_t)(R->nnz > 0 ? R->nnz : 1) * sizeof(float));

  *ref_bad = 0;

  if (!acc || !exact || !mag) {
    free(acc);
    free(exact);
    free(mag);
    return NULL;
  }

  const float *rv = (const float *)R->values;
  const double u = ldexp(1.0, -24);

  const float *av = (const float *)A->values;
  const float *bv = (const float *)B->values;

  for (int32_t i = 0; i < A->rows; i++) {
    for (int32_t ap = A->row_ptr[i]; ap < A->row_ptr[i + 1]; ap++) {
      const int32_t k = A->col_idx[ap];
      const double a = fabs((double)av[ap]);

      const double as = (double)av[ap];

      for (int32_t bp = B->row_ptr[k]; bp < B->row_ptr[k + 1]; bp++) {
        acc[B->col_idx[bp]] += a * fabs((double)bv[bp]);
        exact[B->col_idx[bp]] += as * (double)bv[bp];
      }
    }

    const double kterms = (double)(A->row_ptr[i + 1] - A->row_ptr[i]);

    /* Every column the row touched is in R's structure, so this also
     * clears acc for the next row. */
    for (int32_t p = R->row_ptr[i]; p < R->row_ptr[i + 1]; p++) {
      const int32_t j = R->col_idx[p];
      const double d = fabs((double)rv[p] - exact[j]);

      if (isnan(rv[p]) || d > 4.0 * (kterms + 1.0) * u * acc[j])
        (*ref_bad)++;

      mag[p] = (float)acc[j];
      acc[j] = 0.0;
      exact[j] = 0.0;
    }
  }

  free(acc);
  free(exact);
  return mag;
}

/*
 * What the OS and OpenMP actually did with the threads that will run the
 * kernel. For an OpenMP kernel a probe region with the same ICVs reports the
 * team size and each thread's CPU and affinity mask; the kernel's region gets
 * the same team, and bound threads do not migrate. A serial kernel reports
 * the main thread's affinity mask.
 */
typedef struct {
  int team;
  char cpus[512]; /* "0-7" or "0+2-3", no commas */
  char problem[256];
} binding_t;

/* Ascending CPU list -> "0-3+8"; commas would split the CSV field. */
static void cpu_ranges(const int *cpu, int n, char *out, size_t len) {
  size_t pos = 0;
  out[0] = '\0';

  for (int i = 0; i < n && pos < len; i++) {
    int j = i;
    while (j + 1 < n && cpu[j + 1] == cpu[j] + 1)
      j++;

    int w = (j > i) ? snprintf(out + pos, len - pos, "%s%d-%d", pos ? "+" : "",
                               cpu[i], cpu[j])
                    : snprintf(out + pos, len - pos, "%s%d", pos ? "+" : "",
                               cpu[i]);
    if (w < 0)
      break;
    pos += (size_t)w;
    i = j;
  }
}

static int cmp_int(const void *a, const void *b) {
  int x = *(const int *)a, y = *(const int *)b;
  return (x > y) - (x < y);
}

/* 1 if cpu b is listed in cpu a's thread_siblings_list, 0 if not, -1 if the
 * topology cannot be read. */
static int smt_siblings(int a, int b) {
  char path[128];
  char buf[256];
  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", a);

  FILE *f = fopen(path, "r");
  if (!f)
    return -1;

  int ok = fgets(buf, sizeof(buf), f) != NULL;
  fclose(f);
  if (!ok)
    return -1;

  for (char *p = buf; *p;) {
    char *end;
    long lo = strtol(p, &end, 10);
    if (end == p)
      break;
    long hi = lo;
    if (*end == '-')
      hi = strtol(end + 1, &end, 10);
    if (b >= lo && b <= hi)
      return 1;
    p = (*end == ',') ? end + 1 : end;
    if (*p == '\n')
      break;
  }

  return 0;
}

static void probe_binding(int omp_kernel, int want_threads, binding_t *out) {
  int cpu[1024];
  int n = 0;
  int unbound = 0;

  out->team = 1;
  out->problem[0] = '\0';

#if defined(_OPENMP)
  if (omp_kernel) {
    int max = omp_get_max_threads();
    int *seen = calloc((size_t)(max > 0 ? max : 1), sizeof(int));
    int *masks = calloc((size_t)(max > 0 ? max : 1), sizeof(int));
    int team = 0;

    if (!seen || !masks) {
      snprintf(out->problem, sizeof(out->problem), "probe allocation failed");
      free(seen);
      free(masks);
      return;
    }

#pragma omp parallel
    {
      const int t = omp_get_thread_num();
      cpu_set_t set;
      CPU_ZERO(&set);

#pragma omp single
      team = omp_get_num_threads();

      if (t < max) {
        seen[t] = sched_getcpu();
        masks[t] = sched_getaffinity(0, sizeof(set), &set) == 0
                       ? CPU_COUNT(&set)
                       : -1;
      }
    }

    out->team = team;

    for (int t = 0; t < team && t < max && n < 1024; t++) {
      cpu[n++] = seen[t];
      if (masks[t] != 1)
        unbound++;
    }

    free(seen);
    free(masks);
  } else
#endif
  {
    cpu_set_t set;
    CPU_ZERO(&set);
    (void)omp_kernel;

    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
      for (int c = 0; c < CPU_SETSIZE && n < 1024; c++)
        if (CPU_ISSET(c, &set))
          cpu[n++] = c;
    }

    if (n != 1)
      unbound = 1;
  }

  qsort(cpu, (size_t)n, sizeof(int), cmp_int);
  cpu_ranges(cpu, n, out->cpus, sizeof(out->cpus));

  if (out->team != want_threads) {
    snprintf(out->problem, sizeof(out->problem),
             "team size %d but %d threads were requested", out->team,
             want_threads);
    return;
  }

  if (unbound) {
    snprintf(out->problem, sizeof(out->problem),
             "%s not bound to a single CPU (cpus %s)",
             omp_kernel ? "some threads are" : "the process is", out->cpus);
    return;
  }

  for (int i = 1; i < n; i++) {
    if (cpu[i] == cpu[i - 1]) {
      snprintf(out->problem, sizeof(out->problem),
               "two threads share cpu %d (cpus %s)", cpu[i], out->cpus);
      return;
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      int s = smt_siblings(cpu[i], cpu[j]);

      if (s == 1) {
        snprintf(out->problem, sizeof(out->problem),
                 "cpus %d and %d are SMT siblings of one core", cpu[i],
                 cpu[j]);
        return;
      }

      if (s < 0) {
        fprintf(stderr, "warning: cannot read SMT topology for cpu %d; "
                        "sibling check skipped\n", cpu[i]);
        i = n;
        break;
      }
    }
  }
}

static void err_stats(const rvsp_csr_matrix_t *C, const rvsp_csr_matrix_t *R,
                      const rvsp_csr_matrix_t *A, const float *mag,
                      err_stats_t *out) {
  out->tol_fail = 0;
  out->max_rel = 0.0;
  out->mean_rel = 0.0;
  out->cls = "structural";

  if (C->rows != R->rows || C->cols != R->cols || C->nnz != R->nnz ||
      memcmp(C->row_ptr, R->row_ptr, ((size_t)R->rows + 1) * sizeof(int32_t)) ||
      memcmp(C->col_idx, R->col_idx, (size_t)R->nnz * sizeof(int32_t)))
    return;

  const float *cv = (const float *)C->values;
  const float *rv = (const float *)R->values;
  const double u = ldexp(1.0, -24);

  long differ = 0;
  long beyond = 0;
  long counted = 0;
  double sum_rel = 0.0;

  for (int32_t i = 0; i < R->rows; i++) {
    const double k = (double)(A->row_ptr[i + 1] - A->row_ptr[i]);

    for (int32_t p = R->row_ptr[i]; p < R->row_ptr[i + 1]; p++) {
      const float c = cv[p];
      const float r = rv[p];

      if (memcmp(&c, &r, sizeof(float)) != 0)
        differ++;

      if (isnan(c) || isnan(r)) {
        beyond++;
        out->tol_fail++;
        continue;
      }

      if (!vals_close(cv, rv, p, p, RVSP_DTYPE_FP32))
        out->tol_fail++;

      const double d = fabs((double)c - (double)r);

      if (d > 4.0 * (k + 1.0) * u * (double)mag[p])
        beyond++;

      if (r != 0.0f) {
        const double rel = d / fabs((double)r);
        if (rel > out->max_rel)
          out->max_rel = rel;
        sum_rel += rel;
        counted++;
      }
    }
  }

  out->mean_rel = counted ? sum_rel / (double)counted : 0.0;
  out->cls = beyond ? "value" : differ ? "rounding" : "exact";
}

/* Load inputs into typed CSR storage owned by the harness. */
typedef struct {
  int32_t *row_ptr;
  int32_t *col_idx;
  void *values;
  int free_via_genmat;
} raw_csr_t;

/* Convert generated CSR values to the selected kernel dtype. */
static int build_from_genmat(csr_matrix_t *g, rvsp_dtype_t dtype,
                             rvsp_csr_matrix_t *out, raw_csr_t *raw) {

  int32_t rows = (int32_t)g->nrows;
  int32_t cols = (int32_t)g->ncols;
  int32_t nnz = (int32_t)g->nnz;

  raw->row_ptr = malloc((size_t)(rows + 1) * sizeof(int32_t));
  raw->col_idx = malloc((size_t)nnz * sizeof(int32_t));

  if (!raw->row_ptr || !raw->col_idx)
    return -1;

  for (int32_t i = 0; i <= rows; i++)
    raw->row_ptr[i] = (int32_t)g->row_ptr[i];

  for (int32_t p = 0; p < nnz; p++)
    raw->col_idx[p] = (int32_t)g->col_idx[p];

  {
    float *v = malloc((size_t)nnz * sizeof(float));
    if (!v)
      return -1;

    for (int32_t p = 0; p < nnz; p++)
      v[p] = (float)g->values[p];

    raw->values = v;
  }

  return rvsp_csr_create(out, rows, cols, nnz, raw->row_ptr, raw->col_idx,
                         raw->values, dtype);
}

/* Convert parsed Matrix Market CSR values to the selected kernel dtype. */
/*
 * square forces cols = rows, for --mtx-sq. Inferring cols from the data drops
 * empty trailing columns, which makes A*A dimensionally invalid on any matrix
 * whose last columns are unoccupied.
 */
static int build_from_mtx(struct CSR *parsed, rvsp_dtype_t dtype,
                          rvsp_csr_matrix_t *out, raw_csr_t *raw, int square) {

  int32_t rows = (int32_t)vector_size(parsed->row_ptr) - 1;
  int32_t nnz = (int32_t)vector_size(parsed->col_ind);

  const int32_t *cidx = (const int32_t *)parsed->col_ind->data;

  int32_t cols = 0;

  for (int32_t p = 0; p < nnz; p++)
    if (cidx[p] + 1 > cols)
      cols = cidx[p] + 1;

  if (square) {
    if (cols > rows) {
      fprintf(stderr,
              "--mtx-sq: matrix is not square, a column index reaches "
              "%d but there are only %d rows\n",
              cols - 1, rows);
      return -1;
    }

    cols = rows;
  }

  raw->row_ptr = malloc((size_t)(rows + 1) * sizeof(int32_t));
  raw->col_idx = malloc((size_t)nnz * sizeof(int32_t));

  if (!raw->row_ptr || !raw->col_idx)
    return -1;

  memcpy(raw->row_ptr, parsed->row_ptr->data,
         (size_t)(rows + 1) * sizeof(int32_t));

  memcpy(raw->col_idx, parsed->col_ind->data, (size_t)nnz * sizeof(int32_t));

  const float *fv = (const float *)parsed->val->data;

  {
    float *v = malloc((size_t)nnz * sizeof(float));
    if (!v)
      return -1;

    memcpy(v, fv, (size_t)nnz * sizeof(float));
    raw->values = v;
  }

  return rvsp_csr_create(out, rows, cols, nnz, raw->row_ptr, raw->col_idx,
                         raw->values, dtype);
}

static void free_raw(raw_csr_t *raw) {
  free(raw->row_ptr);
  free(raw->col_idx);
  free(raw->values);

  raw->row_ptr = NULL;
  raw->col_idx = NULL;
  raw->values = NULL;
}

#define CSV_HEADER                                                             \
  "label,kernel,arm,build,march,cflags,cc_version,dtype,"                      \
  "rows,cols,nnz_a,nnz_b,nnz_c,flops,op_mean,op_max,op_var,"                   \
  "run,time_s,gops,correct,cycles,instructions,threads,"                       \
  "stalls_backend,perf_mux,lmul,pinning,perf_note,"                            \
  "tol_fail,max_rel_err,mean_rel_err,err_class,"                               \
  "warmup,git_commit,team_size,bound_cpus"

static void usage(const char *prog) {
  fprintf(
      stderr,
      "usage: %s --kernel NAME "
      "(--gen R C DENSITY SEED | --mtx A.mtx B.mtx | --mtx-sq M.mtx)\n"
      "          [--runs N] [--warmup W] [--threads N] [--label TAG] "
      "[--header]\n"
      "          [--arm ARM] [--build TAG] [--march FLAGS] [--cflags FLAGS]\n"
      "          [--cc-version VER] [--pinning DESC] [--git-commit REV]\n"
      "          [--verify-pinning]\n"
      "arms:    baseline autovec intrinsic\n"
      "kernels:\n",
      prog);

  for (int i = 0; i < N_KERNELS; i++)
    fprintf(stderr, "  %s\n", KERNELS[i].name);

#if !defined(__riscv_vector) && !defined(__AVX512F__)
  fprintf(stderr, "  (vector kernels omitted: this binary was built without "
                  "the V extension or AVX-512F)\n");
#endif
}

/* Reject commas in metadata fields to preserve the fixed CSV schema. */
static int reject_comma(const char *val, const char *what) {
  if (val && strchr(val, ',')) {
    fprintf(stderr, "%s must not contain a comma (got \"%s\")\n", what, val);
    return 1;
  }

  return 0;
}

int main(int argc, char **argv) {
  const char *kernel_name = NULL;
  const char *label = "-";

  const char *arm = "-";
  const char *build = "-";
  const char *march = "-";
  const char *cc_version = "-";
  const char *cflags = "-";
  const char *pinning = "-";
  const char *git_commit = "-";
  int verify_pinning = 0;

  int runs = 10;
  int warmup = 3;
  int threads = 1;
  int header = 0;

  enum { SRC_NONE, SRC_GEN, SRC_MTX, SRC_MTXSQ } src = SRC_NONE;

  int gen_r = 0;
  int gen_c = 0;
  int gen_seed = 0;
  double gen_density = 0.0;

  const char *mtx_a = NULL;
  const char *mtx_b = NULL;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--kernel") && i + 1 < argc)
      kernel_name = argv[++i];
    else if (!strcmp(argv[i], "--runs") && i + 1 < argc)
      runs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
      warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
      threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--label") && i + 1 < argc)
      label = argv[++i];
    else if (!strcmp(argv[i], "--arm") && i + 1 < argc)
      arm = argv[++i];
    else if (!strcmp(argv[i], "--build") && i + 1 < argc)
      build = argv[++i];
    else if (!strcmp(argv[i], "--march") && i + 1 < argc)
      march = argv[++i];
    else if (!strcmp(argv[i], "--cflags") && i + 1 < argc)
      cflags = argv[++i];
    else if (!strcmp(argv[i], "--cc-version") && i + 1 < argc)
      cc_version = argv[++i];
    else if (!strcmp(argv[i], "--pinning") && i + 1 < argc)
      pinning = argv[++i];
    else if (!strcmp(argv[i], "--git-commit") && i + 1 < argc)
      git_commit = argv[++i];
    else if (!strcmp(argv[i], "--verify-pinning"))
      verify_pinning = 1;
    else if (!strcmp(argv[i], "--header"))
      header = 1;
    else if (!strcmp(argv[i], "--gen") && i + 4 < argc) {
      src = SRC_GEN;
      gen_r = atoi(argv[++i]);
      gen_c = atoi(argv[++i]);
      gen_density = atof(argv[++i]);
      gen_seed = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--mtx") && i + 2 < argc) {
      src = SRC_MTX;
      mtx_a = argv[++i];
      mtx_b = argv[++i];
    } else if (!strcmp(argv[i], "--mtx-sq") && i + 1 < argc) {
      src = SRC_MTXSQ;
      mtx_a = argv[++i];
    } else {
      fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]);
      usage(argv[0]);
      return 2;
    }
  }

  if (header)
    printf("%s\n", CSV_HEADER);

  if (reject_comma(label, "--label") || reject_comma(arm, "--arm") ||
      reject_comma(build, "--build") || reject_comma(march, "--march") ||
      reject_comma(cflags, "--cflags") ||
      reject_comma(cc_version, "--cc-version") ||
      reject_comma(pinning, "--pinning") ||
      reject_comma(git_commit, "--git-commit"))
    return 2;

  if (!kernel_name || src == SRC_NONE) {
    usage(argv[0]);
    return 2;
  }

  const kernel_entry_t *K = find_kernel(kernel_name);

  if (!K) {
    fprintf(stderr, "no such kernel: %s\n", kernel_name);
    return 2;
  }

  /* Build input matrices. */
  rvsp_csr_matrix_t A = {0};
  rvsp_csr_matrix_t B = {0};

  raw_csr_t rawA = {0};
  raw_csr_t rawB = {0};

  csr_matrix_t gA = {0};
  csr_matrix_t gB = {0};

  struct CSR pA = {0};
  struct CSR pB = {0};

  if (src == SRC_GEN) {
    genmat_params_t p = genmat_default_params(gen_r, gen_c);

    p.density = gen_density;
    p.random_seed = gen_seed;
    p.cv = 0.5;
    p.min = 1;

    gA = genmat_generate_csr(p);

    genmat_params_t p2 = genmat_default_params(gen_c, gen_r);

    p2.density = gen_density;
    p2.random_seed = gen_seed + 1;
    p2.cv = 0.5;
    p2.min = 1;

    gB = genmat_generate_csr(p2);

    if (gA.nrows < 0 || gB.nrows < 0) {
      fprintf(stderr, "genmat failed\n");
      return 1;
    }

    if (build_from_genmat(&gA, K->dtype, &A, &rawA) != RVSP_SUCCESS ||
        build_from_genmat(&gB, K->dtype, &B, &rawB) != RVSP_SUCCESS) {
      fprintf(stderr, "build_from_genmat failed\n");
      return 1;
    }
  } else if (src == SRC_MTX) {
    pA = assemble_csr_matrix(mtx_a);
    pB = assemble_csr_matrix(mtx_b);

    if (!pA.row_ptr || !pB.row_ptr) {
      fprintf(stderr, "mtx load failed\n");
      return 1;
    }

    if (build_from_mtx(&pA, K->dtype, &A, &rawA, 0) != RVSP_SUCCESS ||
        build_from_mtx(&pB, K->dtype, &B, &rawB, 0) != RVSP_SUCCESS) {
      fprintf(stderr, "build_from_mtx failed\n");
      return 1;
    }
  } else {
    pA = assemble_csr_matrix(mtx_a);

    if (!pA.row_ptr) {
      fprintf(stderr, "mtx load failed\n");
      return 1;
    }

    if (build_from_mtx(&pA, K->dtype, &A, &rawA, 1) != RVSP_SUCCESS) {
      fprintf(stderr, "build_from_mtx failed\n");
      return 1;
    }

    B = A;
  }

  if (rvsp_csr_validate(&A) != RVSP_SUCCESS ||
      rvsp_csr_validate(&B) != RVSP_SUCCESS) {
    fprintf(stderr, "csr validate failed (A or B malformed)\n");
    return 1;
  }

  /* The vector kernels are only correct on canonical CSR: a repeated column
   * in a row would put two lanes of one gather/scatter on the same
   * accumulator and lose an update. The loader canonicalizes; this makes it
   * a checked precondition rather than an assumption. */
  {
    const rvsp_csr_matrix_t *M[2] = {&A, &B};

    for (int m = 0; m < 2; m++) {
      int32_t bad_row = -1;
      rvsp_csr_status_t cs =
          rvsp_csr_check(M[m]->rows, M[m]->cols, M[m]->row_ptr,
                         M[m]->col_idx, &bad_row);

      if (cs != RVSP_CSR_OK) {
        fprintf(stderr, "%s is not canonical CSR: %s (row %d)\n",
                m ? "B" : "A", rvsp_csr_status_string(cs), (int)bad_row);
        return EXIT_PRECONDITION;
      }
    }
  }

  /* Compute workload statistics outside the timed region. */
  int64_t *op_counts = (int64_t *)calloc((size_t)A.rows, sizeof(int64_t));

  if (!op_counts) {
    fprintf(stderr, "op_counts allocation failed\n");
    return 1;
  }

  double flops = 2.0 * intermediate_products(&A, &B, op_counts);

  double op_mean;
  double op_max;
  double op_var;

  op_stats(op_counts, A.rows, &op_mean, &op_max, &op_var);

  /*
   * Two phase path. work_estimation does the symbolic pass and sizes C, the
   * caller owns C's storage, and compute() runs the numeric pass only. That
   * keeps symbolic work and every allocation outside the timed region.
   */
  rvsp_spgemm_descr_t descr = NULL;

  if (rvsp_spgemm_descr_create(&descr) != RVSP_SUCCESS) {
    fprintf(stderr, "rvsp_spgemm_descr_create failed\n");
    return 1;
  }

  if (rvsp_spgemm_set_algo(descr, K->algo) != RVSP_SUCCESS) {
    fprintf(stderr, "rvsp_spgemm_set_algo failed for %s\n", K->name);
    return 1;
  }

  void *ws = NULL;
  size_t ws_bytes = 0;
  int32_t c_nnz = 0;

  {
    rvsp_status_t we =
        rvsp_spgemm_work_estimation(descr, &A, &B, &ws_bytes, &c_nnz);

    if (we != RVSP_SUCCESS) {
      fprintf(stderr, "rvsp_spgemm_work_estimation failed (%d)\n", (int)we);
      return 1;
    }

    ws = malloc(ws_bytes);

    if (!ws) {
      fprintf(stderr, "workspace allocation failed (%zu bytes)\n", ws_bytes);
      return 1;
    }
  }

  /* Caller-allocated C, reused by every timed call. */
  rvsp_csr_matrix_t C = {0};

  {
    const size_t nnz_alloc = c_nnz > 0 ? (size_t)c_nnz : 1;

    C.rows = A.rows;
    C.cols = B.cols;
    C.nnz = c_nnz;
    C.dtype = RVSP_DTYPE_FP32;
    C.format = RVSP_FORMAT_CSR;
    C.owns_data = 0;

    C.row_ptr = malloc(((size_t)A.rows + 1) * sizeof(int32_t));
    C.col_idx = malloc(nnz_alloc * sizeof(int32_t));
    C.values = malloc(nnz_alloc * sizeof(float));

    if (!C.row_ptr || !C.col_idx || !C.values) {
      fprintf(stderr, "output allocation failed\n");
      return 1;
    }
  }

  /* Cross-check the descriptor's op counts against the harness. */
  {
    int64_t *kernel_ops = (int64_t *)calloc((size_t)A.rows, sizeof(int64_t));

    if (kernel_ops &&
        rvsp_spgemm_get_op_counts(descr, kernel_ops, A.rows) == RVSP_SUCCESS) {

      for (int32_t i = 0; i < A.rows; i++) {
        if (kernel_ops[i] != op_counts[i]) {
          fprintf(stderr,
                  "warning: descriptor op_counts disagrees with the "
                  "harness at row %d (%lld vs %lld)\n",
                  i, (long long)kernel_ops[i], (long long)op_counts[i]);
          break;
        }
      }
    }

    free(kernel_ops);
  }

  /*
   * First compute() fills C's column indices and is not timed. Every later
   * call sees the same C.col_idx buffer and skips the symbolic fill, so the
   * timed region is the numeric pass.
   */
  if (rvsp_spgemm_compute(descr, &A, &B, &C, ws) != RVSP_SUCCESS) {
    fprintf(stderr, "first rvsp_spgemm_compute failed\n");
    return 1;
  }

  /*
   * Reference result, built outside the timed region. The one-shot entry
   * point needs the full workspace, which is larger than the compute-only
   * one work_estimation reports, so it gets its own buffer.
   */
  rvsp_csr_matrix_t Cref = {0};
  int have_ref = 0;

  {
    size_t ref_ws_bytes = 0;

    if (rvsp_spgemm_buffer_size(B.cols, &ref_ws_bytes) == RVSP_SUCCESS) {
      void *ref_ws = malloc(ref_ws_bytes);

      if (ref_ws) {
        have_ref =
            (K->ref(&A, &B, &Cref, ref_ws, ref_ws_bytes, NULL) == RVSP_SUCCESS);
        free(ref_ws);
      }
    }

    if (!have_ref) {
      fprintf(stderr, "reference build failed: nothing to validate against\n");
      return EXIT_PRECONDITION;
    }
  }

  /* Rounding bound per reference entry and a float64 check of the
   * reference, outside the timed region. */
  long ref_bad = 0;
  float *ref_mag = term_magnitudes(&A, &B, &Cref, &ref_bad);

  if (!ref_mag) {
    fprintf(stderr, "term magnitude allocation failed\n");
    return 1;
  }

  if (ref_bad > 0) {
    fprintf(stderr,
            "reference %s fails its float64 check: %ld of %d entries are "
            "further from the exact product than rounding allows\n",
            K->ref_name, ref_bad, Cref.nnz);
    return EXIT_PRECONDITION;
  }

  /* Check the binding before any measured work. */
  binding_t bind;
  probe_binding(K->algo == RVSP_SPGEMM_ALGO_OMP, threads, &bind);

  if (bind.problem[0]) {
    fprintf(stderr, "%s: %s\n",
            verify_pinning ? "pinning not as intended" : "warning: pinning",
            bind.problem);

    if (verify_pinning)
      return EXIT_PRECONDITION;
  }

  /*
   * Every iteration, warmup or timed, does the same three things: wipe C's
   * values to NaN, run compute(), validate. The wipe means a kernel that
   * stops writing C cannot pass on the previous run's values, and doing the
   * same work around warmups as around timed runs means run 0 starts from
   * the same cache state as every later run.
   */
  const size_t c_val_bytes = (size_t)(c_nnz > 0 ? c_nnz : 1) * sizeof(float);

  for (int w = 0; w < warmup; w++) {
    err_stats_t ws_es;
    memset(C.values, 0xFF, c_val_bytes); /* all-ones is a NaN */
    rvsp_spgemm_compute(descr, &A, &B, &C, ws);
    err_stats(&C, &Cref, &A, ref_mag, &ws_es);
    (void)csr_equal(&C, &Cref, K->dtype);
  }

  /* Timed runs. */
  perf_init();

  for (int r = 0; r < runs; r++) {
    long long cycles = -1;
    long long insns = -1;
    long long stalls = -1;
    double mux = -1.0;

    memset(C.values, 0xFF, c_val_bytes); /* all-ones is a NaN */

    perf_start();

    double t0 = now_seconds();

    rvsp_status_t st = rvsp_spgemm_compute(descr, &A, &B, &C, ws);

    double t1 = now_seconds();

    perf_stop(&cycles, &insns, &stalls, &mux);

    double dt = t1 - t0;
    int correct = 0;
    int32_t nnz_c = 0;
    err_stats_t es = {0, 0.0, 0.0, "failed"};

    if (st == RVSP_SUCCESS) {
      nnz_c = C.nnz;

      err_stats(&C, &Cref, &A, ref_mag, &es);

      /* Within tolerance is not enough: a structural or beyond-rounding
       * difference is a wrong answer. */
      correct = csr_equal(&C, &Cref, K->dtype) &&
                strcmp(es.cls, "structural") != 0 &&
                strcmp(es.cls, "value") != 0;
    }

    double gops = (dt > 0.0) ? (flops / dt) / 1e9 : 0.0;

    printf("%s,%s,%s,%s,%s,%s,%s,%s,"
           "%d,%d,%d,%d,%d,%.0f,%.4f,%.0f,%.4f,"
           "%d,%.9f,%.4f,%d,",
           label, K->name, arm, build, march, cflags, cc_version, K->dtype_str,
           A.rows, A.cols, A.nnz, B.nnz, nnz_c, flops, op_mean, op_max, op_var,
           r, dt, gops, correct);

    if (cycles >= 0)
      printf("%lld,", cycles);
    else
      printf(",");

    if (insns >= 0)
      printf("%lld,", insns);
    else
      printf(",");

    printf("%d,", threads);

    if (stalls >= 0)
      printf("%lld,", stalls);
    else
      printf(",");

    if (mux >= 0.0)
      printf("%.4f,", mux);
    else
      printf(",");

    /* Blank for kernels that are not RVV. */
    if (K->lmul > 0)
      printf("%d,", K->lmul);
    else
      printf(",");

    printf("%s,%s,", pinning, perf_note);

    if (!strcmp(es.cls, "exact") || !strcmp(es.cls, "rounding") ||
        !strcmp(es.cls, "value"))
      printf("%ld,%.3e,%.3e,%s", es.tol_fail, es.max_rel, es.mean_rel,
             es.cls);
    else
      printf(",,,%s", es.cls);

    printf(",%d,%s,%d,%s\n", warmup, git_commit, bind.team, bind.cpus);

    fflush(stdout);
  }

  perf_close();

  /* Release benchmark resources. */
  rvsp_spgemm_descr_destroy(descr);

  free(C.row_ptr);
  free(C.col_idx);
  free(C.values);

  free(ws);
  free(op_counts);
  free(ref_mag);

  if (have_ref)
    rvsp_csr_destroy(&Cref);

  rvsp_csr_destroy(&A);

  if (src != SRC_MTXSQ)
    rvsp_csr_destroy(&B);

  free_raw(&rawA);

  if (src != SRC_MTXSQ)
    free_raw(&rawB);

  if (src == SRC_GEN) {
    genmat_free_csr(&gA);
    genmat_free_csr(&gB);
  }

  if (src == SRC_MTX || src == SRC_MTXSQ) {
    vector_free(pA.row_ptr);
    vector_free(pA.col_ind);
    vector_free(pA.val);

    if (src == SRC_MTX) {
      vector_free(pB.row_ptr);
      vector_free(pB.col_ind);
      vector_free(pB.val);
    }
  }

  return 0;
}
