/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Standalone microbenchmark for the SpGEMM inner primitive:
 *
 *     acc[idx[i]] += a * val[i]
 *
 * Scalar against hand-written RVV indexed gather/scatter, in isolation from
 * any SpGEMM code. Two sweeps:
 *
 *   1. vector fill, memory held constant. acc stays L1 resident, the active
 *      vector length is forced to 2, 4, 8. Isolates gather/scatter issue cost
 *      from memory latency.
 *
 *   2. working set, fill at the natural maximum. acc grows from L1 to well
 *      past L2. Gives the random-access latency ceiling.
 *
 * Build:
 *   gcc -O3 -march=rv64gcv -mabi=lp64d \
 *       -fno-tree-vectorize -fno-tree-slp-vectorize \
 *       micro/gather_scatter.c -o micro/gather_scatter -lm
 *
 * Add -DPERF for cycles, instructions and stalled-cycles-backend.
 */

#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#else
#error "gather_scatter.c requires the RISC-V V extension (-march=rv64gcv)"
#endif

#ifndef REPS
#define REPS 200
#endif

#define WARMUP 5
#define SEED 20240517

/* Elements pushed through the kernel per repetition. Fixed across every point
 * so time-per-element is comparable. */
#define NELEM (1 << 20)

/* ------------------------------------------------------------------------ */
/* Optional hardware counters. Mirrors bench/bench.c so the two sets of
 * numbers are directly comparable.                                          */
/* ------------------------------------------------------------------------ */

#ifdef PERF
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <errno.h>

struct perf_read {
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
};

static int perf_fd_cycles = -1;
static int perf_fd_insns  = -1;
static int perf_fd_stalls = -1;

static int perf_open_one(uint32_t type, uint64_t config) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type           = type;
    pe.size           = sizeof(pe);
    pe.config         = config;
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    pe.read_format    = PERF_FORMAT_TOTAL_TIME_ENABLED |
                        PERF_FORMAT_TOTAL_TIME_RUNNING;

    return (int)syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
}

static void perf_init(void) {
    perf_fd_cycles = perf_open_one(PERF_TYPE_HARDWARE,
                                   PERF_COUNT_HW_CPU_CYCLES);
    perf_fd_insns  = perf_open_one(PERF_TYPE_HARDWARE,
                                   PERF_COUNT_HW_INSTRUCTIONS);
    perf_fd_stalls = perf_open_one(PERF_TYPE_HARDWARE,
                                   PERF_COUNT_HW_STALLED_CYCLES_BACKEND);

    if (perf_fd_cycles < 0 || perf_fd_insns < 0)
        fprintf(stderr, "warning: perf counters unavailable (%s); "
                        "cycles/instructions will be blank\n",
                strerror(errno));

    if (perf_fd_stalls < 0)
        fprintf(stderr, "warning: stalled-cycles-backend unavailable on this "
                        "PMU; the column will be blank\n");
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

static void perf_stop(long long *cycles, long long *insns,
                      long long *stalls, double *mux) {
    uint64_t enabled = 0;
    uint64_t running = 0;

    *cycles = perf_read_one(perf_fd_cycles, NULL, NULL);
    *insns  = perf_read_one(perf_fd_insns, NULL, NULL);
    *stalls = perf_read_one(perf_fd_stalls, &enabled, &running);

    *mux = (enabled > 0) ? (double)running / (double)enabled : -1.0;
}

static void perf_close(void) {
    if (perf_fd_cycles >= 0) close(perf_fd_cycles);
    if (perf_fd_insns  >= 0) close(perf_fd_insns);
    if (perf_fd_stalls >= 0) close(perf_fd_stalls);
}
#else
static void perf_init(void) {}
static void perf_start(void) {}
static void perf_stop(long long *cycles, long long *insns,
                      long long *stalls, double *mux) {
    *cycles = -1; *insns = -1; *stalls = -1; *mux = -1.0;
}
static void perf_close(void) {}
#endif

/* ------------------------------------------------------------------------ */
/* Kernels                                                                   */
/* ------------------------------------------------------------------------ */

/* Plain loop. The whole file is built with -fno-tree-vectorize so this stays
 * scalar; the RVV kernel below is intrinsics and is unaffected by that. */
static void kernel_scalar(float *acc, const int32_t *idx, const float *val,
                          int32_t n, float a) {
    for (int32_t i = 0; i < n; i++)
        acc[idx[i]] += a * val[i];
}

/*
 * Indexed gather, FMA, indexed scatter at LMUL=1.
 *
 * vl_cap forces the active vector length: pass 0 for the natural maximum, or
 * 2/4/8 to measure a partially filled vector. Index EEW is 32 at SEW=32, so
 * the index vector LMUL matches the data LMUL.
 */
static void kernel_rvv(float *acc, const int32_t *idx, const float *val,
                       int32_t n, float a, size_t vl_cap) {
    int32_t p = 0;

    while (p < n) {
        size_t want = (size_t)(n - p);

        if (vl_cap && want > vl_cap)
            want = vl_cap;

        const size_t vl = __riscv_vsetvl_e32m1(want);

        const vfloat32m1_t vb = __riscv_vle32_v_f32m1(&val[p], vl);
        const vint32m1_t vidx = __riscv_vle32_v_i32m1(&idx[p], vl);

        /* Indices are duplicate-free within a window, so the gather and the
         * scatter below cannot alias inside one vl. */
        const vuint32m1_t voff =
            __riscv_vreinterpret_v_i32m1_u32m1(
                __riscv_vsll_vx_i32m1(vidx, 2, vl));

        vfloat32m1_t vacc = __riscv_vluxei32_v_f32m1(acc, voff, vl);

        vacc = __riscv_vfmacc_vf_f32m1(vacc, a, vb, vl);

        __riscv_vsuxei32_v_f32m1(acc, voff, vacc, vl);

        p += (int32_t)vl;
    }
}

/* ------------------------------------------------------------------------ */
/* Inputs                                                                    */
/* ------------------------------------------------------------------------ */

static uint64_t rng_state;

static void rng_seed(uint64_t s) { rng_state = s ? s : 1; }

static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

/*
 * Random indices into acc, duplicate-free within each window of `window`
 * elements.
 *
 * The duplicate-free constraint is not cosmetic. The real SpGEMM kernel walks
 * canonical CSR rows, whose column indices are unique, so a gathered group
 * never repeats a slot. It also matters for the measurement itself: two equal
 * indices inside one vsuxei32 are a read-modify-write race, and the last write
 * wins, so the result would be wrong and the timing would describe work the
 * real kernel never does.
 *
 * Rejection against a small open-addressed set is enough because window is at
 * most the hardware vector length, far smaller than acc_elems.
 */
static void make_indices(int32_t *idx, int32_t n, int32_t acc_elems,
                         int32_t window) {
    int32_t seen[64];

    for (int32_t base = 0; base < n; base += window) {
        int32_t w = (base + window <= n) ? window : (n - base);

        if (w > (int32_t)(sizeof(seen) / sizeof(seen[0])))
            w = (int32_t)(sizeof(seen) / sizeof(seen[0]));

        for (int32_t j = 0; j < w; j++) {
            int32_t cand;
            int dup;

            do {
                cand = (int32_t)(rng_next() % (uint32_t)acc_elems);
                dup = 0;

                for (int32_t k = 0; k < j; k++)
                    if (seen[k] == cand) { dup = 1; break; }
            } while (dup);

            seen[j] = cand;
            idx[base + j] = cand;
        }

        /* Any tail beyond the dedup window is still in range. */
        for (int32_t j = w; j < window && base + j < n; j++)
            idx[base + j] = (int32_t)(rng_next() % (uint32_t)acc_elems);
    }
}

/* ------------------------------------------------------------------------ */
/* Timing                                                                    */
/* ------------------------------------------------------------------------ */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* Keep the optimiser from deleting the accumulator updates. */
static void keep_alive(void *p) {
    __asm__ __volatile__("" :: "r"(p) : "memory");
}

/* ------------------------------------------------------------------------ */
/* Measurement                                                               */
/* ------------------------------------------------------------------------ */

static void measure(const char *sweep, const char *kernel, long swept,
                    float *acc, int32_t acc_elems,
                    const int32_t *idx, const float *val,
                    int32_t n, int use_rvv, size_t vl_cap) {

    double *samples = malloc((size_t)REPS * sizeof(double));

    if (!samples) {
        fprintf(stderr, "sample allocation failed\n");
        return;
    }

    for (int w = 0; w < WARMUP; w++) {
        if (use_rvv) kernel_rvv(acc, idx, val, n, 1.000001f, vl_cap);
        else         kernel_scalar(acc, idx, val, n, 1.000001f);
        keep_alive(acc);
    }

    long long cycles = -1, insns = -1, stalls = -1;
    double mux = -1.0;

    perf_start();

    for (int r = 0; r < REPS; r++) {
        double t0 = now_seconds();

        if (use_rvv) kernel_rvv(acc, idx, val, n, 1.000001f, vl_cap);
        else         kernel_scalar(acc, idx, val, n, 1.000001f);

        double t1 = now_seconds();

        keep_alive(acc);
        samples[r] = t1 - t0;
    }

    perf_stop(&cycles, &insns, &stalls, &mux);

    const double med = median(samples, REPS);
    const double per_elem_ns = med * 1e9 / (double)n;
    const double melem_s = (med > 0.0) ? (double)n / med / 1e6 : 0.0;

    printf("%s,%s,%ld,%d,%d,%.9f,%.4f,%.2f,",
           sweep, kernel, swept, acc_elems, n, med, per_elem_ns, melem_s);

    if (cycles >= 0) printf("%lld,", cycles); else printf(",");
    if (insns  >= 0) printf("%lld,", insns);  else printf(",");
    if (stalls >= 0) printf("%lld,", stalls); else printf(",");
    if (mux  >= 0.0) printf("%.4f\n", mux);   else printf("\n");

    fflush(stdout);
    free(samples);
}

/* ------------------------------------------------------------------------ */

static int verify(void) {
    const int32_t acc_elems = 4096;
    const int32_t n = 4096;

    float *a1 = calloc((size_t)acc_elems, sizeof(float));
    float *a2 = calloc((size_t)acc_elems, sizeof(float));
    int32_t *idx = malloc((size_t)n * sizeof(int32_t));
    float *val = malloc((size_t)n * sizeof(float));

    if (!a1 || !a2 || !idx || !val)
        return -1;

    rng_seed(SEED);
    make_indices(idx, n, acc_elems, (int32_t)__riscv_vsetvlmax_e32m1());

    for (int32_t i = 0; i < n; i++)
        val[i] = (float)((rng_next() % 1000) + 1) * 0.001f;

    kernel_scalar(a1, idx, val, n, 1.5f);
    kernel_rvv(a2, idx, val, n, 1.5f, 0);

    double worst = 0.0;

    for (int32_t i = 0; i < acc_elems; i++) {
        double d = fabs((double)a1[i] - (double)a2[i]);
        double tol = 1e-5 + 1e-4 * fabs((double)a1[i]);
        if (d > tol && d > worst)
            worst = d;
    }

    free(a1); free(a2); free(idx); free(val);

    if (worst > 0.0) {
        fprintf(stderr, "VERIFY FAILED: worst mismatch %.6g\n", worst);
        return -1;
    }

    fprintf(stderr, "verify: scalar and rvv agree on %d elements\n", n);
    return 0;
}

static void sweep_fill(void) {
    /* acc is 16 KB so it stays L1 resident and memory latency is held out of
     * the comparison. */
    const int32_t acc_elems = 4096;
    const int32_t n = NELEM;

    float *acc = calloc((size_t)acc_elems, sizeof(float));
    int32_t *idx = malloc((size_t)n * sizeof(int32_t));
    float *val = malloc((size_t)n * sizeof(float));

    if (!acc || !idx || !val) {
        fprintf(stderr, "sweep_fill allocation failed\n");
        return;
    }

    const size_t vlmax = __riscv_vsetvlmax_e32m1();

    rng_seed(SEED);
    make_indices(idx, n, acc_elems, (int32_t)vlmax);

    for (int32_t i = 0; i < n; i++)
        val[i] = (float)((rng_next() % 1000) + 1) * 0.001f;

    measure("fill", "scalar", 1, acc, acc_elems, idx, val, n, 0, 0);

    const size_t vls[] = {2, 4, 8};

    for (size_t i = 0; i < sizeof(vls) / sizeof(vls[0]); i++)
        measure("fill", "rvv", (long)vls[i], acc, acc_elems,
                idx, val, n, 1, vls[i]);

    measure("fill", "rvv", (long)vlmax, acc, acc_elems, idx, val, n, 1, 0);

    free(acc); free(idx); free(val);
}

static void sweep_working_set(void) {
    const size_t kb[] = {16, 256, 1024, 8192, 65536};
    const int32_t n = NELEM;

    int32_t *idx = malloc((size_t)n * sizeof(int32_t));
    float *val = malloc((size_t)n * sizeof(float));

    if (!idx || !val) {
        fprintf(stderr, "sweep_working_set allocation failed\n");
        return;
    }

    const size_t vlmax = __riscv_vsetvlmax_e32m1();

    for (size_t s = 0; s < sizeof(kb) / sizeof(kb[0]); s++) {
        const int32_t acc_elems =
            (int32_t)(kb[s] * 1024 / sizeof(float));

        float *acc = calloc((size_t)acc_elems, sizeof(float));

        if (!acc) {
            fprintf(stderr, "acc allocation failed at %zu KB\n", kb[s]);
            continue;
        }

        rng_seed(SEED);
        make_indices(idx, n, acc_elems, (int32_t)vlmax);

        for (int32_t i = 0; i < n; i++)
            val[i] = (float)((rng_next() % 1000) + 1) * 0.001f;

        measure("wss", "scalar", (long)kb[s], acc, acc_elems,
                idx, val, n, 0, 0);
        measure("wss", "rvv", (long)kb[s], acc, acc_elems,
                idx, val, n, 1, 0);

        free(acc);
    }

    free(idx); free(val);
}

int main(void) {
    if (verify() != 0)
        return 1;

    fprintf(stderr, "vlmax(e32m1) = %zu elements, REPS = %d\n",
            __riscv_vsetvlmax_e32m1(), REPS);

    perf_init();

    printf("sweep,kernel,swept,acc_elems,n,time_s,ns_per_elem,melem_per_s,"
           "cycles,instructions,stalls_backend,perf_mux\n");

    sweep_fill();
    sweep_working_set();

    perf_close();
    return 0;
}
