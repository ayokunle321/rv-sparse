/*
 * bench_spgemm.c — Benchmark harness for rv-sparse CSR SpGEMM kernels.
 * Cycle-accurate evaluation under gem5 RISC-V SE mode.
 *
 * Measurement:
 *   ROI (m5_reset_stats → m5_dump_stats) spans one complete kernel
 *   invocation: symbolic pass, output allocation, numeric pass.
 *
 *   Warm protocol (default): one untimed invocation grows the arena
 *   and warms caches, then a timed invocation. This is the headline
 *   "kernel cost" number — cold puts slab mmaps and first-touch
 *   faults inside the ROI, and gem5 SE fault handling is not
 *   representative of real hardware.
 *
 *   Cold protocol (argv[3] = "cold"): single invocation, allocator
 *   syscalls inside ROI. Secondary, end-to-end number only.
 *
 * Verification:
 *   FNV-1a hashes over output structure and values, plus a f64 value
 *   sum. Warm mode: warmup vs timed hashes compared (determinism
 *   check). Cold mode: no second invocation exists, so the check
 *   cannot run — sidecar reports -1 (n/a), never a vacuous pass.
 *
 * Output:
 *   Sidecar CSV (argv[2]) — one line, no header:
 *     matrix,kernel,mode,M,A_nnz,madd_pairs,ops,bytes,AI,
 *     C_nnz,compression,struct_hash,val_hash,val_sum,determinism_ok
 *   determinism_ok: 1 pass, 0 FAIL, -1 not applicable (cold).
 *
 * Build:
 *   riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gc -static -Iinclude \
 *       -I$M5_INC -DKERNEL_I8 \
 *       bench/bench_spgemm.c src/kernels/spgemm/csr_scalar_i8.c \
 *       $M5_LIB/libm5.a -o bench_i8_gc
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "rv_sparse_types.h"

#ifdef NO_M5
  static inline void m5_reset_stats(uint64_t a, uint64_t b) { (void)a;(void)b; }
  static inline void m5_dump_stats(uint64_t a, uint64_t b)  { (void)a;(void)b; }
#else
  #include <gem5/m5ops.h>
#endif

/* ------------------------------------------------------------------ */
/* Bump allocator (gem5 builds only)                                   */
/*                                                                     */
/* gem5 SE mode's brk emulation is unreliable under malloc/free/       */
/* realloc traffic. Sidestep it: malloc bumps a pointer inside mmap'd  */
/* slabs, free is a no-op. Fine for a run-once benchmark. Native       */
/* builds (-DNO_M5) keep glibc malloc so ASan/UBSan still interpose.   */
/* ------------------------------------------------------------------ */
#ifndef NO_M5
#include <sys/mman.h>

#define BUMP_ALIGN 16UL
#define BUMP_SLAB  (256UL * 1024 * 1024)

static unsigned char *bump_cur = 0, *bump_end = 0;

typedef struct { size_t size; size_t pad_; } bump_hdr;

void *malloc(size_t n)
{
    if (n == 0) n = 1;
    size_t need = (n + sizeof(bump_hdr) + BUMP_ALIGN - 1) & ~(BUMP_ALIGN - 1);
    if ((size_t)(bump_end - bump_cur) < need) {
        size_t slab = need > BUMP_SLAB ? need : BUMP_SLAB;
        void *p = mmap(0, slab, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return 0;
        bump_cur = (unsigned char *)p;
        bump_end = bump_cur + slab;
    }
    bump_hdr *h = (bump_hdr *)bump_cur;
    h->size = n;
    bump_cur += need;
    return h + 1;
}

void free(void *p) { (void)p; }

void *calloc(size_t nm, size_t sz)
{
    if (sz && nm > (size_t)-1 / sz) return 0;
    return malloc(nm * sz);  /* bump never reuses; mmap pages are zero */
}

void *realloc(void *p, size_t n)
{
    if (!p) return malloc(n);
    if (n == 0) return 0;
    size_t old = ((bump_hdr *)p - 1)->size;
    if (n <= old) { ((bump_hdr *)p - 1)->size = n; return p; }
    void *q = malloc(n);
    if (q) memcpy(q, p, old);
    return q;
}

void *reallocarray(void *p, size_t nm, size_t sz)
{
    if (sz && nm > (size_t)-1 / sz) return 0;
    return realloc(p, nm * sz);
}
#endif /* !NO_M5 */

/* ------------------------------------------------------------------ */
/* Per-kernel configuration (compile-time)                             */
/* ------------------------------------------------------------------ */
#if defined(KERNEL_F32)
  #define KERNEL_NAME     "csr_scalar_f32"
  typedef float   val_t;  typedef float   out_t;
  #define VAL_IS_FLOAT    1
  #define BYTES_PER_MADD  16.0
  #define INTENSITY_LABEL "FLOP/byte"
  rvsp_status_t rvsp_spgemm_csr_scalar_f32_raw(
      int32_t,int32_t,int32_t,
      const int32_t*,const int32_t*,const float*,
      const int32_t*,const int32_t*,const float*,
      int32_t**,int32_t**,float**,int32_t*);
  #define CALL_KERNEL(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz) \
      rvsp_spgemm_csr_scalar_f32_raw(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz)

#elif defined(KERNEL_I8)
  #define KERNEL_NAME     "csr_scalar_i8"
  typedef int8_t  val_t;  typedef int32_t out_t;
  #define VAL_IS_FLOAT    0
  #define BYTES_PER_MADD  13.0
  #define INTENSITY_LABEL "IOP/byte"
  rvsp_status_t rvsp_spgemm_csr_scalar_i8_raw(
      int32_t,int32_t,int32_t,
      const int32_t*,const int32_t*,const int8_t*,
      const int32_t*,const int32_t*,const int8_t*,
      int32_t**,int32_t**,int32_t**,int32_t*);
  #define CALL_KERNEL(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz) \
      rvsp_spgemm_csr_scalar_i8_raw(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz)

#elif defined(KERNEL_RVV_I8)
  #define KERNEL_NAME     "csr_rvv_i8_indexed_marked"
  typedef int8_t  val_t;  typedef int32_t out_t;
  #define VAL_IS_FLOAT    0
  #define BYTES_PER_MADD  13.0
  #define INTENSITY_LABEL "IOP/byte"
  rvsp_status_t rvsp_spgemm_csr_rvv_i8_indexed_marked_raw(
      int32_t,int32_t,int32_t,
      const int32_t*,const int32_t*,const int8_t*,
      const int32_t*,const int32_t*,const int8_t*,
      int32_t**,int32_t**,int32_t**,int32_t*);
  #define CALL_KERNEL(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz) \
      rvsp_spgemm_csr_rvv_i8_indexed_marked_raw(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz)

#elif defined(KERNEL_RVV_F32)
  #define KERNEL_NAME     "csr_rvv_f32_indexed_marked"
  typedef float   val_t;  typedef float   out_t;
  #define VAL_IS_FLOAT    1
  #define BYTES_PER_MADD  16.0
  #define INTENSITY_LABEL "FLOP/byte"
  rvsp_status_t rvsp_spgemm_csr_rvv_f32_indexed_marked_raw(
      int32_t,int32_t,int32_t,
      const int32_t*,const int32_t*,const float*,
      const int32_t*,const int32_t*,const float*,
      int32_t**,int32_t**,float**,int32_t*);
  #define CALL_KERNEL(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz) \
      rvsp_spgemm_csr_rvv_f32_indexed_marked_raw(M,K,N,arp,aci,av,brp,bci,bv,crp,cci,cv,cnnz)

#else
    #error "Define one of KERNEL_F32, KERNEL_UNROLL4, KERNEL_I8, KERNEL_RVV_I8, KERNEL_RVV_F32"
#endif

/* ------------------------------------------------------------------ */
/* FNV-1a 64-bit, word-at-a-time.                                      */
/*                                                                     */
/* Byte-at-a-time FNV over ~100MB of output made post-ROI              */
/* verification cost more simulated instructions than the ROI itself.  */
/* Mixing 8 bytes per step is ~8x fewer sim insts. Not byte-FNV        */
/* compatible, but hashes are only ever compared across runs of this   */
/* same harness (warm-vs-timed, kernel-vs-kernel), so consistency is   */
/* all that matters. Runs entirely outside the ROI either way.         */
/* ------------------------------------------------------------------ */
static uint64_t fnv1a(const void *data, size_t nbytes, uint64_t h)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t nwords = nbytes / 8;
    for (size_t i = 0; i < nwords; i++) {
        uint64_t w;
        memcpy(&w, p + i*8, 8);
        h ^= w;
        h *= 0x100000001b3ULL;
    }
    for (size_t i = nwords*8; i < nbytes; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
#define FNV_INIT 0xcbf29ce484222325ULL

/* ------------------------------------------------------------------ */
/* Matrix Market (.mtx) → CSR loader (outside ROI)                     */
/* ------------------------------------------------------------------ */
static int32_t load_mtx(const char *path, int32_t *out_rows, int32_t *out_cols,
                        int32_t **row_ptr, int32_t **col_idx, val_t **values)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(2); }

    char line[2048];
    int is_pattern = 0, is_symmetric = 0, is_complex = 0;
    if (!fgets(line, sizeof(line), f)) { fprintf(stderr, "ERROR: empty file\n"); exit(2); }
    if (strstr(line, "pattern"))                                 is_pattern   = 1;
    if (strstr(line, "complex"))                                 is_complex   = 1;
    if (strstr(line, "symmetric") || strstr(line, "hermitian"))  is_symmetric = 1;

    do { if (!fgets(line, sizeof(line), f)) { fprintf(stderr,"ERROR: no dims\n"); exit(2);} }
    while (line[0] == '%');

    long rows=0, cols=0, entries=0;
    if (sscanf(line, "%ld %ld %ld", &rows, &cols, &entries) != 3) {
        fprintf(stderr, "ERROR: bad dim line\n"); exit(2);
    }

    long cap = is_symmetric ? entries*2 : entries; if (cap < 1) cap = 1;
    int32_t *Ir = malloc((size_t)cap*sizeof(int32_t));
    int32_t *Jc = malloc((size_t)cap*sizeof(int32_t));
    val_t   *Vv = malloc((size_t)cap*sizeof(val_t));
    if (!Ir||!Jc||!Vv){ fprintf(stderr,"ERROR: OOM COO\n"); exit(2);}

    long nnz = 0;
    for (long e = 0; e < entries; e++) {
        if (!fgets(line, sizeof(line), f)) break;
        long r=0,c=0; double v=1.0, vi=0.0;
        if (is_pattern)      { if (sscanf(line,"%ld %ld",&r,&c)!=2) continue; }
        else if (is_complex) { if (sscanf(line,"%ld %ld %lf %lf",&r,&c,&v,&vi)<3) continue; }
        else                 { if (sscanf(line,"%ld %ld %lf",&r,&c,&v)<2) continue; }
        r--; c--;
        if (r < 0 || c < 0 || r >= rows || c >= cols) continue;
        Ir[nnz]=(int32_t)r; Jc[nnz]=(int32_t)c;
#if VAL_IS_FLOAT
        Vv[nnz]=(val_t)v;
#else
        { int t=(int)v; if(t==0)t=1; Vv[nnz]=(val_t)t; }
#endif
        nnz++;
        if (is_symmetric && r!=c) { Ir[nnz]=(int32_t)c; Jc[nnz]=(int32_t)r; Vv[nnz]=Vv[nnz-1]; nnz++; }
    }
    fclose(f);

    /* COO → CSR */
    int32_t *rp = calloc((size_t)rows+1, sizeof(int32_t));
    int32_t *ci = malloc((size_t)(nnz>0?nnz:1)*sizeof(int32_t));
    val_t   *vv = malloc((size_t)(nnz>0?nnz:1)*sizeof(val_t));
    int32_t *cur= malloc((size_t)rows*sizeof(int32_t));
    if(!rp||!ci||!vv||!cur){ fprintf(stderr,"ERROR: OOM CSR\n"); exit(2);}
    for (long e=0;e<nnz;e++) rp[Ir[e]+1]++;
    for (long r=0;r<rows;r++) rp[r+1]+=rp[r];
    for (long r=0;r<rows;r++) cur[r]=rp[r];
    for (long e=0;e<nnz;e++){ int32_t r=Ir[e]; int32_t d=cur[r]++; ci[d]=Jc[e]; vv[d]=Vv[e]; }
    free(Ir);free(Jc);free(Vv);free(cur);

    *out_rows=(int32_t)rows; *out_cols=(int32_t)cols;
    *row_ptr=rp; *col_idx=ci; *values=vv;
    return (int32_t)nnz;
}

/* ------------------------------------------------------------------ */
/* Output digest: structural hash, value hash, value sum               */
/* ------------------------------------------------------------------ */
static void digest_c(int32_t M, const int32_t *c_rp, const int32_t *c_ci,
                     const out_t *c_v, int32_t c_nnz,
                     uint64_t *struct_hash, uint64_t *val_hash, double *val_sum)
{
    uint64_t sh = FNV_INIT;
    sh = fnv1a(c_rp, (size_t)(M+1)*sizeof(int32_t), sh);
    sh = fnv1a(c_ci, (size_t)c_nnz*sizeof(int32_t), sh);
    *struct_hash = sh;
    *val_hash = fnv1a(c_v, (size_t)c_nnz*sizeof(out_t), FNV_INIT);
    double s = 0.0;
    for (int32_t i = 0; i < c_nnz; i++) s += (double)c_v[i];
    *val_sum = s;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <matrix.mtx> <sidecar.csv> [warm|cold]\n", argv[0]);
        return 2;
    }
    const char *mtx_path = argv[1];
    const char *sidecar  = argv[2];
    int warm = 1;                      /* warm is the default protocol */
    if (argc >= 4 && strcmp(argv[3], "cold") == 0) warm = 0;
    const char *mode = warm ? "warm" : "cold";

    /* ---- setup (outside ROI) ---- */
    int32_t M=0, K=0;
    int32_t *a_rp=NULL,*a_ci=NULL; val_t *a_v=NULL;
    int32_t a_nnz = load_mtx(mtx_path, &M, &K, &a_rp, &a_ci, &a_v);
    if (M != K) { fprintf(stderr,"ERROR: matrix %dx%d not square\n",M,K); return 2; }
    int32_t N = K;
    int32_t *b_rp=a_rp,*b_ci=a_ci; val_t *b_v=a_v;  /* C = A² */

    const char *base = strrchr(mtx_path,'/'); base = base ? base+1 : mtx_path;

    /* analytical work count (matrix property, kernel-independent) */
    long long madds = 0;
    for (int32_t row=0; row<M; row++)
        for (int32_t ap=a_rp[row]; ap<a_rp[row+1]; ap++) {
            int32_t k=a_ci[ap];
            madds += (long long)(b_rp[k+1]-b_rp[k]);
        }
    double ops   = 2.0*(double)madds;
    double bytes = BYTES_PER_MADD*(double)madds;
    double ai    = bytes>0 ? ops/bytes : 0.0;

    printf("SETUP %s kernel=%s mode=%s A:%dx%d nnz=%d madds=%lld AI=%.4f %s\n",
           base, KERNEL_NAME, mode, M, K, a_nnz, madds, ai, INTENSITY_LABEL);
    fflush(stdout);

    /* ---- warmup (untimed, outside ROI) ---- */
    uint64_t w_sh=0, w_vh=0; double w_sum=0.0; int have_warm_digest=0;
    if (warm) {
        int32_t *w_rp=NULL,*w_ci=NULL; out_t *w_v=NULL; int32_t w_nnz=0;
        rvsp_status_t wst = CALL_KERNEL(M,K,N,
            a_rp,a_ci,a_v, b_rp,b_ci,b_v,
            &w_rp,&w_ci,&w_v,&w_nnz);
        if (wst != RVSP_SUCCESS) {
            fprintf(stderr, "ERROR: warmup kernel returned %d\n", (int)wst);
            return 3;
        }
        digest_c(M, w_rp, w_ci, w_v, w_nnz, &w_sh, &w_vh, &w_sum);
        have_warm_digest = 1;
        printf("WARMUP done: C_nnz=%d struct=%016llx vals=%016llx sum=%.17g\n",
               w_nnz, (unsigned long long)w_sh, (unsigned long long)w_vh, w_sum);
        fflush(stdout);
        free(w_rp); free(w_ci); free(w_v);
    }

    /* ---- ROI ---- */
    int32_t *c_rp=NULL,*c_ci=NULL; out_t *c_v=NULL; int32_t c_nnz=0;

    m5_reset_stats(0, 0);
    rvsp_status_t st = CALL_KERNEL(M,K,N,
        a_rp,a_ci,a_v, b_rp,b_ci,b_v,
        &c_rp,&c_ci,&c_v,&c_nnz);
    m5_dump_stats(0, 0);

    if (st != RVSP_SUCCESS) {
        fprintf(stderr, "ERROR: kernel returned %d\n", (int)st);
        return 3;
    }

    /* ---- verification (outside ROI) ---- */
    uint64_t sh=0, vh=0; double vsum=0.0;
    digest_c(M, c_rp, c_ci, c_v, c_nnz, &sh, &vh, &vsum);

    /* 1 = pass, 0 = FAIL, -1 = n/a (cold mode has no reference run) */
    int det_ok = have_warm_digest ? 1 : -1;
    if (have_warm_digest && (sh != w_sh || vh != w_vh)) {
        det_ok = 0;
        fprintf(stderr, "ERROR: determinism check FAILED "
                "(warm %016llx/%016llx vs timed %016llx/%016llx)\n",
                (unsigned long long)w_sh, (unsigned long long)w_vh,
                (unsigned long long)sh,   (unsigned long long)vh);
    }

    double compression = (c_nnz>0) ? (double)madds/(double)c_nnz : 0.0;

    /* ---- sidecar ---- */
    FILE *sc = fopen(sidecar, "w");
    if (!sc) { fprintf(stderr,"ERROR: cannot write sidecar %s\n", sidecar); return 4; }
    fprintf(sc, "%s,%s,%s,%d,%d,%lld,%.0f,%.0f,%.6f,%d,%.4f,%016llx,%016llx,%.17g,%d\n",
            base, KERNEL_NAME, mode, M, a_nnz, madds, ops, bytes, ai,
            c_nnz, compression,
            (unsigned long long)sh, (unsigned long long)vh, vsum, det_ok);
    fclose(sc);

    printf("DONE %s C_nnz=%d compression=%.3f struct=%016llx vals=%016llx "
           "sum=%.17g det_ok=%d (sidecar written)\n",
           base, c_nnz, compression,
           (unsigned long long)sh, (unsigned long long)vh, vsum, det_ok);
    fflush(stdout);

    free(a_rp); free(a_ci); free(a_v);
    free(c_rp); free(c_ci); free(c_v);
    return (det_ok == 0) ? 5 : 0;
}
