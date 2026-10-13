#!/usr/bin/env bash

# Driver for the rv-sparse SpGEMM benchmarks on x86 (the Intel arm).
#
# A port of bench/riscv/run_bench.sh. Resume, validation, CSV layout, shuffle,
# retries and the integrity check are unchanged, so the raw and summary CSVs
# from both machines share one schema. What differs is the ISA probe, the
# build-to-march mapping, and pinning: OpenMP threads get one physical core
# each, never an SMT sibling.
#
# The experiment has the RISC-V arms except the m1/m4 LMUL rows:
#
#   baseline       scalar source, auto-vectorization off, 1 thread
#   autovec        the same scalar source, vectorizer on, 1 thread
#   intrinsic      AVX-512 port of the RVV LMUL=2 kernel, 1 thread
#   omp            scalar source with OpenMP, auto-vectorization off,
#                  threads 1 2 4 8 as on RISC-V
#
# Every build is disassembled before it runs. Arms that claim to be scalar
# (baseline, omp) fail the sweep if their kernel touches ymm, zmm or mask
# registers or does packed arithmetic; the intrinsic arm fails unless its
# kernel gathers, FMAs and scatters. --check-disasm does only this.
#
# Pinning, recorded per run in the CSV's pinning column: serial arms run under
# taskset -c PIN_CORE; an omp run with T threads is restricted with taskset to
# the first hardware thread of T consecutive physical cores starting at
# PIN_CORE, and runs with OMP_PLACES=cores OMP_PROC_BIND=close, so each thread
# owns a core and no thread shares one with an SMT sibling.
#
# Setup:
#   cp bench/intel/env.sh.example bench/intel/env.sh
#   # set CC
#   bash bench/intel/run_bench.sh --check
#
# Run from the repository root.
#
# Flags:
#   --check                 preflight only
#   --check-matrices        validate canonical CSR input matrices
#   --check-disasm          build every variant and run the disassembly
#                           checks, then exit without benchmarking
#   --dtype f32             restrict by dtype; repeatable or comma-separated
#   --kernels a,b           restrict by kernel name or arm
#   --baseline NAME         baseline kernel for batch composition
#   --runs N                timed runs per config (default 10)
#   --experiments PATH      alternate experiment table
#   -h | --help             show this message
#
# Common cases:
#   bash bench/intel/run_bench.sh
#   bash bench/intel/run_bench.sh --kernels avx512_f32
#   bash bench/intel/run_bench.sh --kernels omp
#
# Everything else is configured in bench/intel/env.sh.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="bench/intel"
cd "$REPO_ROOT" || {
    echo "!! cannot cd to $REPO_ROOT" >&2
    exit 1
}

die() {
    echo "!! $*" >&2
    exit 1
}

# Start a CMake build directory from empty. Reusing one is unsafe: when the
# compiler changes, CMake discards the cache and re-configures without the -D
# options given on the command line, so a variant can silently build without
# its flags or OpenMP. Only directories under build/intel/ are touched.
fresh_builddir() {
    case "$1" in
        build/intel/?*) ;;
        *) die "refusing to clear '$1': not under build/intel/" ;;
    esac

    rm -rf -- "$1" || die "could not clear $1"
}

# CLI
CHECK_ONLY=0
CHECK_MATRICES=0
CHECK_DISASM=0
DTYPE_FILTER=""
KERNEL_FILTER=""
RUNS_FLAG=""
EXP_FILE="$BENCH_DIR/experiments.tsv"

BASELINE_KERNEL="scalar_f32"

usage() {
    awk 'NR==1 { next }
         /^#/ {
             sub(/^# ?/, "")
             print
             next
         }
         /^[[:space:]]*$/ { next }
         { exit }' "${BASH_SOURCE[0]}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --check)
            CHECK_ONLY=1
            shift
            ;;
        --check-matrices)
            CHECK_MATRICES=1
            shift
            ;;
        --check-disasm)
            CHECK_DISASM=1
            shift
            ;;
        --dtype)
            [ $# -ge 2 ] || die "--dtype needs a value"
            DTYPE_FILTER="${DTYPE_FILTER}${DTYPE_FILTER:+,}$2"
            shift 2
            ;;
        --dtype=*)
            DTYPE_FILTER="${DTYPE_FILTER}${DTYPE_FILTER:+,}${1#*=}"
            shift
            ;;
        --kernels)
            [ $# -ge 2 ] || die "--kernels needs a value"
            KERNEL_FILTER="${KERNEL_FILTER}${KERNEL_FILTER:+,}$2"
            shift 2
            ;;
        --kernels=*)
            KERNEL_FILTER="${KERNEL_FILTER}${KERNEL_FILTER:+,}${1#*=}"
            shift
            ;;
        --baseline)
            [ $# -ge 2 ] || die "--baseline needs a value"
            BASELINE_KERNEL="$2"
            shift 2
            ;;
        --baseline=*)
            BASELINE_KERNEL="${1#*=}"
            shift
            ;;
        --runs)
            [ $# -ge 2 ] || die "--runs needs a value"
            RUNS_FLAG="$2"
            shift 2
            ;;
        --runs=*)
            RUNS_FLAG="${1#*=}"
            shift
            ;;
        --experiments)
            [ $# -ge 2 ] || die "--experiments needs a value"
            EXP_FILE="$2"
            shift 2
            ;;
        --experiments=*)
            EXP_FILE="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown flag: $1 (try --help)"
            ;;
    esac
done

ENV_FILE="${ENV_FILE:-$BENCH_DIR/env.sh}"

if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE" || die "failed to source $ENV_FILE"
else
    die "no $ENV_FILE on this box.

cp $BENCH_DIR/env.sh.example $BENCH_DIR/env.sh
\$EDITOR $BENCH_DIR/env.sh

Set CC to a GCC that supports -march=native with AVX-512, then re-run with --check."
fi

OUT_DIR="${OUT_DIR:-$BENCH_DIR/results}"
CSV="${CSV:-$OUT_DIR/spgemm_raw.csv}"
RUNS="${RUNS_FLAG:-${RUNS:-10}}"
WARMUP="${WARMUP:-3}"
PIN_CORE="${PIN_CORE:-0}"
CC="${CC:-}"
PERF="${PERF:-0}"
FORCE="${FORCE:-0}"
ONLY_MTX="${ONLY_MTX:-}"
RETRIES="${RETRIES:-1}"
SHUFFLE="${SHUFFLE:-1}"
SHUF_SEED="${SHUF_SEED:-$RANDOM}"
NO_GOVERNOR="${NO_GOVERNOR:-0}"
MACHINE="${MACHINE:-}"
OBJDUMP="${OBJDUMP:-objdump}"
export OBJDUMP

# The code version every row was measured with; -dirty when tracked files
# have uncommitted changes.
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"

if [ "$GIT_COMMIT" != "unknown" ] &&
   [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    GIT_COMMIT="$GIT_COMMIT-dirty"
fi

MATRICES=(
    wiki-Vote
    email-Enron
    p2p-Gnutella31
    cage12
    2cubes_sphere
    scircuit
    poisson3Da
    mario002
    cop20k_A
    offshore
    m133-b3
    filter3D
    ca-CondMat
    amazon0312
    web-Google
    roadNet-CA
    patents_main
    cit-Patents
    webbase-1M
)

CSV_HEADER="label,kernel,arm,build,march,cflags,cc_version,dtype,rows,cols,nnz_a,nnz_b,nnz_c,flops,op_mean,op_max,op_var,run,time_s,gops,correct,cycles,instructions,threads,stalls_backend,perf_mux,lmul,pinning,perf_note,tol_fail,max_rel_err,mean_rel_err,err_class,warmup,git_commit,team_size,bound_cpus"
CSV_NFIELDS=37

F_LABEL=1
F_KERNEL=2
F_BUILD=4
F_CFLAGS=6
F_CORRECT=21
F_THREADS=24
F_ERRCLASS=33

VALID_ARMS="baseline autovec intrinsic omp"
VALID_BUILDS="native"
VALID_DTYPES="f32"

# Vector kernels need AVX-512F. Their sources #error without it rather than
# silently falling back to scalar code.
vector_kernel() {
    case "$1" in
        avx512_*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

cflags_slug() {
    if [ "$1" = "-" ] || [ -z "$1" ]; then
        echo "default"
    else
        printf '%s' "$1" |
            tr -cs 'A-Za-z0-9' '_' |
            sed 's/^_//; s/_$//'
    fi
}

FREQ_DRIFT_WARN=0.01

# The build-to-march mapping lives here so the experiment table never encodes
# ISA flags directly.
build_march() {
    case "$1" in
        native)
            echo "-march=native"
            ;;
        *)
            die "unknown build '$1' (expected one of: $VALID_BUILDS)"
            ;;
    esac
}

in_list() {
    local n="$1"
    local h="$2"
    local x

    for x in $h; do
        [ "$x" = "$n" ] && return 0
    done

    return 1
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_uint "$RUNS" ||
    die "RUNS must be a non-negative integer, got '$RUNS'"

is_uint "$WARMUP" ||
    die "WARMUP must be a non-negative integer, got '$WARMUP'"

is_uint "$PIN_CORE" ||
    die "PIN_CORE must be a non-negative integer, got '$PIN_CORE'"

is_uint "$RETRIES" ||
    die "RETRIES must be a non-negative integer, got '$RETRIES'"

is_uint "$SHUF_SEED" ||
    die "SHUF_SEED must be a non-negative integer, got '$SHUF_SEED'"

[ "$RUNS" -ge 1 ] ||
    die "RUNS must be >= 1"

if [ "$RUNS" -lt 4 ]; then
    echo ">> WARNING: RUNS=$RUNS (<4) — analyze.py cannot bootstrap a speedup CI with so few runs. Use RUNS>=10 for a paper."
fi

# ============================================================================
# PREFLIGHT
# ============================================================================

CC_VERSION=""
PRESENT_MATRICES=()

# Matrix files live flat as matrices/<name>.mtx. The nested layout that
# SuiteSparse tarballs extract to is still accepted so existing checkouts and
# partially fetched trees keep working.
mtx_path() {
    local name="$1"

    if [ -f "matrices/$name.mtx" ]; then
        echo "matrices/$name.mtx"
    elif [ -f "matrices/$name/$name.mtx" ]; then
        echo "matrices/$name/$name.mtx"
    else
        return 1
    fi
}

# First hardware thread of every physical core, ascending. Taking only these
# keeps OpenMP threads off SMT siblings.
primary_cpus() {
    local f

    for f in /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list; do
        [ -r "$f" ] || continue

        # "0,86" or "0-1" -> 0
        sed -E 's/[,-].*//' "$f"
    done | sort -n -u
}

PRIMARY_CPUS=()

cores_from_pin() {
    local i
    local n=0
    local seen=0

    for i in "${PRIMARY_CPUS[@]}"; do
        [ "$i" = "$PIN_CORE" ] && seen=1
        [ "$seen" = "1" ] && n=$((n + 1))
    done

    echo "$n"
}

# Comma list of T primary CPUs starting at PIN_CORE. Fails if too few remain.
cpu_list() {
    local t="$1"
    local i
    local seen=0
    local -a out=()

    for i in "${PRIMARY_CPUS[@]}"; do
        [ "$i" = "$PIN_CORE" ] && seen=1
        [ "$seen" = "1" ] || continue

        out+=("$i")
        [ "${#out[@]}" -eq "$t" ] && break
    done

    [ "${#out[@]}" -eq "$t" ] || return 1

    local IFS=,
    echo "${out[*]}"
}

preflight() {
    echo ">> preflight"

    [ -n "$CC" ] ||
        die "CC is not set. Edit $ENV_FILE and set CC to a GCC for this machine."

    command -v "$CC" >/dev/null 2>&1 || [ -x "$CC" ] ||
        die "CC='$CC' is not executable. Fix CC in $ENV_FILE."

    echo "   CC = $CC"

    local missing=()
    local t

    for t in cmake python3 awk flock mktemp; do
        command -v "$t" >/dev/null 2>&1 || missing+=("$t")
    done

    if [ "${#missing[@]}" -gt 0 ]; then
        die "missing required tool(s): ${missing[*]}"
    fi

    local libgcc
    local libdir

    libgcc="$("$CC" -print-file-name=libgcc_s.so.1 2>/dev/null)"

    if [ -n "$libgcc" ] &&
       [ "$libgcc" != "libgcc_s.so.1" ] &&
       [ -e "$libgcc" ]; then

        libdir="$(cd "$(dirname "$libgcc")" && pwd)"

        export LIBRARY_PATH="${libdir}${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

        echo "   lib path (derived from CC, this process only): $libdir"
    else
        echo "   NOTE: could not derive a libgcc path from CC — assuming a system compiler."
    fi

    CC_VERSION="$("$CC" -dumpfullversion 2>/dev/null)"
    [ -n "$CC_VERSION" ] ||
        CC_VERSION="$("$CC" -dumpversion 2>/dev/null)"

    [ -n "$CC_VERSION" ] ||
        die "could not read a version from CC='$CC'"

    case "$CC_VERSION" in
        *,*)
            die "CC version '$CC_VERSION' contains a comma; it would corrupt the CSV"
            ;;
    esac

    echo "   cc_version = $CC_VERSION"

    # The Intel arm is GCC only: the no-vectorize flags in the experiment
    # table are GCC spelling, and clang ignores them with a warning.
    "$CC" --version 2>/dev/null | head -1 | grep -qi 'clang' &&
        die "CC='$CC' is clang. The scalar cells rely on GCC's -fno-tree-vectorize."

    CC_BANNER="$("$CC" --version 2>/dev/null | head -1)"
    echo "   cc_banner = $CC_BANNER"

    # What -march=native resolves to, for the record.
    NATIVE_MARCH="$("$CC" -march=native -Q --help=target 2>/dev/null |
        awk '$1 == "-march=" { print $2; exit }')"
    echo "   -march=native resolves to: ${NATIVE_MARCH:-unknown}"

    CPU_MODEL="$(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null)"
    [ -n "$MACHINE" ] || MACHINE="$CPU_MODEL"

    case "$MACHINE" in
        *,*)
            die "MACHINE '$MACHINE' contains a comma; it would corrupt the CSV"
            ;;
    esac

    echo "   machine = $MACHINE"

    # Compile and execute a real AVX-512 gather, FMA and scatter so the
    # preflight catches both toolchain support and a CPU without AVX-512.
    local tmpd
    local rc

    tmpd="$(mktemp -d)" ||
        die "mktemp -d failed"

    cat > "$tmpd/probe.c" <<'PROBE'
#include <immintrin.h>

int main(void) {
    float acc[32] = {0};
    int idx[16];

    for (int i = 0; i < 16; i++)
        idx[i] = (15 - i) * 2 * 4;

    __m512i voff = _mm512_loadu_si512(idx);
    __m512 v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0x7fff, voff, acc, 1);
    v = _mm512_fmadd_ps(_mm512_set1_ps(1.5f), _mm512_set1_ps(2.0f), v);
    _mm512_mask_i32scatter_ps(acc, 0x7fff, voff, v, 1);

    /* lane 0 hits acc[30], lane 15 is masked off and would hit acc[0] */
    return (acc[30] > 2.9f && acc[30] < 3.1f && acc[0] == 0.0f) ? 0 : 2;
}
PROBE

    if ! "$CC" -march=native -O2 \
        "$tmpd/probe.c" -o "$tmpd/probe" 2>"$tmpd/err"; then

        echo "---- compiler output ----" >&2
        cat "$tmpd/err" >&2
        echo "-------------------------" >&2

        rm -rf "$tmpd"

        die "CC='$CC' cannot compile an AVX-512 program with -march=native."
    fi

    "$tmpd/probe"
    rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "   avx512 probe: compiled and ran correctly"
    elif [ "$rc" -eq 2 ]; then
        rm -rf "$tmpd"

        die "avx512 probe compiled and ran but produced the WRONG result."
    else
        rm -rf "$tmpd"

        die "avx512 probe compiled but could not execute here (exit $rc).
This CPU most likely lacks AVX-512."
    fi

    rm -rf "$tmpd"

    local gov_path="/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_governor"

    if [ "$NO_GOVERNOR" = "1" ]; then
        echo "   NOTE: governor pin disabled (NO_GOVERNOR=1)"
    elif [ -w "$gov_path" ]; then
        echo "   cpufreq governor is writable — will pin to 'performance'"
    else
        echo "   WARNING: $gov_path not writable (need root). DVFS stays live — do NOT trust sub-5% effects."
    fi

    if [ "$PERF" = "1" ]; then
        if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
            local p
            p="$(cat /proc/sys/kernel/perf_event_paranoid)"

            if [ "$p" -le 2 ] 2>/dev/null; then
                echo "   perf counters available (perf_event_paranoid=$p)"
            else
                echo "   WARNING: perf_event_paranoid=$p (need <= 2). cycles/instructions will be blank."
            fi
        else
            echo "   WARNING: cannot read perf_event_paranoid; perf counters may be unavailable."
        fi
    fi

    PRIMARY_CPUS=($(primary_cpus))

    [ "${#PRIMARY_CPUS[@]}" -gt 0 ] ||
        die "could not read CPU topology from /sys/devices/system/cpu"

    in_list "$PIN_CORE" "${PRIMARY_CPUS[*]}" ||
        die "PIN_CORE=$PIN_CORE is an SMT sibling, not the first hardware thread of a core. Pick one of: ${PRIMARY_CPUS[*]:0:8} ..."

    echo "   physical cores: ${#PRIMARY_CPUS[@]} (one hardware thread each used for pinning)"

    local th_max
    th_max="$(awk -F'\t' '!/^[[:space:]]*#/ && $1 != "arm" && NF >= 6 { if ($6 > m) m = $6 } END { print m + 0 }' "$EXP_FILE" 2>/dev/null)"

    if [ -n "$th_max" ] && [ "$th_max" -gt 0 ]; then
        cpu_list "$th_max" >/dev/null ||
            die "the experiment table asks for $th_max threads, but only $(cores_from_pin) physical cores remain from PIN_CORE=$PIN_CORE"
        echo "   largest thread count $th_max -> cpus $(cpu_list "$th_max")"
    fi

    local isolated
    isolated="$(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo '')"

    if [ -z "$isolated" ] ||
       ! grep -qw "$PIN_CORE" <<< "${isolated//,/ }"; then

        echo "   WARNING: core $PIN_CORE is not isolated ('${isolated:-none}')."
        echo "            For sub-5% claims, boot with isolcpus=$PIN_CORE."
    fi

    local name
    local missing_m=()

    for name in "${MATRICES[@]}"; do
        if mtx_path "$name" >/dev/null; then
            PRESENT_MATRICES+=("$name")
        else
            missing_m+=("$name")
        fi
    done

    if [ "${#PRESENT_MATRICES[@]}" -eq 0 ]; then
        die "no .mtx files found under matrices/.

Fetch them first:
    cd matrices && bash getResources.sh && cd .."
    elif [ "${#missing_m[@]}" -gt 0 ]; then
        echo "   ${#PRESENT_MATRICES[@]}/${#MATRICES[@]} matrices present; skipping ${#missing_m[@]}: ${missing_m[*]}"
    else
        echo "   all ${#MATRICES[@]} matrices present"
    fi

    echo ">> preflight OK"
}

preflight

if [ "$CHECK_ONLY" = "1" ]; then
    echo ""
    echo ">> --check requested: preflight only, nothing was built and nothing was run."
    exit 0
fi

# ============================================================================
# --check-matrices
#
# The kernels assume canonical CSR input. Validate the dataset independently of the
# kernels before benchmarking.
# ============================================================================

if [ "$CHECK_MATRICES" = "1" ]; then
    if [ "${#PRESENT_MATRICES[@]}" -eq 0 ]; then
        die "no .mtx files present under matrices/ — nothing to check."
    fi

    b=native
    march="$(build_march "$b")"

    builddir="build/intel/csr_check"
    fresh_builddir "$builddir"

    echo ">> building csr_check ($march)"

    cmake -S . -B "$builddir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" \
        -DRVSP_ARCH_FLAGS="$march" \
        -DRVSP_ISA=intel \
        -DRVSP_BUILD_TESTS=OFF \
        -DRVSP_BUILD_EXAMPLES=OFF \
        >/dev/null ||
        die "cmake configure failed"

    cmake --build "$builddir" -j >/dev/null ||
        die "library build failed"

    "$CC" -Wall -Wextra -std=c11 \
        -Iinclude \
        -Itools/include \
        -Isrc/kernels/spgemm/common \
        $march \
        -O2 \
        bench/common/csr_check.c \
        -L"$builddir" \
        -lrvsparse_tools \
        -lrvsparse \
        -lm \
        -o $BENCH_DIR/csr_check ||
        die "csr_check build failed"

    MTX_PATHS=()

    for name in "${PRESENT_MATRICES[@]}"; do
        MTX_PATHS+=("$(mtx_path "$name")")
    done

    echo ">> checking ${#MTX_PATHS[@]} matrices"
    echo ""

    ./$BENCH_DIR/csr_check "${MTX_PATHS[@]}"
    rc=$?

    echo ""

    if [ "$rc" -eq 0 ]; then
        echo ">> all matrices satisfy the canonical-CSR precondition."
    else
        echo ">> PRECONDITION VIOLATED — do not run the kernels on this dataset until fixed."
    fi

    exit "$rc"
fi

# ============================================================================
# EXPERIMENT TABLE
# ============================================================================

EXP_ARM=()
EXP_KERNEL=()
EXP_DTYPE=()
EXP_BUILD=()
EXP_CFLAGS=()
EXP_THREADS=()

POOL_ARM=()
POOL_KERNEL=()
POOL_DTYPE=()
POOL_BUILD=()
POOL_CFLAGS=()
POOL_THREADS=()

ALL_KERNELS=()

load_experiments() {
    [ -f "$EXP_FILE" ] ||
        die "experiment table not found: $EXP_FILE"

    local arm kernel dtype build cflags
    local n=0
    local kept=0

    local -a want_dtypes=()

    if [ -n "$DTYPE_FILTER" ]; then
        IFS=',' read -ra want_dtypes <<< "$DTYPE_FILTER"

        local d

        for d in "${want_dtypes[@]}"; do
            in_list "$d" "$VALID_DTYPES" ||
                die "--dtype '$d' is not one of: $VALID_DTYPES"
        done
    fi

    # Validate the whole table before filtering so malformed rows are always
    # reported rather than hidden by the current selection.
    while IFS=$'\t' read -r arm kernel dtype build cflags threads; do
        n=$((n + 1))

        [ -n "$arm" ] &&
        [ -n "$kernel" ] &&
        [ -n "$dtype" ] &&
        [ -n "$build" ] ||
            die "$EXP_FILE row $n: expected 5 tab-separated fields"

        [ -n "$cflags" ] || cflags="-"
        [ -n "$threads" ] || threads=1

        is_uint "$threads" ||
            die "$EXP_FILE row $n: threads '$threads' is not a non-negative integer"

        case "$cflags" in
            *,*)
                die "$EXP_FILE row $n: cflags '$cflags' contains a comma; it would corrupt the CSV"
                ;;
        esac

        in_list "$arm" "$VALID_ARMS" ||
            die "$EXP_FILE row $n: arm '$arm' is not one of: $VALID_ARMS"

        in_list "$dtype" "$VALID_DTYPES" ||
            die "$EXP_FILE row $n: dtype '$dtype' is not one of: $VALID_DTYPES"

        in_list "$build" "$VALID_BUILDS" ||
            die "$EXP_FILE row $n: build '$build' is not one of: $VALID_BUILDS"


        # The dtype must appear as a suffix, or before a variant tag.
        case "$kernel" in
            *_"$dtype"|*_"$dtype"_*)
                ;;
            *)
                die "$EXP_FILE row $n: kernel '$kernel' does not contain '_$dtype' — dtype column and kernel name disagree."
                ;;
        esac

        in_list "$kernel" "${ALL_KERNELS[*]:-}" ||
            ALL_KERNELS+=("$kernel")

        if [ -n "$DTYPE_FILTER" ] &&
           ! in_list "$dtype" "${want_dtypes[*]}"; then
            continue
        fi

        POOL_ARM+=("$arm")
        POOL_KERNEL+=("$kernel")
        POOL_DTYPE+=("$dtype")
        POOL_BUILD+=("$build")
        POOL_CFLAGS+=("$cflags")
        POOL_THREADS+=("$threads")

        kept=$((kept + 1))

    done < <(
        awk -F'\t' '
            /^[[:space:]]*#/ { next }
            /^[[:space:]]*$/ { next }
            $1 == "arm" && $2 == "kernel" { next }
            { print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 }
        ' "$EXP_FILE"
    )

    [ "$n" -gt 0 ] ||
        die "$EXP_FILE has no experiment rows"

    [ "$kept" -gt 0 ] ||
        die "no experiment rows match --dtype '$DTYPE_FILTER'"

    local -a want_kernels=()

    if [ -n "$KERNEL_FILTER" ]; then
        IFS=',' read -ra want_kernels <<< "$KERNEL_FILTER"

        local k

        for k in "${want_kernels[@]}"; do
            if in_list "$k" "${ALL_KERNELS[*]}" ||
               in_list "$k" "$VALID_ARMS"; then
                continue
            fi

            die "--kernels '$k' is neither a kernel in $EXP_FILE nor an arm.

kernels: ${ALL_KERNELS[*]}
arms:    $VALID_ARMS"
        done
    fi

    local i

    for ((i = 0; i < ${#POOL_ARM[@]}; i++)); do
        if [ -n "$KERNEL_FILTER" ]; then
            if ! in_list "${POOL_KERNEL[i]}" "${want_kernels[*]}" &&
               ! in_list "${POOL_ARM[i]}" "${want_kernels[*]}"; then
                continue
            fi
        fi

        EXP_ARM+=("${POOL_ARM[i]}")
        EXP_KERNEL+=("${POOL_KERNEL[i]}")
        EXP_DTYPE+=("${POOL_DTYPE[i]}")
        EXP_BUILD+=("${POOL_BUILD[i]}")
        EXP_CFLAGS+=("${POOL_CFLAGS[i]}")
        EXP_THREADS+=("${POOL_THREADS[i]}")
    done

    [ "${#EXP_ARM[@]}" -gt 0 ] ||
        die "no experiment rows match --kernels '$KERNEL_FILTER'"

    # Filtered batches must contain the requested baseline so speedups can be
    # computed without requiring a second benchmark invocation.
    if [ -n "$KERNEL_FILTER" ] &&
       ! in_list "$BASELINE_KERNEL" "${EXP_KERNEL[*]}"; then

        local -a add=()
        local baseline_idx=-1

        for ((i = 0; i < ${#POOL_ARM[@]}; i++)); do
            [ "${POOL_KERNEL[i]}" = "$BASELINE_KERNEL" ] || continue

            if [ "${POOL_ARM[i]}" = "baseline" ]; then
                baseline_idx="$i"
                break
            fi
        done

        if [ "$baseline_idx" -ge 0 ]; then
            add=("$baseline_idx")
        else
            for ((i = 0; i < ${#POOL_ARM[@]}; i++)); do
                [ "${POOL_KERNEL[i]}" = "$BASELINE_KERNEL" ] || continue
                add+=("$i")
            done
        fi

        [ "${#add[@]}" -gt 0 ] ||
            die "--baseline '$BASELINE_KERNEL' has no row in $EXP_FILE${DTYPE_FILTER:+ for dtype '$DTYPE_FILTER'}."

        local j

        for j in "${add[@]}"; do
            EXP_ARM+=("${POOL_ARM[j]}")
            EXP_KERNEL+=("${POOL_KERNEL[j]}")
            EXP_DTYPE+=("${POOL_DTYPE[j]}")
            EXP_BUILD+=("${POOL_BUILD[j]}")
            EXP_CFLAGS+=("${POOL_CFLAGS[j]}")
            EXP_THREADS+=("${POOL_THREADS[j]}")
        done

        echo ">> baseline '$BASELINE_KERNEL' was not in --kernels; added ${#add[@]} row(s) for it."
    fi

    kept="${#EXP_ARM[@]}"

    echo ">> experiment table: $EXP_FILE — $kept of $n rows selected${DTYPE_FILTER:+ (dtype: $DTYPE_FILTER)}${KERNEL_FILTER:+ (kernels: $KERNEL_FILTER)}"

    for ((i = 0; i < ${#EXP_ARM[@]}; i++)); do
        printf '     %-14s %-17s %-4s %-6s t%-3s %s\n' \
            "${EXP_ARM[i]}" \
            "${EXP_KERNEL[i]}" \
            "${EXP_DTYPE[i]}" \
            "${EXP_BUILD[i]}" \
            "${EXP_THREADS[i]}" \
            "${EXP_CFLAGS[i]}"
    done
}

load_experiments

# Compile-time tunables require separate build artifacts for every unique
# (build,cflags) pair.
VARIANT_BUILD=()
VARIANT_CFLAGS=()
VARIANT_TAG=()

for ((i = 0; i < ${#EXP_BUILD[@]}; i++)); do
    tag="${EXP_BUILD[i]}__$(cflags_slug "${EXP_CFLAGS[i]}")"

    if in_list "$tag" "${VARIANT_TAG[*]:-}"; then
        continue
    fi

    VARIANT_BUILD+=("${EXP_BUILD[i]}")
    VARIANT_CFLAGS+=("${EXP_CFLAGS[i]}")
    VARIANT_TAG+=("$tag")
done

echo ">> binaries to build: ${#VARIANT_TAG[@]}"

for ((i = 0; i < ${#VARIANT_TAG[@]}; i++)); do
    printf '     %s/bench_%-28s march=%s  cflags=%s\n' "$BENCH_DIR" \
        "${VARIANT_TAG[i]}" \
        "$(build_march "${VARIANT_BUILD[i]}")" \
        "${VARIANT_CFLAGS[i]}"
done

exp_tag() {
    echo "${EXP_BUILD[$1]}__$(cflags_slug "${EXP_CFLAGS[$1]}")"
}

# ============================================================================
# LOCK / CLEANUP
# ============================================================================

mkdir -p "$OUT_DIR" ||
    die "cannot create $OUT_DIR"

LOCKFILE="$OUT_DIR/.run_bench.lock"

exec 200>"$LOCKFILE"

if ! flock -n 200; then
    die "another run_bench.sh is already running against $OUT_DIR"
fi

GOV_PATH="/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_governor"
FREQ_PATH="/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_cur_freq"
ORIG_GOV=""

read_freq() {
    cat "$FREQ_PATH" 2>/dev/null || echo 0
}

set_governor_performance() {
    [ "$NO_GOVERNOR" = "1" ] && {
        echo ">> governor pin skipped (NO_GOVERNOR=1)"
        return
    }

    if [ -w "$GOV_PATH" ]; then
        ORIG_GOV="$(cat "$GOV_PATH" 2>/dev/null || echo '')"

        if echo performance > "$GOV_PATH" 2>/dev/null; then
            echo ">> cpufreq governor set to 'performance' on core $PIN_CORE (was '${ORIG_GOV:-unknown}')"
        else
            echo ">> WARNING: could not write $GOV_PATH — DVFS is live."
            ORIG_GOV=""
        fi
    else
        echo ">> WARNING: $GOV_PATH not writable (need root). DVFS is live."
    fi
}

restore_governor() {
    if [ -n "$ORIG_GOV" ] && [ -w "$GOV_PATH" ]; then
        echo "$ORIG_GOV" > "$GOV_PATH" 2>/dev/null &&
            echo ">> restored governor to '$ORIG_GOV'"
    fi
}

cleanup() {
    restore_governor
}

trap cleanup EXIT

on_interrupt() {
    echo ""
    echo "!! interrupted — CSV writes are atomic, so $CSV is not corrupted."
    echo "!! re-run this script to resume."
    exit 130
}

trap on_interrupt INT TERM

# ============================================================================
# BUILD
# ============================================================================

PERF_FLAG=""

if [ "$PERF" = "1" ]; then
    PERF_FLAG="-DUSE_PERF"
    echo ">> perf counters ENABLED (cycles, instructions — user-space only)"
fi

build_all() {
    local b="$1"
    local cflags="$2"
    local tag="$3"

    local march
    local extra
    local builddir

    march="$(build_march "$b")"
    extra=""

    [ "$cflags" != "-" ] && extra="$cflags"

    # The OpenMP kernel source is gated in CMakeLists.txt. A row asks for it
    # with -fopenmp in cflags, which also gives it its own build variant so the
    # baseline is never rebuilt with OpenMP enabled.
    local openmp=OFF

    case "$cflags" in
        *-fopenmp*) openmp=ON ;;
    esac

    # One CMake build tree per (build, cflags) variant.
    builddir="build/intel/$tag"
    fresh_builddir "$builddir"

    echo ">> [$tag] configuring ($march${extra:+ $extra})"

    cmake -S . -B "$builddir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" \
        -DRVSP_ARCH_FLAGS="$march $extra" \
        -DRVSP_ISA=intel \
        -DRVSP_OPENMP="$openmp" \
        -DRVSP_BUILD_TESTS=OFF \
        -DRVSP_BUILD_EXAMPLES=OFF \
        >/dev/null ||
        die "[$tag] cmake configure failed"

    echo ">> [$tag] building librvsparse.a"

    cmake --build "$builddir" -j >/dev/null ||
        die "[$tag] library build failed"

    local f

    for f in "$builddir/librvsparse.a" "$builddir/librvsparse_tools.a"; do
        [ -f "$f" ] ||
            die "[$tag] expected build artifact missing: $f"
    done

    echo ">> [$tag] compiling bench.c -> $BENCH_DIR/bench_$tag"

    "$CC" \
        -Wall -Wextra \
        -std=c11 \
        -Iinclude \
        -Itools/include \
        -Isrc/kernels/spgemm/common \
        $march \
        $extra \
        -O3 \
        $PERF_FLAG \
        bench/common/bench.c \
        -L"$builddir" \
        -lrvsparse_tools \
        -lrvsparse \
        -lm \
        -o "$BENCH_DIR/bench_$tag" ||
        die "[$tag] bench build failed"

    [ -x "$BENCH_DIR/bench_$tag" ] ||
        die "$BENCH_DIR/bench_$tag was not produced or is not executable"

    # The exact command CMake used for the kernel sources of this variant, for
    # the flags column of the summary.
    local flags
    flags="$(python3 -c '
import json, sys
for e in json.load(open(sys.argv[1])):
    if e["file"].endswith("accum_scalar_f32.c"):
        args = e.get("arguments") or e["command"].split()
        out = [a for a in args[1:] if a.startswith("-") and a not in ("-o", "-c")]
        print(" ".join(out))
        break
' "$builddir/compile_commands.json" 2>/dev/null)"

    [ -n "$flags" ] ||
        die "[$tag] could not read compile flags from $builddir/compile_commands.json"

    VARIANT_FLAGS_LINES+=("$b"$'\t'"$cflags"$'\t'"$flags")
    echo "   [$tag] kernel flags: $flags"

    check_variant_disasm "$tag"
}

# Disassembly checks for every experiment row built into this variant, by arm:
#   baseline, omp   kernel must contain no vector instruction
#   intrinsic       kernel must contain the gather, FMA and scatter
#   autovec         vector content is reported, never fails
check_variant_disasm() {
    local tag="$1"
    local bin="$BENCH_DIR/bench_$tag"
    local i
    local done_keys=""
    local key
    local prefix
    local rc

    for ((i = 0; i < ${#EXP_ARM[@]}; i++)); do
        [ "$(exp_tag "$i")" = "$tag" ] || continue

        key="${EXP_ARM[i]}:${EXP_KERNEL[i]}"
        in_list "$key" "$done_keys" && continue
        done_keys="$done_keys $key"

        prefix="rvsp_spgemm_${EXP_KERNEL[i]}"

        case "${EXP_ARM[i]}" in
            baseline|omp)
                bash bench/common/check_disasm.sh intel scalar "$bin" "$prefix"
                rc=$?
                ;;
            intrinsic)
                bash bench/common/check_disasm.sh intel vector "$bin" "$prefix" \
                    vgatherdps vscatterdps vfmadd
                rc=$?
                ;;
            autovec)
                bash bench/common/check_disasm.sh intel report "$bin" "$prefix"
                rc=$?
                ;;
            *)
                rc=0
                ;;
        esac

        [ "$rc" -eq 0 ] ||
            die "[$tag] disassembly check failed for ${EXP_ARM[i]} ${EXP_KERNEL[i]} (rc=$rc)"
    done
}

VARIANT_FLAGS_LINES=()

command -v taskset >/dev/null 2>&1 ||
    die "taskset not found; the Intel arm requires pinning"

RUNNER="taskset -c $PIN_CORE"
echo ">> pinning: serial arms on cpu $PIN_CORE; omp with T threads on T physical cores from cpu $PIN_CORE (OMP_PLACES=cores OMP_PROC_BIND=close)"

# "0,1,2,3,8" -> "0-3+8", for the CSV, where commas are field separators.
cpu_ranges() {
    awk -F, '{
        out = ""
        start = prev = $1
        for (i = 2; i <= NF + 1; i++) {
            if (i <= NF && $i == prev + 1) { prev = $i; continue }
            out = out (out == "" ? "" : "+") (start == prev ? start : start "-" prev)
            if (i <= NF) start = prev = $i
        }
        print out
    }' <<< "$1"
}

# Validate each binary before the real sweep using a tiny configuration.
smoke_test() {
    local b="$1"
    local cflags="$2"
    local tag="$3"

    local march
    local tmp
    local ok

    march="$(build_march "$b")"

    echo ">> [$tag] smoke test"

    tmp="$(mktemp "$OUT_DIR/.smoke_XXXXXX")"

    if ! $RUNNER "./$BENCH_DIR/bench_$tag" \
        --kernel scalar_f32 \
        --gen 64 64 0.05 1 \
        --runs 2 \
        --warmup 1 \
        --label smoke \
        --arm baseline \
        --build "$b" \
        --march "$march" \
        --cflags "$cflags" \
        --cc-version "$CC_VERSION" \
        > "$tmp" 2>&1; then

        echo "---- smoke output ----"
        cat "$tmp"
        echo "----------------------"

        rm -f "$tmp"

        die "[$tag] smoke test failed to run"
    fi

    ok="$(
        awk -F, \
            -v nf="$CSV_NFIELDS" \
            -v b="$b" \
            -v cf="$cflags" \
            -v fb="$F_BUILD" \
            -v fk="$F_KERNEL" \
            -v fc="$F_CORRECT" \
            -v fcf="$F_CFLAGS" '
            NF == nf &&
            $fk == "scalar_f32" &&
            $fb == b &&
            $fcf == cf &&
            $fc == 1
            ' "$tmp" |
        wc -l |
        tr -d ' '
    )"

    if [ "$ok" -ne 2 ]; then
        echo "---- smoke output ----"
        cat "$tmp"
        echo "----------------------"

        rm -f "$tmp"

        die "[$tag] smoke test produced $ok/2 valid+correct rows"
    fi

    rm -f "$tmp"

    echo "   [$tag] smoke test passed"
}

for ((i = 0; i < ${#VARIANT_TAG[@]}; i++)); do
    build_all \
        "${VARIANT_BUILD[i]}" \
        "${VARIANT_CFLAGS[i]}" \
        "${VARIANT_TAG[i]}"

    [ "$CHECK_DISASM" = "1" ] && continue

    smoke_test \
        "${VARIANT_BUILD[i]}" \
        "${VARIANT_CFLAGS[i]}" \
        "${VARIANT_TAG[i]}"
done

if [ "$CHECK_DISASM" = "1" ]; then
    echo ""
    echo ">> --check-disasm: every variant built and passed its disassembly checks; nothing was run."
    exit 0
fi

# Record the flags of every variant so the summary can carry them.
{
    printf 'build\tcflags\tflags\n'
    printf '%s\n' "${VARIANT_FLAGS_LINES[@]}"
} > "$OUT_DIR/variants.tsv"

set_governor_performance

# ============================================================================
# CSV SETUP
# ============================================================================

if [ ! -f "$CSV" ]; then
    echo "$CSV_HEADER" > "$CSV" ||
        die "cannot write $CSV"

    echo ">> created $CSV"
else
    existing_header="$(head -1 "$CSV")"

    if [ "$existing_header" != "$CSV_HEADER" ]; then
        die "existing $CSV has a different header than expected.
Raw CSVs from before the pinning, perf_note and error columns cannot be appended to;
start a fresh OUT_DIR."
    fi

    echo ">> appending to existing $CSV ($(( $(wc -l < "$CSV") - 1 )) rows)"

    # Groups are complete only when they contain exactly RUNS rows. A changed
    # RUNS value therefore invalidates existing partial groups.
    stale_groups="$(
        awk -F, \
            -v r="$RUNS" \
            -v fb="$F_BUILD" \
            -v fc="$F_CFLAGS" \
            -v fk="$F_KERNEL" \
            -v fl="$F_LABEL" \
            -v fth="$F_THREADS" '
            NR > 1 {
                c[$fb FS $fc FS $fk FS $fl FS $fth]++
            }
            END {
                n = 0
                for (g in c)
                    if (c[g] != r)
                        n++
                print n
            }
        ' "$CSV"
    )"

    if [ "${stale_groups:-0}" -gt 0 ]; then
        echo ">> NOTE: $stale_groups existing group(s) do not have exactly RUNS=$RUNS rows."
        echo "         Those will be purged and re-run."
    fi
fi

{
    echo "--- run $(date -Iseconds) ---"
    echo "host: $(hostname)"
    echo "uname: $(uname -srm)"
    echo "cc: $CC"
    echo "cc_version: $CC_VERSION"
    echo "git_commit: $GIT_COMMIT"
    echo "cc_banner: $CC_BANNER"
    echo "machine: $MACHINE"
    echo "cpu_model: $CPU_MODEL"
    echo "native_march: ${NATIVE_MARCH:-unknown}"
    echo "smt: $(cat /sys/devices/system/cpu/smt/active 2>/dev/null || echo n/a) (1 = siblings online; runs use one thread per core)"
    echo "physical_cores: ${#PRIMARY_CPUS[@]}"
    echo "pinning: serial arms taskset -c $PIN_CORE; omp arm taskset to T primary cpus from $PIN_CORE, OMP_PLACES=cores OMP_PROC_BIND=close"
    echo "objdump: $OBJDUMP ($("$OBJDUMP" --version 2>/dev/null | head -1))"
    echo "perf: PERF=$PERF perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"

    for ((i = 0; i < ${#VARIANT_TAG[@]}; i++)); do
        echo "variant[${VARIANT_TAG[i]}]: $(build_march "${VARIANT_BUILD[i]}") ${VARIANT_CFLAGS[i]} -O3 $PERF_FLAG"
        echo "variant_flags[${VARIANT_TAG[i]}]: $(printf '%s\n' "${VARIANT_FLAGS_LINES[@]}" | awk -F'\t' -v b="${VARIANT_BUILD[i]}" -v c="${VARIANT_CFLAGS[i]}" '$1 == b && $2 == c { print $3 }')"
    done

    echo "experiments: $EXP_FILE (${#EXP_ARM[@]} rows selected${DTYPE_FILTER:+, dtype=$DTYPE_FILTER})"
    echo "csv: $CSV"
    echo "runs: $RUNS  warmup: $WARMUP  pin_core: $PIN_CORE  retries: $RETRIES"
    echo "shuffle: $SHUFFLE  shuf_seed: $SHUF_SEED"
    echo "governor_requested: performance  governor_orig: ${ORIG_GOV:-not_changed}"
    echo "governor_now: $(cat "$GOV_PATH" 2>/dev/null || echo n/a)"
    echo "core_isolated: $(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo none)"
    echo "freq_khz: $(read_freq)"
    echo "isa: x86_64 $(grep -o -m1 -w 'avx512f' /proc/cpuinfo 2>/dev/null | head -1)"
    echo ""
} | tee -a "$OUT_DIR/env_log.txt" > "$OUT_DIR/env.txt"

echo ">> environment appended to $OUT_DIR/env_log.txt"

# ============================================================================
# HELPERS
#
# Resume key: (label, kernel, build, cflags). Build and cflags distinguish
# otherwise identical kernels compiled into different binaries.
# ============================================================================

atomic_append_csv() {
    local rows_file="$1"

    [ -s "$rows_file" ] || return 0

    local tmp

    tmp="$(mktemp "$OUT_DIR/.csv_XXXXXX")" ||
        die "mktemp failed"

    if cat "$CSV" "$rows_file" > "$tmp" &&
       mv -f "$tmp" "$CSV"; then
        :
    else
        rm -f "$tmp"
        die "atomic CSV write failed"
    fi
}

config_count() {
    awk -F, \
        -v l="$1" \
        -v k="$2" \
        -v b="$3" \
        -v cf="$4" \
        -v th="$5" \
        -v fl="$F_LABEL" \
        -v fk="$F_KERNEL" \
        -v fb="$F_BUILD" \
        -v fcf="$F_CFLAGS" \
        -v fth="$F_THREADS" '
        NR > 1 &&
        $fl == l &&
        $fk == k &&
        $fb == b &&
        $fcf == cf &&
        $fth == th
        ' "$CSV" |
    wc -l |
    tr -d ' '
}

purge_config() {
    local tmp

    tmp="$(mktemp "$OUT_DIR/.csv_XXXXXX")" ||
        die "mktemp failed"

    if awk -F, \
        -v l="$1" \
        -v k="$2" \
        -v b="$3" \
        -v cf="$4" \
        -v th="$5" \
        -v fl="$F_LABEL" \
        -v fk="$F_KERNEL" \
        -v fb="$F_BUILD" \
        -v fcf="$F_CFLAGS" \
        -v fth="$F_THREADS" '
        NR == 1 ||
        !($fl == l && $fk == k && $fb == b && $fcf == cf && $fth == th)
        ' "$CSV" > "$tmp" &&
        mv -f "$tmp" "$CSV"; then
        :
    else
        rm -f "$tmp"
        die "purge failed for $1/$2/$3/$4"
    fi
}

filter_valid_rows() {
    awk -F, \
        -v l="$2" \
        -v k="$3" \
        -v b="$4" \
        -v cf="$5" \
        -v nf="$CSV_NFIELDS" \
        -v fl="$F_LABEL" \
        -v fk="$F_KERNEL" \
        -v fb="$F_BUILD" \
        -v fcf="$F_CFLAGS" \
        -v tag="$2/$3/$4/$5" '
        NF == nf &&
        $fl == l &&
        $fk == k &&
        $fb == b &&
        $fcf == cf {
            print
            next
        }
        {
            bad++
        }
        END {
            if (bad > 0)
                print "   !! dropped " bad " malformed/mismatched row(s) for " tag > "/dev/stderr"
        }
        ' "$1" > "$6"
}

# Shuffle order per matrix to reduce systematic thermal/order effects. The
# seed is fixed once so the order is reproducible.
seeded_shuffle_idx() {
    local n="$1"
    local i
    local j
    local tmp

    local -a idx=()

    for ((i = 0; i < n; i++)); do
        idx+=("$i")
    done

    for ((i = n - 1; i > 0; i--)); do
        j=$((RANDOM % (i + 1)))

        tmp="${idx[i]}"
        idx[i]="${idx[j]}"
        idx[j]="$tmp"
    done

    printf '%s\n' "${idx[@]}"
}

experiment_order() {
    local n="${#EXP_ARM[@]}"
    local i

    if [ "$SHUFFLE" = "1" ]; then
        seeded_shuffle_idx "$n"
    else
        for ((i = 0; i < n; i++)); do
            echo "$i"
        done
    fi
}

FAILED_CONFIGS=()
COMPLETED_COUNT=0
ATTEMPTED_COUNT=0

# ============================================================================
# RUN
# ============================================================================

run_config() {
    local ei="$1"
    local label="$2"

    shift 2

    local arm="${EXP_ARM[ei]}"
    local k="${EXP_KERNEL[ei]}"
    local b="${EXP_BUILD[ei]}"
    local cf="${EXP_CFLAGS[ei]}"
    local th="${EXP_THREADS[ei]}"

    local tag
    tag="$(exp_tag "$ei")"

    local march
    march="$(build_march "$b")"

    # Serial arms run on PIN_CORE. An omp run gets T physical cores and OpenMP
    # binds thread i to core i of them.
    local -a pin_env=()
    local pin_runner="$RUNNER"
    local pin_desc="taskset=$PIN_CORE"

    if [ "$arm" = "omp" ]; then
        local cpus
        cpus="$(cpu_list "$th")" ||
            die "not enough physical cores from PIN_CORE=$PIN_CORE for $th threads"

        pin_env=(OMP_PLACES=cores OMP_PROC_BIND=close)
        pin_runner="taskset -c $cpus"
        pin_desc="taskset=$(cpu_ranges "$cpus") places=cores bind=close"
    fi

    local desc="$k@$b"

    [ "$cf" != "-" ] && desc="$k@$b[$cf]"
    [ "$th" != "1" ] && desc="$desc t$th"

    ATTEMPTED_COUNT=$((ATTEMPTED_COUNT + 1))

    local have
    have="$(config_count "$label" "$k" "$b" "$cf" "$th")"

    if [ "$FORCE" != "1" ] && [ "$have" -eq "$RUNS" ]; then
        echo "   $label  $desc ($arm)  [have]"
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        return 0
    fi

    if [ "$have" -gt 0 ]; then
        echo "   $label  $desc ($arm)  [partial: $have/$RUNS present — purging and re-running clean]"
        purge_config "$label" "$k" "$b" "$cf" "$th"
    else
        echo "   $label  $desc ($arm)"
    fi

    local freq_before
    freq_before="$(read_freq)"

    local attempt=0
    local ok=0

    while [ "$attempt" -le "$RETRIES" ]; do
        attempt=$((attempt + 1))

        local tmp
        local valid
        local n
        local rc

        tmp="$(mktemp "$OUT_DIR/.run_XXXXXX")"
        valid="$(mktemp "$OUT_DIR/.valid_XXXXXX")"

        if env OMP_NUM_THREADS="$th" "${pin_env[@]}" $pin_runner "./$BENCH_DIR/bench_$tag" \
            --kernel "$k" \
            "$@" \
            --runs "$RUNS" \
            --warmup "$WARMUP" \
            --label "$label" \
            --arm "$arm" \
            --build "$b" \
            --march "$march" \
            --cflags "$cf" \
            --threads "$th" \
            --cc-version "$CC_VERSION" \
            --pinning "$pin_desc" \
            --git-commit "$GIT_COMMIT" \
            --verify-pinning \
            > "$tmp" \
            2>"$tmp.err"; then
            :
        else
            rc=$?

            # Exit 3 is a failed precondition (input not canonical, no or bad
            # reference, pinning not as intended). Retrying cannot fix it, and
            # carrying on would measure something other than intended.
            if [ "$rc" -eq 3 ]; then
                echo "   !! PRECONDITION FAILED: $label / $desc ($arm)" >&2
                sed 's/^/      /' "$tmp.err" >&2
                rm -f "$tmp" "$tmp.err" "$valid"
                die "stopping: $label / $desc failed a precondition (see above). Nothing appended to $CSV."
            fi

            echo "   !! failed (rc=$rc) — $label / $desc (attempt $attempt/$((RETRIES + 1))): $(head -1 "$tmp.err" 2>/dev/null)"
        fi

        filter_valid_rows \
            "$tmp" \
            "$label" \
            "$k" \
            "$b" \
            "$cf" \
            "$valid"

        n="$(wc -l < "$valid" | tr -d ' ')"

        # A structural, beyond-rounding or failed result is a wrong answer.
        # Keep it out of the CSV, set it aside, and stop the sweep.
        local wrong
        wrong="$(awk -F, -v f="$F_ERRCLASS" \
            '$f == "structural" || $f == "value" || $f == "failed" { print $f }' "$valid" |
            sort | uniq -c | tr '\n' ' ')"

        if [ -n "$wrong" ]; then
            [ -f "$OUT_DIR/rejected.csv" ] || echo "$CSV_HEADER" > "$OUT_DIR/rejected.csv"
            cat "$valid" >> "$OUT_DIR/rejected.csv"
            echo "   !! WRONG ANSWER: $label / $desc ($arm): $wrong" >&2
            echo "   !! the result differs from the scalar reference by more than rounding" >&2
            rm -f "$tmp" "$tmp.err" "$valid"
            die "stopping: $label / $desc is not rounding error. Rows saved to $OUT_DIR/rejected.csv, nothing appended to $CSV."
        fi

        if [ "$n" -eq "$RUNS" ]; then
            atomic_append_csv "$valid"

            rm -f "$tmp" "$tmp.err" "$valid"

            ok=1
            break
        fi

        echo "   .. attempt $attempt/$((RETRIES + 1)) produced $n/$RUNS valid rows"

        rm -f "$tmp" "$tmp.err" "$valid"
    done

    local freq_after
    freq_after="$(read_freq)"

    if [ "$freq_before" -gt 0 ] 2>/dev/null &&
       [ "$freq_after" -gt 0 ] 2>/dev/null; then

        local drift

        drift="$(
            awk \
                -v a="$freq_before" \
                -v b2="$freq_after" '
                BEGIN {
                    d = (a > b2 ? a - b2 : b2 - a)
                    printf "%.4f", (a > 0 ? d / a : 0)
                }'
        )"

        if awk \
            -v d="$drift" \
            -v w="$FREQ_DRIFT_WARN" \
            'BEGIN { exit !(d > w) }'; then

            echo "   !! FREQ DRIFT $label / $desc: ${freq_before}kHz -> ${freq_after}kHz ($(awk -v d="$drift" 'BEGIN {printf "%.1f", d*100}')%) — number may be DVFS-tainted"
        fi
    fi

    if [ "$ok" -eq 1 ]; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    else
        echo "   !! GIVING UP on $label / $desc after $((RETRIES + 1)) attempt(s) — no rows written for this config"
        FAILED_CONFIGS+=("$label | $desc")
    fi
}

RANDOM="$SHUF_SEED"

echo ">> experiment order per matrix: $(
    [ "$SHUFFLE" = "1" ] &&
        echo "shuffled (seed $SHUF_SEED)" ||
        echo "fixed"
)"

# ============================================================================
# SWEEP
# ============================================================================

echo ">> matrix sweep (C = A*A)"

if [ -n "$ONLY_MTX" ]; then
    MATRIX_LIST=("$ONLY_MTX")

    mtx_path "$ONLY_MTX" >/dev/null ||
        die "ONLY_MTX=$ONLY_MTX but matrices/$ONLY_MTX.mtx does not exist"
else
    MATRIX_LIST=("${PRESENT_MATRICES[@]}")
fi

for name in "${MATRIX_LIST[@]}"; do
    mtx="$(mtx_path "$name")"

    mapfile -t ORDER < <(experiment_order)

    echo "   [$name] order: $(
        for i in "${ORDER[@]}"; do
            printf '%s@%s ' "${EXP_KERNEL[i]}" "$(exp_tag "$i")"
        done
    )"

    for i in "${ORDER[@]}"; do
        run_config "$i" "$name" \
            --mtx-sq "$mtx"
    done
done

# ============================================================================
# FINAL INTEGRITY CHECK
# ============================================================================

echo ""
echo ">> integrity check: re-scanning $CSV"

BAD_GROUPS=0

while IFS= read -r grp; do
    [ -z "$grp" ] && continue

    IFS=$'\t' read -r lbl kern bld cfl th <<< "$grp"

    n="$(config_count "$lbl" "$kern" "$bld" "$cfl" "$th")"

    if [ "$n" -ne "$RUNS" ]; then
        echo "   !! $lbl / $kern@$bld[$cfl] has $n/$RUNS rows (not exactly RUNS)"
        BAD_GROUPS=$((BAD_GROUPS + 1))
    fi

done < <(
    awk -F, \
        -v fl="$F_LABEL" \
        -v fk="$F_KERNEL" \
        -v fb="$F_BUILD" \
        -v fcf="$F_CFLAGS" \
        -v fth="$F_THREADS" '
        NR > 1 {
            print $fl "\t" $fk "\t" $fb "\t" $fcf "\t" $fth
        }
        ' "$CSV" |
    sort -u
)

echo ""
echo ">> done. $CSV has $(( $(wc -l < "$CSV") - 1 )) rows"
echo ">> configs attempted this run: $ATTEMPTED_COUNT, completed: $COMPLETED_COUNT, failed: ${#FAILED_CONFIGS[@]}"

if [ "${#FAILED_CONFIGS[@]}" -gt 0 ]; then
    echo ">> FAILED configs (no rows written, will retry next run):"

    for f in "${FAILED_CONFIGS[@]}"; do
        echo "     $f"
    done
fi

if [ "$BAD_GROUPS" -gt 0 ]; then
    echo ">> WARNING: $BAD_GROUPS group(s) in the CSV do not have exactly $RUNS rows."
else
    echo ">> integrity check passed: every group has exactly $RUNS rows."
fi

SUMMARY="$OUT_DIR/summary.csv"
TAGGED="$OUT_DIR/summary_tagged.csv"

echo ">> summarising -> $SUMMARY"

python3 bench/common/analyze.py "$CSV" --csv-out "$SUMMARY" --baseline "$BASELINE_KERNEL" > "$OUT_DIR/analyze.txt" ||
    echo "!! analyze.py failed; see $OUT_DIR/analyze.txt"

python3 bench/common/tag_summary.py "$SUMMARY" --out "$TAGGED" \
    --machine "$MACHINE" \
    --isa "x86_64 AVX-512" \
    --compiler "$CC_BANNER" \
    --variants "$OUT_DIR/variants.tsv" ||
    echo "!! tag_summary.py failed"

echo ">> summary with machine, isa, compiler and flags columns: $TAGGED"

[ "${#FAILED_CONFIGS[@]}" -eq 0 ] || exit 1

exit 0