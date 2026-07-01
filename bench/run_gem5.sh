#!/usr/bin/env bash
#
# bench/run_gem5.sh — run CSR SpGEMM binaries under gem5 (cycle-accurate, RISCV
# SE mode) and produce ONE merged CSV with everything for the eval table.
#
# Per run, two streams are captured and joined:
#   - harness stdout  -> ops, bytes(analytical), AI, C_nnz, compression
#   - gem5 stats.txt  -> cycles, time, vec ratio, cache misses, MPKI,
#                        measured DRAM traffic (R+W)
#
# Correctness guard: C_nnz for a given matrix must match across gc/gcv builds.
# A 'cnnz_ref_mismatch' column flags any row whose C_nnz differs from the first
# build seen for that matrix (catches phantom speedups from dropped work).
#
# NOTE on vec_ratio: it is process-wide (committed Simd / committed total) and
# therefore includes libc/startup vector ops, not only the SpGEMM kernel.
# Treat near-zero values as "kernel not vectorized"; report with this caveat.
#
# Usage:
#   ./bench/run_gem5.sh           # default: i8
#   ./bench/run_gem5.sh i8|f32|unroll4|all
#
# Output: bench/results_gem5.csv
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CC=/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc
INC="-Iinclude"
HARNESS="bench/bench_spgemm_mtx.c"
KDIR="src/kernels/spgemm"
GEM5=/root/gem5/build/RISCV/gem5.opt
SE_CONFIG=/root/gem5/configs/deprecated/example/se.py
MATRIX_DIR="matrices"
OUTROOT="$ROOT/results_gem5"
RESULTS="bench/results_gem5.csv"
REPS=1

CPU=MinorCPU
CLOCK=2GHz
GEM5_OPTS="--cpu-type=$CPU --caches --l2cache --l1d_size=64kB --l1i_size=64kB --l2_size=1MB --sys-clock=$CLOCK"

KSEL="${1:-i8}"
case "$KSEL" in
    i8)      KERNELS="i8:KERNEL_I8:csr_scalar_i8.c" ;;
    f32)     KERNELS="f32:KERNEL_F32:csr_scalar_f32.c" ;;
    unroll4) KERNELS="unroll4:KERNEL_UNROLL4:csr_scalar_unroll4_f32.c" ;;
    all)     KERNELS="f32:KERNEL_F32:csr_scalar_f32.c unroll4:KERNEL_UNROLL4:csr_scalar_unroll4_f32.c i8:KERNEL_I8:csr_scalar_i8.c" ;;
    *) echo "unknown kernel '$KSEL' (use: i8 | f32 | unroll4 | all)"; exit 1 ;;
esac

echo "Ensuring binaries for: $KSEL"
BINS=""
for spec in $KERNELS; do
    tag="${spec%%:*}"; rest="${spec#*:}"; def="${rest%%:*}"; src="${rest#*:}"
    [ -x "bench_${tag}_gc"  ] || $CC -O3 -march=rv64gc  -static $INC -D$def "$HARNESS" "$KDIR/$src" -o "bench_${tag}_gc"
    [ -x "bench_${tag}_gcv" ] || $CC -O3 -march=rv64gcv -ftree-vectorize -static $INC -D$def "$HARNESS" "$KDIR/$src" -o "bench_${tag}_gcv"
    BINS="$BINS bench_${tag}_gc bench_${tag}_gcv"
done

mkdir -p "$OUTROOT"

echo "build,matrix,kernel,M,A_nnz,madd_pairs,ops_analytical,bytes_analytical,AI,C_nnz,compression,checksum,simInsts,simOps,numCycles,simSeconds,vec_insts,vec_ratio,l1d_misses,l2_misses,mpki,dram_read_bytes,dram_write_bytes,dram_traffic_total,cnnz_ref_mismatch" > "$RESULTS"

sval() { grep -m1 "$1" "$2" 2>/dev/null | awk '{print $2}'; }

# first DRAM byte-stat name seen (handles old vs new gem5 naming) — resolved on first run
DRAM_RD_KEY=""
DRAM_WR_KEY=""
resolve_dram_keys() {
    local st="$1"
    # try new naming first (gem5 >= ~22 renamed to dramBytesRead/Written), then old
    for k in 'dram.dramBytesRead' 'dram.bytesRead'; do
        grep -q "$k" "$st" && { DRAM_RD_KEY="$k"; break; }
    done
    for k in 'dram.dramBytesWritten' 'dram.bytesWritten'; do
        grep -q "$k" "$st" && { DRAM_WR_KEY="$k"; break; }
    done
}

# track reference C_nnz per matrix (first build seen)
declare -A CNNZ_REF

for mtx in "$MATRIX_DIR"/*/*.mtx; do
    name="$(basename "$mtx")"; mname="${name%.mtx}"
    case "$name" in *_b.mtx) continue ;; esac

    for bin in $BINS; do
        case "$bin" in *_gcv) build="gcv" ;; *) build="gc" ;; esac
        outdir="$OUTROOT/${mname}_${bin}"
        echo ">> gem5 [$build] $bin  $name" >&2

        hline="$("$GEM5" --outdir="$outdir" "$SE_CONFIG" \
                    --cmd="$ROOT/$bin" --options="$ROOT/$mtx $REPS" \
                    $GEM5_OPTS 2>/dev/null | grep -E '^[^,]+\.mtx,')"

        st="$outdir/stats.txt"
        if [ ! -f "$st" ]; then echo "!! NO STATS: $bin $name" >&2; continue; fi

        # resolve DRAM stat names once
        [ -z "$DRAM_RD_KEY" ] && resolve_dram_keys "$st"

        insts=$(sval 'simInsts'  "$st")
        ops=$(sval 'simOps'      "$st")
        cyc=$(sval 'system.cpu.numCycles' "$st")
        secs=$(sval 'simSeconds' "$st")
        l1miss=$(sval 'system.cpu.dcache.overallMisses::total' "$st")
        l2miss=$(sval 'system.l2.overallMisses::total' "$st")

        total=$(grep -m1 'commitStats0.committedInstType::total' "$st" | awk '{print $2}')
        vec=$(grep 'commitStats0.committedInstType::Simd' "$st" | awk '{s+=$2} END{print s+0}')
        vratio=""
        [ -n "${total:-}" ] && [ "${total:-0}" != "0" ] && \
            vratio=$(awk -v v="${vec:-0}" -v t="$total" 'BEGIN{printf "%.6f", v/t}')

        mpki=""
        [ -n "${l2miss:-}" ] && [ -n "${insts:-}" ] && [ "${insts:-0}" != "0" ] && \
            mpki=$(awk -v m="$l2miss" -v i="$insts" 'BEGIN{printf "%.4f", (m/i)*1000}')

        drd=""; dwr=""; dtot=""
        [ -n "$DRAM_RD_KEY" ] && drd=$(sval "$DRAM_RD_KEY" "$st")
        [ -n "$DRAM_WR_KEY" ] && dwr=$(sval "$DRAM_WR_KEY" "$st")
        if [ -n "${drd:-}" ] && [ -n "${dwr:-}" ]; then
            dtot=$(awk -v r="$drd" -v w="$dwr" 'BEGIN{printf "%.0f", r+w}')
        fi

        # correctness guard: extract C_nnz (col 9 of harness line: matrix,kernel,M,A_nnz,madd,ops,bytes,AI,C_nnz,...)
        cnnz=""
        if [ -n "${hline:-}" ]; then
            cnnz=$(echo "$hline" | awk -F, '{print $9}')
        fi
        mismatch="0"
        if [ -n "${cnnz:-}" ]; then
            if [ -z "${CNNZ_REF[$mname]:-}" ]; then
                CNNZ_REF[$mname]="$cnnz"
            elif [ "${CNNZ_REF[$mname]}" != "$cnnz" ]; then
                mismatch="1"
                echo "!! C_nnz MISMATCH $name $bin: ${cnnz} vs ref ${CNNZ_REF[$mname]}" >&2
            fi
        fi

        [ -z "${hline:-}" ] && hline="${name},${KSEL},,,,,,,,,"
        echo "$build,$hline,${insts:-},${ops:-},${cyc:-},${secs:-},${vec:-},${vratio:-},${l1miss:-},${l2miss:-},${mpki:-},${drd:-},${dwr:-},${dtot:-},${mismatch}" >> "$RESULTS"
    done
done

echo "Done. Results in $RESULTS"
[ -z "$DRAM_RD_KEY" ] && echo "WARN: DRAM byte stats not found — measured-traffic columns are blank. Check stat names with: grep -i dram.*ytes <a stats.txt>" >&2
