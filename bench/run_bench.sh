#!/usr/bin/env bash
#
# SpGEMM benchmark driver for rv-sparse.
#
# Runs each kernel configuration in a single process on a single CPU core to
# produce clean, comparable performance measurements.
#
# Usage:
#   bash bench/run_bench.sh
#
# Environment variables:
#   RUNS=10        Number of timed runs per configuration.
#   WARMUP=3       Number of warmup runs.
#   PIN_CORE=0     CPU core to pin the benchmark to (via taskset).
#   CC=gcc         Compiler to use.
#   ARCH="-march=..."
#                  Architecture flags (defaults to rv64gcv on the K1).
#   PERF=1         Collect hardware cycle and instruction counters.
#   BIG=1          Include the large benchmark matrices.
#   FORCE=1        Re-run configurations even if results already exist.
#   ONLY_MTX=name  Benchmark only the specified matrix.
#   TIMEOUT=600    Per-configuration timeout in seconds (0 disables).
#   RETRIES=1      Additional retry attempts before giving up.
#   SHUFFLE=1      Randomize kernel order for each matrix (default: enabled).
#   SHUF_SEED=N    Seed used for kernel randomization (recorded in results).
#   NO_GOVERNOR=1  Skip setting the CPU performance governor.

set -uo pipefail   # NOTE: no -e; a failing config should not kill the sweep

# config
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-bench/results}"
CSV="${CSV:-$OUT_DIR/spgemm_bench.csv}"
RUNS="${RUNS:-10}"
WARMUP="${WARMUP:-3}"
PIN_CORE="${PIN_CORE:-0}"
CC="${CC:-gcc}"
ARCH="${ARCH:--march=rv64gcv -mabi=lp64d}"
PERF="${PERF:-0}"
BIG="${BIG:-0}"
FORCE="${FORCE:-0}"
ONLY_MTX="${ONLY_MTX:-}"
TIMEOUT="${TIMEOUT:-0}"
RETRIES="${RETRIES:-1}"
SHUFFLE="${SHUFFLE:-1}"
SHUF_SEED="${SHUF_SEED:-$RANDOM}"
NO_GOVERNOR="${NO_GOVERNOR:-0}"

if [ -n "${KERNELS:-}" ]; then read -ra KERNELS <<< "$KERNELS"; else KERNELS=(scalar_f32 unroll4_f32 rvv_f32 scalar_f64 rvv_f64 scalar_i8 rvv_i8); fi

GEN_CASES=(
  "512 512 0.02 42"
  "1024 1024 0.01 42"
  "2048 2048 0.005 42"
  "4096 4096 0.002 42"
)

SMALL_MATRICES=(wiki-Vote email-Enron p2p-Gnutella31 cage12 2cubes_sphere
                scircuit poisson3Da mario002 cop20k_A offshore m133-b3
                filter3D ca-CondMat)
BIG_MATRICES=(amazon0312 web-Google roadNet-CA patents_main cit-Patents webbase-1M)

CSV_HEADER="label,kernel,dtype,src,rows,cols,nnz_a,nnz_b,nnz_c,flops,run,time_s,gops,correct,cycles,instructions"
CSV_NFIELDS=16

# freq drift larger than this between start and end of a config is flagged
FREQ_DRIFT_WARN=0.01   # 1%

# input validation
die() { echo "!! $*" >&2; exit 1; }

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

is_uint "$RUNS"      || die "RUNS must be a non-negative integer, got '$RUNS'"
is_uint "$WARMUP"    || die "WARMUP must be a non-negative integer, got '$WARMUP'"
is_uint "$PIN_CORE"  || die "PIN_CORE must be a non-negative integer, got '$PIN_CORE'"
is_uint "$TIMEOUT"   || die "TIMEOUT must be a non-negative integer, got '$TIMEOUT'"
is_uint "$RETRIES"   || die "RETRIES must be a non-negative integer, got '$RETRIES'"
is_uint "$SHUF_SEED" || die "SHUF_SEED must be a non-negative integer, got '$SHUF_SEED'"
[ "$RUNS" -ge 1 ]    || die "RUNS must be >= 1"
[ "$RUNS" -ge 4 ] || echo ">> WARNING: RUNS=$RUNS (<4) — analyze.py cannot bootstrap a speedup CI with so few runs. Use RUNS>=10 for a paper."

mkdir -p "$OUT_DIR" || die "cannot create $OUT_DIR"

# cpufreq governor and frequency helpers
GOV_PATH="/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_governor"
FREQ_PATH="/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_cur_freq"
ORIG_GOV=""

read_freq() { cat "$FREQ_PATH" 2>/dev/null || echo 0; }

set_governor_performance() {
  [ "$NO_GOVERNOR" = "1" ] && { echo ">> governor pin skipped (NO_GOVERNOR=1)"; return; }
  if [ -w "$GOV_PATH" ]; then
    ORIG_GOV="$(cat "$GOV_PATH" 2>/dev/null || echo '')"
    if echo performance > "$GOV_PATH" 2>/dev/null; then
      echo ">> cpufreq governor set to 'performance' on core $PIN_CORE (was '${ORIG_GOV:-unknown}')"
    else
      echo ">> WARNING: could not write $GOV_PATH — DVFS is live. Small (<5%) effects may be frequency noise."
      ORIG_GOV=""
    fi
  else
    echo ">> WARNING: $GOV_PATH not writable (need root). DVFS is live — do NOT trust sub-5% effects from this run."
  fi
}

restore_governor() {
  if [ -n "$ORIG_GOV" ] && [ -w "$GOV_PATH" ]; then
    echo "$ORIG_GOV" > "$GOV_PATH" 2>/dev/null && echo ">> restored governor to '$ORIG_GOV'"
  fi
}

# exclusive lock, refuse to run if another instance holds it
LOCKFILE="$OUT_DIR/.run_bench.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
  die "another run_bench.sh is already running against $OUT_DIR (lock: $LOCKFILE). Refusing to start — running two at once would corrupt $CSV."
fi
# lock is released automatically when this process exits, for any reason

# cleanup and interrupt handling
cleanup() { restore_governor; }
trap cleanup EXIT

INTERRUPTED=0
on_interrupt() {
  INTERRUPTED=1
  echo ""
  echo "!! interrupted — CSV writes are atomic, so $CSV is not corrupted."
  echo "!! re-run this script to resume; completed configs will be skipped."
  exit 130   # fires the EXIT trap, which restores the governor
}
trap on_interrupt INT TERM

# build
echo ">> building librvsparse.a"
make CC="$CC" ARCH_FLAGS="$ARCH" >/dev/null || die "library build failed"

for f in obj/tools/genmat.o obj/tools/mtx_to_csr_formatter.o obj/tools/vec.o; do
  [ -f "$f" ] || die "expected build artifact missing: $f (did 'make' actually build it?)"
done

PERF_FLAG=""
if [ "$PERF" = "1" ]; then
  PERF_FLAG="-DUSE_PERF"
  echo ">> perf counters ENABLED (cycles, instructions — user-space only)"
  if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
    echo "   perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid) (needs <= 2)"
  fi
fi

echo ">> compiling bench.c"
$CC -Wall -Wextra -std=c11 -Iinclude -Itools/include $ARCH -O3 $PERF_FLAG \
    bench/bench.c \
    obj/tools/genmat.o obj/tools/mtx_to_csr_formatter.o obj/tools/vec.o \
    -Llib -lrvsparse -lm -o bench/bench || die "bench build failed"

[ -x bench/bench ] || die "bench/bench was not produced or is not executable"

# runner
RUNNER=""
if command -v taskset >/dev/null 2>&1; then
  RUNNER="taskset -c $PIN_CORE"
  echo ">> pinning to core $PIN_CORE"
else
  echo ">> WARNING: taskset not found — runs will NOT be pinned to one core. Numbers may be noisier."
fi
if [ "$TIMEOUT" != "0" ]; then
  if command -v timeout >/dev/null 2>&1; then
    RUNNER="timeout $TIMEOUT $RUNNER"
    echo ">> per-config timeout: ${TIMEOUT}s"
  else
    echo ">> WARNING: TIMEOUT=$TIMEOUT set but 'timeout' command not found — timeouts will NOT be enforced."
  fi
fi

# core isolation is separate from pinning, warn if the bench core is not isolated
ISOLATED="$(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo '')"
if [ -z "$ISOLATED" ] || ! grep -qw "$PIN_CORE" <<< "${ISOLATED//,/ }"; then
  echo ">> NOTE: core $PIN_CORE is not in /sys/devices/system/cpu/isolated ('${ISOLATED:-none}')."
  echo ">>       taskset pins the bench process but the scheduler can still run other work here."
  echo ">>       For sub-5% effect claims, boot with isolcpus=$PIN_CORE (or use a cpuset)."
fi

# pin the governor now, after the build so we don't hold it during compilation
set_governor_performance

# CSV setup and schema validation
if [ ! -f "$CSV" ]; then
  echo "$CSV_HEADER" > "$CSV" || die "cannot write $CSV"
  echo ">> created $CSV"
else
  existing_header="$(head -1 "$CSV")"
  if [ "$existing_header" != "$CSV_HEADER" ]; then
    die "existing $CSV has a different header than expected.
    expected: $CSV_HEADER
    found:    $existing_header
This usually means the CSV schema changed since these results were collected
(e.g. the old 'gflops' column is now 'gops'). Move/rename the old file (or set
CSV=path/to/new.csv) rather than mixing schemas."
  fi
  echo ">> appending to existing $CSV ($(( $(wc -l < "$CSV") - 1 )) rows)"
fi

# record the exact environment for reproducibility
{
  echo "date: $(date -Iseconds)"
  echo "host: $(hostname)"
  echo "uname: $(uname -srm)"
  echo "cc: $($CC --version 2>/dev/null | head -1)"
  echo "arch_flags: $ARCH -O3 $PERF_FLAG"
  echo "runs: $RUNS  warmup: $WARMUP  pin_core: $PIN_CORE  retries: $RETRIES"
  echo "shuffle: $SHUFFLE  shuf_seed: $SHUF_SEED"
  echo "governor_requested: performance  governor_orig: ${ORIG_GOV:-not_changed}"
  echo "governor_now: $(cat "$GOV_PATH" 2>/dev/null || echo n/a)"
  echo "core_isolated: ${ISOLATED:-none}"
  echo "freq_khz: $(read_freq)"
  echo "isa: $(grep -m1 isa /proc/cpuinfo 2>/dev/null || echo n/a)"
} > "$OUT_DIR/env.txt"
echo ">> environment recorded in $OUT_DIR/env.txt"

# helpers, atomic append, counting, purge, validation
atomic_append_csv() {
  local rows_file="$1"
  [ -s "$rows_file" ] || return 0
  local tmp
  tmp="$(mktemp "$OUT_DIR/.csv_XXXXXX")" || die "mktemp failed"
  cat "$CSV" "$rows_file" > "$tmp" && mv -f "$tmp" "$CSV" \
    || { rm -f "$tmp"; die "atomic CSV write failed"; }
}

config_count() {
  awk -F, -v l="$1" -v k="$2" 'NR>1 && $1==l && $2==k' "$CSV" | wc -l | tr -d ' '
}

purge_config() {
  local label="$1" k="$2" tmp
  tmp="$(mktemp "$OUT_DIR/.csv_XXXXXX")" || die "mktemp failed"
  awk -F, -v l="$label" -v k="$k" 'NR==1 || !($1==l && $2==k)' "$CSV" > "$tmp" \
    && mv -f "$tmp" "$CSV" || { rm -f "$tmp"; die "purge failed for $label/$k"; }
}

filter_valid_rows() {
  local infile="$1" label="$2" k="$3" outfile="$4"
  awk -F, -v l="$label" -v k="$k" -v nf="$CSV_NFIELDS" -v lbl="$label" '
    NF==nf && $1==l && $2==k { print; next }
    { bad++ }
    END { if (bad>0) print "   !! dropped " bad " malformed/mismatched row(s) for " lbl > "/dev/stderr" }
  ' "$infile" > "$outfile"
}

# seeded Fisher-Yates over the kernel list. Uses the global RANDOM stream, which
# was seeded once from SHUF_SEED, so the whole sweep order is reproducible.
seeded_shuffle() {
  local arr=("$@") i j tmp n=$#
  for ((i=n-1; i>0; i--)); do
    j=$(( RANDOM % (i+1) ))
    tmp="${arr[i]}"; arr[i]="${arr[j]}"; arr[j]="$tmp"
  done
  printf '%s\n' "${arr[@]}"
}

kernel_order() {
  if [ "$SHUFFLE" = "1" ]; then
    seeded_shuffle "${KERNELS[@]}"
  else
    printf '%s\n' "${KERNELS[@]}"
  fi
}

# smoke test, must pass before we touch the real sweep
echo ">> smoke test (tiny config, must pass before the real sweep)"
SMOKE_TMP="$(mktemp "$OUT_DIR/.smoke_XXXXXX")"
if ! $RUNNER ./bench/bench --kernel scalar_f32 --gen 64 64 0.05 1 \
        --runs 2 --warmup 1 --label __smoke__ > "$SMOKE_TMP" 2>&1; then
  echo "---- smoke output ----"; cat "$SMOKE_TMP"; echo "----------------------"
  rm -f "$SMOKE_TMP"
  die "smoke test failed to run — the build is broken, fix that before benchmarking"
fi
# field 14 is 'correct'; expect exactly 2 well-formed rows with correct==1
SMOKE_OK="$(awk -F, -v nf="$CSV_NFIELDS" \
  'NF==nf && $2=="scalar_f32" && $14==1' "$SMOKE_TMP" | wc -l | tr -d ' ')"
rm -f "$SMOKE_TMP"
[ "$SMOKE_OK" -eq 2 ] || die "smoke test produced $SMOKE_OK/2 valid+correct rows — something is wrong before the sweep even starts"
echo "   smoke test passed"

# state
FAILED_CONFIGS=()
COMPLETED_COUNT=0
ATTEMPTED_COUNT=0

# run_config <kernel> <label> <bench args...>
run_config() {
  local k="$1"; local label="$2"; shift 2
  ATTEMPTED_COUNT=$((ATTEMPTED_COUNT+1))

  local have
  have="$(config_count "$label" "$k")"

  if [ "$FORCE" != "1" ] && [ "$have" -eq "$RUNS" ]; then
    echo "   $label  $k  [have]"
    COMPLETED_COUNT=$((COMPLETED_COUNT+1))
    return 0
  fi

  if [ "$have" -gt 0 ]; then
    echo "   $label  $k  [partial: $have/$RUNS present — purging and re-running clean]"
    purge_config "$label" "$k"
  else
    echo "   $label  $k"
  fi

  local freq_before; freq_before="$(read_freq)"

  local attempt=0
  local ok=0
  while [ "$attempt" -le "$RETRIES" ]; do
    attempt=$((attempt+1))
    local tmp; tmp="$(mktemp "$OUT_DIR/.run_XXXXXX")"
    local valid; valid="$(mktemp "$OUT_DIR/.valid_XXXXXX")"

    if $RUNNER ./bench/bench --kernel "$k" "$@" \
          --runs "$RUNS" --warmup "$WARMUP" --label "$label" > "$tmp" 2>"$tmp.err"; then
      :
    else
      local rc=$?
      if [ "$rc" -eq 124 ]; then
        echo "   !! TIMEOUT after ${TIMEOUT}s — $label / $k (attempt $attempt/$((RETRIES+1)))"
      else
        echo "   !! failed (rc=$rc) — $label / $k (attempt $attempt/$((RETRIES+1))): $(head -1 "$tmp.err" 2>/dev/null)"
      fi
    fi

    filter_valid_rows "$tmp" "$label" "$k" "$valid"
    local n; n="$(wc -l < "$valid" | tr -d ' ')"

    if [ "$n" -eq "$RUNS" ]; then
      atomic_append_csv "$valid"
      rm -f "$tmp" "$tmp.err" "$valid"
      ok=1
      break
    fi

    echo "   .. attempt $attempt/$((RETRIES+1)) produced $n/$RUNS valid rows"
    rm -f "$tmp" "$tmp.err" "$valid"
  done

  # frequency stability, if the core clock moved across this config the
  # timing is suspect even if the run succeeded
  local freq_after; freq_after="$(read_freq)"
  if [ "$freq_before" -gt 0 ] 2>/dev/null && [ "$freq_after" -gt 0 ] 2>/dev/null; then
    local drift
    drift="$(awk -v a="$freq_before" -v b="$freq_after" \
      'BEGIN { d=(a>b?a-b:b-a); printf "%.4f", (a>0? d/a : 0) }')"
    if awk -v d="$drift" -v w="$FREQ_DRIFT_WARN" 'BEGIN{exit !(d>w)}'; then
      echo "   !! FREQ DRIFT $label / $k: ${freq_before}kHz -> ${freq_after}kHz ($(awk -v d="$drift" 'BEGIN{printf "%.1f", d*100}')%) — number may be DVFS-tainted"
    fi
  fi

  if [ "$ok" -eq 1 ]; then
    COMPLETED_COUNT=$((COMPLETED_COUNT+1))
  else
    echo "   !! GIVING UP on $label / $k after $((RETRIES+1)) attempt(s) — no rows written for this config"
    FAILED_CONFIGS+=("$label,$k")
  fi
}

# seed the shuffle stream once, deterministically
RANDOM="$SHUF_SEED"
echo ">> kernel order per matrix: $([ "$SHUFFLE" = 1 ] && echo "shuffled (seed $SHUF_SEED)" || echo "fixed")"

# synthetic sweep
echo ">> synthetic sweep"
for case in "${GEN_CASES[@]}"; do
  read -r R C D S <<< "$case"
  label="gen_${R}x${C}_d${D}"
  mapfile -t KRUN < <(kernel_order)
  echo "   [$label] order: ${KRUN[*]}"
  for k in "${KRUN[@]}"; do
    run_config "$k" "$label" --gen "$R" "$C" "$D" "$S"
  done
done

# real matrices
echo ">> real matrix sweep (C = A*A)"
if [ -n "$ONLY_MTX" ]; then
  MATRIX_LIST=("$ONLY_MTX")
else
  MATRIX_LIST=("${SMALL_MATRICES[@]}")
  [ "$BIG" = "1" ] && MATRIX_LIST+=("${BIG_MATRICES[@]}")
fi

for name in "${MATRIX_LIST[@]}"; do
  mtx="matrices/$name/$name.mtx"
  if [ ! -f "$mtx" ]; then
    echo "   (missing $mtx — skipping)"
    continue
  fi
  mapfile -t KRUN < <(kernel_order)
  echo "   [$name] order: ${KRUN[*]}"
  for k in "${KRUN[@]}"; do
    run_config "$k" "$name" --mtx-sq "$mtx"
  done
done

# final integrity report
echo ""
echo ">> integrity check: re-scanning $CSV"
BAD_GROUPS=0
while IFS= read -r pair; do
  [ -z "$pair" ] && continue
  lbl="${pair%%,*}"
  kern="${pair##*,}"
  n="$(config_count "$lbl" "$kern")"
  if [ "$n" -ne "$RUNS" ]; then
    echo "   !! $lbl / $kern has $n/$RUNS rows (not exactly RUNS)"
    BAD_GROUPS=$((BAD_GROUPS+1))
  fi
done < <(awk -F, 'NR>1 {print $1","$2}' "$CSV" | sort -u)

echo ""
echo ">> done. $CSV has $(( $(wc -l < "$CSV") - 1 )) rows"
echo ">> configs attempted this run: $ATTEMPTED_COUNT, completed: $COMPLETED_COUNT, failed: ${#FAILED_CONFIGS[@]}"
if [ "${#FAILED_CONFIGS[@]}" -gt 0 ]; then
  echo ">> FAILED configs (no rows written, will retry next run):"
  for f in "${FAILED_CONFIGS[@]}"; do echo "     $f"; done
fi
if [ "$BAD_GROUPS" -gt 0 ]; then
  echo ">> WARNING: $BAD_GROUPS group(s) in the CSV do not have exactly $RUNS rows — see above."
  echo ">> this can happen with leftover data from a run using a different RUNS value."
else
  echo ">> integrity check passed: every group has exactly $RUNS rows."
fi
echo ">> analyze with: python3 bench/analyze.py $CSV --csv-out $OUT_DIR/summary.csv"

[ "${#FAILED_CONFIGS[@]}" -eq 0 ] || exit 1
exit 0
