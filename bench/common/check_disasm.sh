#!/usr/bin/env bash

# Check what a bench binary's kernels actually compiled to.
#
#   check_disasm.sh ISA scalar BINARY PREFIX
#       Fail if any function whose name starts with PREFIX contains a vector
#       instruction. Use it for every arm that claims to be scalar.
#
#   check_disasm.sh ISA vector BINARY PREFIX MNEMONIC...
#       Fail unless the PREFIX functions contain every listed mnemonic (matched
#       as a prefix, so vfmadd matches vfmadd231ps). Use it to prove a vector
#       kernel really gathers, FMAs and scatters.
#
#   check_disasm.sh ISA report BINARY PREFIX
#       List the vector instructions in the PREFIX functions and never fail.
#       Use it where vector code is allowed but worth recording, such as the
#       autovec arm.
#
# ISA is riscv or intel.
#
#   riscv  Every RVV instruction mnemonic starts with "v" and no scalar one
#          does, so any v* mnemonic is vector code, vsetvli included.
#   intel  Vector code is anything on ymm, zmm or a mask register, or packed
#          arithmetic on xmm. Scalar float uses xmm with ss/sd forms, plain
#          register moves and the self-xor zeroing idiom, which are allowed.
#
# An instruction objdump cannot decode (.insn, .word, .4byte, ...) could be
# vector code it does not know, so a scalar check refuses to pass over one.
# On RISC-V, use a binutils new enough to know RVV (2.38+), or set OBJDUMP.
#
# Functions are matched by name after LTO, so suffixes such as .constprop.0,
# .lto_priv.0 and OpenMP's ._omp_fn.0 bodies are included.
#
# Exit status: 0 pass, 1 check failed, 2 usage error or no functions found,
# 3 undecodable instructions in a scalar check.

set -uo pipefail

ISA="${1:-}"
MODE="${2:-}"
BIN="${3:-}"
PREFIX="${4:-}"

shift 4 2>/dev/null || true

OBJDUMP="${OBJDUMP:-objdump}"

usage() {
    echo "usage: $0 riscv|intel scalar|vector|report BINARY PREFIX [MNEMONIC...]" >&2
    exit 2
}

case "$ISA" in riscv|intel) ;; *) usage ;; esac
case "$MODE" in scalar|report) ;; vector) [ $# -gt 0 ] || usage ;; *) usage ;; esac
[ -n "$PREFIX" ] && [ -f "$BIN" ] || usage

disasm="$("$OBJDUMP" -d --no-show-raw-insn "$BIN")" || {
    echo "!! $OBJDUMP could not disassemble $BIN" >&2
    exit 2
}

# One line per instruction of a matching function: "func<TAB>mnemonic<TAB>operands".
insns="$(awk -v prefix="$PREFIX" '
    /^[0-9a-f]+ <.*>:$/ {
        name = $2
        gsub(/[<>:]/, "", name)
        keep = (index(name, prefix) == 1)
        next
    }
    keep && /^[[:space:]]*[0-9a-f]+:\t/ {
        line = $0
        sub(/^[[:space:]]*[0-9a-f]+:\t/, "", line)
        sub(/[[:space:]]*#.*$/, "", line)
        mnem = line
        sub(/[[:space:]].*$/, "", mnem)
        ops = line
        sub(/^[^[:space:]]*[[:space:]]*/, "", ops)
        print name "\t" mnem "\t" ops
    }
' <<< "$disasm")"

funcs="$(cut -f1 <<< "$insns" | sed '/^$/d' | sort -u | tr '\n' ' ')"

if [ -z "$funcs" ]; then
    echo "!! no functions starting with $PREFIX in $BIN" >&2
    exit 2
fi

echo "   [$MODE] $(basename "$BIN"): $funcs"

# Vector instructions among them, one "func<TAB>mnemonic<TAB>operands" per line.
if [ "$ISA" = "riscv" ]; then
    vec="$(awk -F'\t' '$2 ~ /^v/' <<< "$insns")"
else
    vec="$(awk -F'\t' '
        $3 ~ /%[yz]mm[0-9]+|%k[0-7]([^0-9]|$)/ { print; next }
        $3 ~ /%xmm/ {
            m = $2
            # register moves and self-xor zeroing are scalar idioms
            if (m ~ /^v?(movap[sd]|movup[sd]|movd|movq|movss|movsd)$/) next
            if (m ~ /^v?(xorp[sd]|pxor)$/) {
                n = split($3, r, ",")
                same = 1
                for (i = 2; i <= n; i++) if (r[i] != r[1]) same = 0
                if (same) next
            }
            # packed forms: ...ps / ...pd, or integer p* ops
            if (m ~ /p[sd]$/ || m ~ /^v?p[a-z]/) { print; next }
        }
    ' <<< "$insns")"
fi

undecoded="$(awk -F'\t' '$2 ~ /^(\.insn|\.word|\.short|\.byte|\.2byte|\.4byte|\.8byte|\(bad\))$/ || $2 ~ /unknown/' <<< "$insns")"

histogram() {
    cut -f2 | sort | uniq -c | sort -rn | awk '{ printf "%s x%s  ", $2, $1 } END { print "" }'
}

case "$MODE" in
    scalar)
        if [ -n "$vec" ]; then
            echo "!! VECTOR CODE in functions that must be scalar ($PREFIX*) in $BIN:" >&2
            echo "   $(histogram <<< "$vec")" >&2
            head -10 <<< "$vec" | sed 's/^/     /' >&2
            exit 1
        fi

        if [ -n "$undecoded" ]; then
            echo "!! $OBJDUMP could not decode some instructions in $PREFIX*, so scalar cannot be confirmed:" >&2
            head -5 <<< "$undecoded" | sed 's/^/     /' >&2
            echo "   use a binutils that knows this ISA (set OBJDUMP)" >&2
            exit 3
        fi

        echo "   no vector instructions: scalar confirmed"
        ;;

    vector)
        missing=()

        for m in "$@"; do
            awk -F'\t' -v m="$m" 'index($2, m) == 1 { found = 1 } END { exit !found }' <<< "$insns" ||
                missing+=("$m")
        done

        if [ "${#missing[@]}" -gt 0 ]; then
            echo "!! expected vector instructions missing from $PREFIX* in $BIN: ${missing[*]}" >&2
            exit 1
        fi

        echo "   found $*: vector kernel confirmed"
        echo "   vector instructions: $(histogram <<< "$vec")"
        ;;

    report)
        if [ -n "$vec" ]; then
            echo "   vector instructions present (allowed here): $(histogram <<< "$vec")"
        else
            echo "   no vector instructions"
        fi
        ;;
esac

exit 0
