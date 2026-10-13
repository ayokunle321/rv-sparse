#!/usr/bin/env bash

# Confirm the scalar kernels in an x86 bench binary are genuinely scalar.
#
# Disassembles every function whose name starts with rvsp_spgemm_scalar
# (the scalar and OpenMP kernels, their numeric passes and any OpenMP outlined
# bodies), then fails if any of them touches a ymm or zmm register, uses a
# mask register, or does packed arithmetic on xmm. Scalar SSE/AVX forms (ss,
# sd), register moves and the xor zeroing idiom are allowed, since that is how
# x86 does scalar float.
#
# usage: check_scalar_disasm.sh BINARY [OBJDUMP]
# Exit 0 when clean, 1 when vector code is found, 2 on usage error.

set -uo pipefail

BIN="${1:-}"
OBJDUMP="${2:-objdump}"

[ -n "$BIN" ] && [ -f "$BIN" ] || {
    echo "usage: $0 BINARY [OBJDUMP]" >&2
    exit 2
}

disasm="$("$OBJDUMP" -d --no-show-raw-insn -M att "$BIN")" || exit 2

# Keep only the scalar kernel functions.
funcs="$(awk '
    /^[0-9a-f]+ <.*>:$/ {
        name = $2
        gsub(/[<>:]/, "", name)
        keep = (name ~ /^rvsp_spgemm_scalar/)
        if (keep) print "== " name
        next
    }
    keep && NF { print }
' <<< "$disasm")"

nfuncs="$(grep -c '^== ' <<< "$funcs")"

if [ "$nfuncs" -eq 0 ]; then
    echo "!! no rvsp_spgemm_scalar* functions found in $BIN" >&2
    exit 2
fi

echo "   scalar functions checked in $(basename "$BIN"): $(grep '^== ' <<< "$funcs" | cut -c4- | tr '\n' ' ')"

# Anything on ymm/zmm or k registers is vector code outright.
wide="$(grep -E '%[yz]mm[0-9]+|%k[0-7]\b' <<< "$funcs")"

# Packed arithmetic on xmm. Moves and self-xor zeroing are allowed.
packed="$(grep -E '%xmm' <<< "$funcs" |
    grep -E '\b(v?p[a-z]+|v?[a-z0-9]+p[sd])\b' |
    grep -vE '\b(v?movap[sd]|v?movup[sd]|v?movd|v?movq)\b' |
    grep -vE '\b(v?xorp[sd]|v?pxor)[[:space:]]+(%xmm[0-9]+),\2,\2' |
    grep -vE '\b(v?xorp[sd]|v?pxor)[[:space:]]+(%xmm[0-9]+),\2$' )"

if [ -n "$wide" ] || [ -n "$packed" ]; then
    echo "!! vector instructions found in scalar kernels of $BIN:" >&2
    { echo "$wide"; echo "$packed"; } | grep -v '^$' | head -20 >&2
    exit 1
fi

echo "   no ymm/zmm, mask-register or packed xmm arithmetic: scalar confirmed"
exit 0
