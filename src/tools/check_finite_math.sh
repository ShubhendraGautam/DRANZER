#!/usr/bin/env bash
#
# Structural guard for the one rule -ffast-math imposes on this codebase: a
# translation unit compiled with it may not depend on infinity or NaN.
#
# -ffast-math implies -ffinite-math-only, which promises the compiler that
# neither value will ever appear. Under that promise isnan(x) folds to false,
# isfinite(x) folds to true, x != x folds to false, and an INFINITY written
# into a variable is compared against finite values under an assumption that
# says it cannot be there. Clang 18 warns (-Wnan-infinity-disabled); clang 14
# does not, and gcc does not, so the warning is not a reliable gate - it fires
# on some of the toolchains this project supports and not others.
#
# This checks the property directly instead, on every compiler and every
# version: no forbidden construct in the code of any file built with the flag.
# The exempt list is read out of the Makefile rather than duplicated here, so a
# file that legitimately drops -ffast-math (core/quantize.c does, for a
# documented reason) is exempted by the build rule that drops it.
#
# Comments are stripped before matching, via -fpreprocessed -dD -E: that mode
# removes comments without expanding includes or macros. Discussing INFINITY in
# a comment is how the decision stays documented; the guard is about code.
#
# Values may still be *produced* through bit patterns - see
# tests/core/test_bf16.c, which builds an infinity with memcpy() from
# 0x7F800000 to test that the bf16 encoder handles it. That is a deliberate
# input to a conversion, not a promise-violating dependence, and it does not
# use the library macros this scans for.
set -uo pipefail

cd "$(dirname "$0")/.."

STRIPPER=${CC:-cc}

# Files the Makefile explicitly compiles without -ffast-math.
exempt=$(sed -n 's/^\([A-Za-z0-9_/.-]*\)\.o:[[:space:]]*CFLAGS.*filter-out -ffast-math.*/\1.c/p' Makefile)

# The library macros whose behaviour -ffinite-math-only changes. NAN and
# INFINITY are included because naming them is what clang 18 warns about.
FORBIDDEN='\b(isnan|isinf|isfinite|fpclassify|INFINITY|NAN|HUGE_VAL|HUGE_VALF)\b'

status=0
checked=0
while IFS= read -r file; do
    case " $exempt " in *" $file "*) continue ;; esac
    checked=$((checked + 1))
    hits=$("$STRIPPER" -x c -fpreprocessed -dD -E "$file" 2>/dev/null |
           grep -nE "$FORBIDDEN" || true)
    if [ -n "$hits" ]; then
        echo "$file: depends on IEEE special values but is compiled with -ffast-math:"
        echo "$hits" | sed 's/^/    /'
        status=1
    fi
done < <(find . \( -name '*.c' -o -name '*.h' \) -not -path './.arm-check-stub/*' |
         sed 's|^\./||' | sort)

if [ "$status" -eq 0 ]; then
    echo "finite-math check: $checked files clean$([ -n "$exempt" ] && echo ", exempt: $(echo $exempt | tr '\n' ' ')")"
else
    echo ""
    echo "Fix by not depending on the value: mask with a finite sentinel, or"
    echo "classify by exponent bits with include/common/fp_bits.h. If the file"
    echo "genuinely needs IEEE semantics, drop -ffast-math for it in the"
    echo "Makefile with a comment saying why - that exempts it here too."
fi
exit $status
