#ifndef FP_BITS_H
#define FP_BITS_H

#include <stdint.h>
#include <string.h>

/*
 * IEEE-754 classification by exponent bits, for code that must keep working
 * under -ffast-math.
 *
 * This project builds with -ffast-math (see src/Makefile, which records what
 * the flag is worth in measured throughput). -ffast-math implies
 * -ffinite-math-only, which is a promise to the compiler that no infinity and
 * no NaN will ever reach it. Given that promise, isnan(x) folds to false,
 * isfinite(x) folds to true, and x != x folds to false - so every library
 * predicate for detecting a bad float becomes a constant.
 *
 * That is not a hypothetical. It has produced two wrong results in this repo:
 * tests/core/test_bf16.c reported two failures against a correct conversion
 * because its isnan()/!= INFINITY checks were folded away, and CLI validation
 * of strtof() output silently accepted "inf" until it was rewritten to look at
 * bits (the original of the float helper below, once local to cli/main.c).
 *
 * Reading the bits is immune, because no permission the compiler has been
 * given says anything about integers. A float is non-finite exactly when all
 * eight exponent bits are set: NaN if the mantissa is non-zero, infinity if it
 * is zero. memcpy() is the aliasing-safe way to see them and every compiler
 * this project targets lowers it to a register move at -O1 and above.
 *
 * Model code should not need these: the forward pass masks with a finite value
 * rather than -INFINITY precisely so nothing in it depends on IEEE special
 * cases (see attention_head_forward() in core/transformer.c). They exist for
 * validation at the edges - parsed input, tests asserting a conversion
 * contract, and benchmarks checking that a kernel did not produce garbage.
 */

static inline int dranzer_float_is_finite(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static inline int dranzer_float_is_nan(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
           (bits & UINT32_C(0x007fffff)) != 0;
}

static inline int dranzer_double_is_finite(double value) {
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

static inline int dranzer_double_is_nan(double value) {
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) &&
           (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

#endif /* FP_BITS_H */
