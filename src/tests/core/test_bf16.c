/* bfloat16 conversion and the bf16-weight matmul (core/bf16.c).
 *
 * The kernel's contract is narrow and worth stating precisely, because the
 * obvious wrong reading of "bf16 matmul" is a much more dangerous change:
 * only **B's stored values** are rounded to 8 mantissa bits. Every product
 * and every partial sum is still binary32. So the right reference is not the
 * fp32 result on the original weights - it is the fp32 result on weights that
 * have been round-tripped through bf16. Comparing against the original would
 * conflate "the kernel is wrong" with "bf16 has less mantissa", and only the
 * first is a bug.
 *
 * That reference also makes the comparison tight rather than tolerant: the
 * two computations sum identical float values, differing only in association,
 * so they agree to reduction-order error and not to bf16's 2^-8.
 */
#include "core/bf16.h"
#include "core/matmul.h"
#include "core/cpu_features.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

/* ---- conversion ---- */

static void check_roundtrip_exactness(void) {
    printf("--- conversion ---\n");

    /* A float whose low 16 bits are already zero is exactly representable and
     * must survive unchanged. */
    const float exact[] = { 0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 256.0f, -768.0f };
    for (size_t i = 0; i < sizeof(exact) / sizeof(exact[0]); i++) {
        float back = bf16_to_f32(bf16_from_f32(exact[i]));
        if (back != exact[i]) {
            fprintf(stderr, "FAIL: %g did not survive the round trip (got %g)\n",
                    (double)exact[i], (double)back);
            failures++;
        }
    }

    /* Infinities pass through; a NaN must stay a NaN rather than becoming an
     * infinity, which is what a bare >>16 does when the payload lives entirely
     * in the discarded low bits.
     *
     * Checked on BIT PATTERNS, not with isnan() or a comparison against
     * INFINITY. This project builds with -ffast-math, which implies
     * -ffinite-math-only: the compiler is then entitled to assume no NaN or
     * infinity ever occurs and to fold `isnan(x)` to false and `x != INFINITY`
     * to true regardless of x. An earlier version of this test used those and
     * reported two failures against a conversion that was perfectly correct.
     * Integer comparisons on the encoded value are immune, and they test the
     * contract more directly anyway. */
    const struct { uint32_t in; uint16_t want; const char *what; } exact_bits[] = {
        { 0x7F800000u, 0x7F80u, "+infinity" },
        { 0xFF800000u, 0xFF80u, "-infinity" },
    };
    for (size_t i = 0; i < sizeof(exact_bits) / sizeof(exact_bits[0]); i++) {
        float in;
        memcpy(&in, &exact_bits[i].in, sizeof(in));
        bf16_t got = bf16_from_f32(in);
        if (got != exact_bits[i].want) {
            fprintf(stderr, "FAIL: %s encoded as 0x%04X, expected 0x%04X\n",
                    exact_bits[i].what, got, exact_bits[i].want);
            failures++;
        }
    }

    /* A NaN whose payload lives entirely in the discarded low 16 bits. The
     * encoded value must have an all-ones exponent AND a non-zero mantissa;
     * losing the mantissa turns it into an infinity. */
    const uint32_t low_payload_nan_bits = 0x7F800001u;
    float low_payload_nan;
    memcpy(&low_payload_nan, &low_payload_nan_bits, sizeof(low_payload_nan));
    const bf16_t encoded_nan = bf16_from_f32(low_payload_nan);
    const int exponent_all_ones = ((encoded_nan >> 7) & 0xFFu) == 0xFFu;
    const int mantissa_nonzero = (encoded_nan & 0x7Fu) != 0u;
    if (!exponent_all_ones || !mantissa_nonzero) {
        fprintf(stderr, "FAIL: a NaN with a low-bit payload encoded as 0x%04X "
                        "(exponent all ones: %d, mantissa non-zero: %d) - it "
                        "became an infinity\n",
                encoded_nan, exponent_all_ones, mantissa_nonzero);
        failures++;
    }

    /* Relative error is bounded by half an ULP of an 8-bit mantissa. */
    float worst_rel = 0.0f;
    for (int i = -20000; i <= 20000; i++) {
        float v = (float)i / 137.0f;
        if (v == 0.0f) continue;
        float back = bf16_to_f32(bf16_from_f32(v));
        float rel = fabsf(back - v) / fabsf(v);
        if (rel > worst_rel) worst_rel = rel;
    }
    const float bound = 1.0f / 256.0f; /* 2^-8, half an ULP of 8 explicit bits */
    printf("  worst relative round-trip error over 40001 values: %.3e (bound %.3e)\n",
           (double)worst_rel, (double)bound);
    if (worst_rel > bound) {
        fprintf(stderr, "FAIL: round-trip error exceeds half an ULP\n");
        failures++;
    }
}

/* Round-to-nearest-even against truncation. Truncation biases every value
 * toward zero, so summing many of them drifts with the reduction length
 * instead of cancelling - the reason bf16_from_f32 costs an add. */
static void check_rounding_is_unbiased(void) {
    const size_t count = 20000;
    double round_bias = 0.0, truncate_bias = 0.0;
    for (size_t i = 0; i < count; i++) {
        float v = (float)((int)(i % 7919) - 3959) / 313.0f;
        if (v == 0.0f) continue;

        round_bias += (double)bf16_to_f32(bf16_from_f32(v)) - (double)v;

        uint32_t bits;
        memcpy(&bits, &v, sizeof(bits));
        uint32_t truncated_bits = bits & 0xFFFF0000u;
        float truncated;
        memcpy(&truncated, &truncated_bits, sizeof(truncated));
        truncate_bias += (double)truncated - (double)v;
    }
    printf("  summed signed error: round-to-nearest %+.6g, truncation %+.6g\n",
           round_bias, truncate_bias);

    /* Truncation's bias is one-directional per sign, and this input set is
     * symmetric about zero, so the telling comparison is magnitude: rounding
     * must be dramatically closer to zero than truncation. */
    if (!(fabs(round_bias) < fabs(truncate_bias) * 0.2)) {
        fprintf(stderr, "FAIL: rounding is not measurably less biased than "
                        "truncation (%.6g vs %.6g) - is bf16_from_f32 truncating?\n",
                round_bias, truncate_bias);
        failures++;
    }
}

/* ---- the kernel ---- */

typedef struct { size_t m, k, n; const char *what; } shape_t;

static const shape_t shapes[] = {
    { 1,   64,  1000, "decode head" },
    { 128, 256, 4000, "all-position head" },
    { 32,  256, 64,   "prefill FFN down" },
    { 64,  64,  256,  "prefill FFN up" },
    { 17,  33,  47,   "no extent a multiple of 8 or 16" },
    { 5,   7,   3,    "smaller than one vector in every extent" },
    { 3,   9,   17,   "one past a vector in n" },
    { 9,   16,  16,   "n exactly one AVX-512 vector" },
};

static void fill(float *buffer, size_t count, int salt) {
    for (size_t i = 0; i < count; i++) {
        buffer[i] = (float)((int)((i * 11 + (size_t)salt * 5) % 37) - 18) / 7.0f;
    }
}

static float check_shape(const shape_t *shape, cpu_isa_t isa) {
    const size_t m = shape->m, k = shape->k, n = shape->n;
    float *A = malloc(m * k * sizeof(float));
    float *B = malloc(k * n * sizeof(float));
    float *B_rounded = malloc(k * n * sizeof(float));
    bf16_t *B_narrow = malloc(k * n * sizeof(bf16_t));
    float *reference = malloc(m * n * sizeof(float));
    float *actual = malloc(m * n * sizeof(float));
    if (!A || !B || !B_rounded || !B_narrow || !reference || !actual) {
        fprintf(stderr, "FAIL: allocation for %s\n", shape->what);
        failures++;
        free(A); free(B); free(B_rounded); free(B_narrow); free(reference); free(actual);
        return -1.0f;
    }

    fill(A, m * k, 1);
    fill(B, k * n, 2);
    bf16_encode_array(B, B_narrow, k * n);
    /* The reference: fp32 arithmetic on exactly the values the bf16 kernel
     * will see. Any disagreement past reduction-order error is a kernel bug,
     * not a property of the format. */
    bf16_decode_array(B_narrow, B_rounded, k * n);

    cpu_features_set_max_isa(isa);
    matrix_multiply(A, B_rounded, reference, m, k, n);
    matmul_bf16_weight(A, B_narrow, actual, m, k, n, 0);
    cpu_features_clear_max_isa();

    float worst = 0.0f, magnitude = 0.0f;
    for (size_t i = 0; i < m * n; i++) {
        float d = fabsf(reference[i] - actual[i]);
        if (d > worst) worst = d;
        if (fabsf(reference[i]) > magnitude) magnitude = fabsf(reference[i]);
    }
    const float tolerance = 1e-4f * (1.0f + magnitude);
    if (!(worst <= tolerance)) {
        fprintf(stderr, "FAIL: bf16 kernel diverged on %s: %g > %g\n",
                shape->what, (double)worst, (double)tolerance);
        failures++;
    }
    printf("  %-34s %zux%zux%zu  worst %.3g\n", shape->what, m, k, n, (double)worst);

    free(A); free(B); free(B_rounded); free(B_narrow); free(reference); free(actual);
    return worst;
}

/* The whole point of the format is that the stored array is half the size.
 * Cheap to assert, and it fails loudly if bf16_t ever grows a struct wrapper
 * or padding. */
static void check_storage_is_halved(void) {
    if (sizeof(bf16_t) * 2 != sizeof(float)) {
        fprintf(stderr, "FAIL: bf16_t is %zu bytes, expected half of float's %zu\n",
                sizeof(bf16_t), sizeof(float));
        failures++;
    }
}

int main(void) {
    printf("=== bfloat16 weight storage ===\n");
    printf("cpu: %s\n\n", cpu_features_summary());

    check_storage_is_halved();
    check_roundtrip_exactness();
    check_rounding_is_unbiased();

    /* Every rung, for the reason recorded in test_matmul_backward.c: a test
     * that only exercises the widest one lets the narrower kernels ship
     * unverified on the machines that actually run them. Availability is read
     * before anything touches the cap, since clearing it also clears
     * DRANZER_CPU_ISA. */
    const struct { cpu_isa_t isa; const char *name; } rungs[] = {
        { CPU_ISA_AVX512, "AVX-512" },
        { CPU_ISA_AVX2,   "AVX2" },
        { CPU_ISA_BASELINE, "portable" },
    };
    int usable[3];
    for (size_t r = 0; r < 3; r++) usable[r] = cpu_isa_available(rungs[r].isa);

    float worst = 0.0f;
    for (size_t r = 0; r < 3; r++) {
        if (!usable[r]) {
            printf("\n%s: absent on this CPU - skipped\n", rungs[r].name);
            continue;
        }
        printf("\n%s kernel against fp32 on bf16-rounded weights:\n", rungs[r].name);
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
            float d = check_shape(&shapes[i], rungs[r].isa);
            if (d > worst) worst = d;
        }
    }
    cpu_features_clear_max_isa();

    printf("\nworst difference from the fp32 reference: %.8g\n", (double)worst);
    if (failures != 0) {
        printf("\nBF16 CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nBF16 CHECK PASSED\n");
    return 0;
}
