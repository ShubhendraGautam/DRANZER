/*
 * The backward matmuls have a vector implementation and a portable one, and
 * they must agree. Unlike the forward kernels there is no enum to select
 * between them - dispatch is internal, decided by cpu_isa_available() - so
 * this test reaches the portable path the same way a CPU without AVX-512
 * would: by capping detection to baseline.
 *
 * Three things are checked, each corresponding to a way these two functions
 * have actually been got wrong in this project:
 *
 *   - Agreement with the portable reference on shapes whose extents are not
 *     multiples of the 16-float vector width, so the masked tails run.
 *   - Accumulation. Both functions add into their destination, because every
 *     backward call site contributes to a gradient a whole minibatch shares.
 *     A kernel that stores instead of adding passes any test starting from a
 *     zeroed buffer and then silently discards every gradient but the last,
 *     so every shape is run twice against a pre-filled destination.
 *   - The k tail. backward_weight processes four rows of dB at a time; values
 *     of k that are not a multiple of four take a separate one-row path that
 *     is easy to leave unwritten.
 *
 * Reports honestly when there is no AVX-512 to compare against, rather than
 * passing silently on a machine where both paths are the same code.
 */

#include "core/cpu_features.h"
#include "core/matmul.h"
#include "core/matmul_simd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { size_t m, k, n; const char *what; } shape_t;

static const shape_t shapes[] = {
    { 1,  16, 260,  "output head gradient, k below one block" },
    { 1,  64, 1000, "output head gradient, wide n" },
    { 32, 256, 64,  "prefill FFN down" },
    { 64, 64,  256, "prefill FFN up" },
    { 17, 33,  47,  "no extent a multiple of 16 or of 4" },
    { 5,  7,   3,   "smaller than one vector in every extent" },
    { 8,  6,   16,  "k not a multiple of 4, n exactly one vector" },
    { 3,  9,   17,  "one past a vector in n, k tail of 1" },
};

static int failures;

/* Divided by 7, not by a power of two, on purpose.
 *
 * An earlier version of this test scaled by 1/8. Every value, every product,
 * and every partial sum was then exactly representable in binary32, so
 * reassociating the sum changed nothing and every shape compared bit-identical
 * - including, and this is the problem, in the case where dispatch silently
 * failed to reach the vector path at all. Inexact values make the two
 * implementations genuinely differ in the last bits, which is what lets
 * check_dispatch_reaches_simd() below tell them apart. */
static void fill(float *buffer, size_t count, int salt) {
    for (size_t i = 0; i < count; i++) {
        buffer[i] = (float)((int)((i * 11 + (size_t)salt * 5) % 37) - 18) / 7.0f;
    }
}

static float max_abs_diff(const float *a, const float *b, size_t count, float *magnitude) {
    float worst = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > worst) worst = d;
        if (fabsf(a[i]) > *magnitude) *magnitude = fabsf(a[i]);
    }
    return worst;
}

/* Run both backward functions twice under the current ISA setting, into
 * destinations that start non-zero. */
static void accumulate_twice(float *dA, float *dB, const float *A, const float *B,
                             const float *dC, size_t m, size_t k, size_t n) {
    for (int pass = 0; pass < 2; pass++) {
        matmul_backward_input((float *)dC, (float *)B, dA, m, k, n);
        matmul_backward_weight((float *)A, (float *)dC, dB, m, k, n);
    }
}

static float check_shape(const shape_t *shape) {
    size_t m = shape->m, k = shape->k, n = shape->n;
    float *A = malloc(m * k * sizeof(float));
    float *B = malloc(k * n * sizeof(float));
    float *dC = malloc(m * n * sizeof(float));
    float *ref_dA = malloc(m * k * sizeof(float));
    float *simd_dA = malloc(m * k * sizeof(float));
    float *ref_dB = malloc(k * n * sizeof(float));
    float *simd_dB = malloc(k * n * sizeof(float));
    if (!A || !B || !dC || !ref_dA || !simd_dA || !ref_dB || !simd_dB) {
        fprintf(stderr, "FAIL: allocation for %s\n", shape->what);
        failures++;
        free(A); free(B); free(dC);
        free(ref_dA); free(simd_dA); free(ref_dB); free(simd_dB);
        return -1.0f;
    }

    fill(A, m * k, 1);
    fill(B, k * n, 2);
    fill(dC, m * n, 3);
    /* Both destinations start from the same non-zero values. */
    fill(ref_dA, m * k, 4);  fill(simd_dA, m * k, 4);
    fill(ref_dB, k * n, 5);  fill(simd_dB, k * n, 5);

    cpu_features_set_max_isa(CPU_ISA_BASELINE);
    accumulate_twice(ref_dA, ref_dB, A, B, dC, m, k, n);

    cpu_features_clear_max_isa();
    accumulate_twice(simd_dA, simd_dB, A, B, dC, m, k, n);

    float magnitude = 0.0f;
    float d_input = max_abs_diff(ref_dA, simd_dA, m * k, &magnitude);
    float d_weight = max_abs_diff(ref_dB, simd_dB, k * n, &magnitude);
    /* Scales with the result: the vector path reduces across lanes, so it
     * sums the same products in a different order. */
    float tolerance = 1e-4f * (1.0f + magnitude);

    if (!(d_input <= tolerance)) {
        fprintf(stderr, "FAIL: backward_input diverged on %s: %g > %g\n",
                shape->what, (double)d_input, (double)tolerance);
        failures++;
    }
    if (!(d_weight <= tolerance)) {
        fprintf(stderr, "FAIL: backward_weight diverged on %s: %g > %g\n",
                shape->what, (double)d_weight, (double)tolerance);
        failures++;
    }

    float worst = d_input > d_weight ? d_input : d_weight;
    printf("  %-44s %zux%zux%zu  worst %.3g\n", shape->what, m, k, n, (double)worst);

    free(A); free(B); free(dC);
    free(ref_dA); free(simd_dA); free(ref_dB); free(simd_dB);
    return worst;
}

/* Proves the vector code actually ran.
 *
 * Everything above compares "capped to baseline" against "uncapped" and
 * requires them to agree. That comparison passes just as happily if the
 * uncapped path quietly ran the portable code too - a broken cpu_isa_available
 * gate, a missing DRANZER_HAVE_X86_SIMD, a dispatch branch that was never
 * taken. So call the AVX-512 entry point directly and require the dispatched
 * result to match it *bit for bit*, which it can only do by being the same
 * code. With the inexact inputs above, the portable path differs from both.
 *
 * Skipped where there is no AVX-512, since calling the entry point there
 * would execute instructions the CPU does not have. */
static void check_dispatch_reaches_simd(void) {
#ifdef DRANZER_HAVE_X86_SIMD
    if (!cpu_isa_available(CPU_ISA_AVX512)) return;

    const size_t m = 9, k = 21, n = 53;
    float A[9 * 21], B[21 * 53], dC[9 * 53];
    float dispatched_dA[9 * 21], direct_dA[9 * 21], portable_dA[9 * 21];
    float dispatched_dB[21 * 53], direct_dB[21 * 53];
    fill(A, m * k, 1); fill(B, k * n, 2); fill(dC, m * n, 3);
    fill(dispatched_dA, m * k, 4); fill(direct_dA, m * k, 4); fill(portable_dA, m * k, 4);
    fill(dispatched_dB, k * n, 5); fill(direct_dB, k * n, 5);

    cpu_features_clear_max_isa();
    matmul_backward_input(dC, B, dispatched_dA, m, k, n);
    matmul_backward_weight(A, dC, dispatched_dB, m, k, n);

    matmul_backward_input_avx512(dC, B, direct_dA, m, k, n);
    matmul_backward_weight_avx512(A, dC, direct_dB, m, k, n);

    cpu_features_set_max_isa(CPU_ISA_BASELINE);
    matmul_backward_input(dC, B, portable_dA, m, k, n);
    cpu_features_clear_max_isa();

    for (size_t i = 0; i < m * k; i++) {
        if (dispatched_dA[i] != direct_dA[i]) {
            fprintf(stderr, "FAIL: dispatch did not reach the AVX-512 backward_input "
                            "(element %zu: %.9g vs %.9g)\n",
                    i, (double)dispatched_dA[i], (double)direct_dA[i]);
            failures++;
            break;
        }
    }
    for (size_t i = 0; i < k * n; i++) {
        if (dispatched_dB[i] != direct_dB[i]) {
            fprintf(stderr, "FAIL: dispatch did not reach the AVX-512 backward_weight "
                            "(element %zu: %.9g vs %.9g)\n",
                    i, (double)dispatched_dB[i], (double)direct_dB[i]);
            failures++;
            break;
        }
    }

    /* And the two implementations really are distinguishable on this data,
     * so the bit-identity above is evidence rather than a coincidence. */
    int differs = 0;
    for (size_t i = 0; i < m * k; i++) {
        if (portable_dA[i] != direct_dA[i]) { differs = 1; break; }
    }
    if (!differs) {
        fprintf(stderr, "FAIL: portable and AVX-512 results are bit-identical on "
                        "this data, so the dispatch check above proves nothing - "
                        "choose inputs whose sums are not exactly representable\n");
        failures++;
    }
#endif
}

/* A destination of zero, written once, must equal the plain product - the
 * check that supporting accumulation did not break the ordinary case. */
static void check_single_pass_from_zero(void) {
    const size_t m = 6, k = 10, n = 19;
    float A[6 * 10], B[10 * 19], dC[6 * 19];
    float ref[6 * 10] = {0}, simd[6 * 10] = {0};
    fill(A, m * k, 1); fill(B, k * n, 2); fill(dC, m * n, 3);

    cpu_features_set_max_isa(CPU_ISA_BASELINE);
    matmul_backward_input(dC, B, ref, m, k, n);
    cpu_features_clear_max_isa();
    matmul_backward_input(dC, B, simd, m, k, n);

    float magnitude = 0.0f;
    float diff = max_abs_diff(ref, simd, m * k, &magnitude);
    if (!(diff <= 1e-4f * (1.0f + magnitude))) {
        fprintf(stderr, "FAIL: single pass from zero diverged: %g\n", (double)diff);
        failures++;
    }
}

int main(void) {
    printf("cpu: %s\n", cpu_features_summary());
    const int has_avx512 = cpu_isa_available(CPU_ISA_AVX512);
    if (!has_avx512) {
        printf("NOTE: no AVX-512 on this CPU, so both paths below are the same\n"
               "      portable code and this run proves only that it is stable.\n");
    }
    printf("\nvector backward kernels against the portable reference, "
           "accumulating twice into a non-zero destination:\n");

    float worst = 0.0f;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        float d = check_shape(&shapes[i]);
        if (d > worst) worst = d;
    }
    check_single_pass_from_zero();
    check_dispatch_reaches_simd();
    cpu_features_clear_max_isa();

    printf("\nworst difference from the portable backward reference: %.8g\n", (double)worst);
    if (failures != 0) {
        printf("\nMATMUL BACKWARD CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nMATMUL BACKWARD CHECK %s\n",
           has_avx512 ? "PASSED" : "PASSED (portable only)");
    return 0;
}
