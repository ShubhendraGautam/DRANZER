/* Compares gpu_matmul_backward_input()/gpu_matmul_backward_weight() (both
 * hand-written PTX) against core/matmul.c's CPU versions, which are the
 * reference the whole backward pass is built on.
 *
 * Two things make this test different from test_gpu_matmul.c, and both are
 * where a transposed-GEMM kernel actually goes wrong:
 *
 *   - These functions ACCUMULATE into their destination. A kernel that
 *     stores instead of adding passes any test that starts from a zeroed
 *     buffer, and then silently discards every gradient but the last at the
 *     real call sites. So every shape here is run twice against a
 *     pre-poisoned destination, and the second pass must double the
 *     contribution exactly as the CPU version does.
 *   - The two kernels index their operands transposed, in opposite
 *     directions. Square or symmetric shapes hide an index swap, so every
 *     shape below has three distinct extents and none is a multiple of the
 *     16-wide thread block.
 *
 * Self-skips (reports success) without a usable CUDA GPU, like every other
 * test in tests/gpu/.
 */

#include "core/matmul.h"
#include "backends/gpu/gpu_matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { size_t m, k, n; const char *what; } shape_t;

/* Shapes taken from the live backward call sites in core/training.c, plus
 * awkward ones. m=1 is the output-head gradient, which is the single most
 * common backward shape in this project. */
static const shape_t shapes[] = {
    { 1,  16, 260,  "output head gradient (tiny tier)" },
    { 1,  64, 1000, "output head gradient (small tier)" },
    { 37, 53, 41,   "no extent a multiple of the block" },
    { 64, 64, 256,  "prefill FFN up" },
    { 33, 65, 129,  "one past the block in every extent" },
    { 3,  5,  7,    "smaller than one thread block" },
};

static int failures;

/* Fill deterministically, sign-mixed so a transposed index cannot coincide. */
static void fill(float *buffer, size_t count, int salt) {
    for (size_t i = 0; i < count; i++) {
        buffer[i] = (float)((int)((i * 13 + (size_t)salt * 7) % 41) - 20) / 9.0f;
    }
}

static float max_abs_diff(const float *a, const float *b, size_t count) {
    float worst = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

/* Run one shape through both implementations twice, accumulating into a
 * destination that starts non-zero. `tolerance` scales with the result
 * magnitude: the GPU sums in a different order than the CPU. */
static float check_shape(const shape_t *shape) {
    size_t m = shape->m, k = shape->k, n = shape->n;
    float *A = malloc(m * k * sizeof(float));
    float *B = malloc(k * n * sizeof(float));
    float *dC = malloc(m * n * sizeof(float));
    float *cpu_dA = malloc(m * k * sizeof(float));
    float *gpu_dA = malloc(m * k * sizeof(float));
    float *cpu_dB = malloc(k * n * sizeof(float));
    float *gpu_dB = malloc(k * n * sizeof(float));
    if (!A || !B || !dC || !cpu_dA || !gpu_dA || !cpu_dB || !gpu_dB) {
        fprintf(stderr, "FAIL: allocation for %s\n", shape->what);
        failures++;
        free(A); free(B); free(dC);
        free(cpu_dA); free(gpu_dA); free(cpu_dB); free(gpu_dB);
        return -1.0f;
    }

    fill(A, m * k, 1);
    fill(B, k * n, 2);
    fill(dC, m * n, 3);

    /* Non-zero starting gradients: this is what catches a store-instead-of-
     * accumulate kernel. Both sides start from the same values. */
    fill(cpu_dA, m * k, 4);
    fill(gpu_dA, m * k, 4);
    fill(cpu_dB, k * n, 5);
    fill(gpu_dB, k * n, 5);

    float worst = 0.0f;
    for (int pass = 0; pass < 2; pass++) {
        matmul_backward_input(dC, B, cpu_dA, m, k, n);
        if (gpu_matmul_backward_input(dC, B, gpu_dA, m, k, n) != 0) {
            fprintf(stderr, "FAIL: gpu_matmul_backward_input failed on %s\n", shape->what);
            failures++;
            break;
        }

        matmul_backward_weight(A, dC, cpu_dB, m, k, n);
        if (gpu_matmul_backward_weight(A, dC, gpu_dB, m, k, n) != 0) {
            fprintf(stderr, "FAIL: gpu_matmul_backward_weight failed on %s\n", shape->what);
            failures++;
            break;
        }

        float magnitude = 0.0f;
        for (size_t i = 0; i < m * k; i++) {
            if (fabsf(cpu_dA[i]) > magnitude) magnitude = fabsf(cpu_dA[i]);
        }
        for (size_t i = 0; i < k * n; i++) {
            if (fabsf(cpu_dB[i]) > magnitude) magnitude = fabsf(cpu_dB[i]);
        }
        float tolerance = 1e-4f * (1.0f + magnitude);

        float d_input = max_abs_diff(cpu_dA, gpu_dA, m * k);
        float d_weight = max_abs_diff(cpu_dB, gpu_dB, k * n);
        if (d_input > worst) worst = d_input;
        if (d_weight > worst) worst = d_weight;

        if (!(d_input <= tolerance)) {
            fprintf(stderr, "FAIL: backward_input diverged on %s (pass %d): %g > %g\n",
                    shape->what, pass + 1, (double)d_input, (double)tolerance);
            failures++;
        }
        if (!(d_weight <= tolerance)) {
            fprintf(stderr, "FAIL: backward_weight diverged on %s (pass %d): %g > %g\n",
                    shape->what, pass + 1, (double)d_weight, (double)tolerance);
            failures++;
        }
    }

    printf("  %-42s %zux%zux%zu  worst diff %.3g\n", shape->what, m, k, n, (double)worst);

    free(A); free(B); free(dC);
    free(cpu_dA); free(gpu_dA); free(cpu_dB); free(gpu_dB);
    return worst;
}

/* A destination that starts at zero and is written once must equal the plain
 * product - the sanity check that the accumulate path did not also break the
 * ordinary case. */
static void check_single_pass_from_zero(void) {
    const size_t m = 12, k = 20, n = 7;
    float A[12 * 20], B[20 * 7], dC[12 * 7];
    float cpu_dA[12 * 20] = {0}, gpu_dA[12 * 20] = {0};
    fill(A, m * k, 1);
    fill(B, k * n, 2);
    fill(dC, m * n, 3);

    matmul_backward_input(dC, B, cpu_dA, m, k, n);
    if (gpu_matmul_backward_input(dC, B, gpu_dA, m, k, n) != 0) {
        fprintf(stderr, "FAIL: single-pass backward_input call failed\n");
        failures++;
        return;
    }
    float diff = max_abs_diff(cpu_dA, gpu_dA, m * k);
    if (!(diff <= 1e-3f)) {
        fprintf(stderr, "FAIL: single pass from zero diverged: %g\n", (double)diff);
        failures++;
    }
}

int main(void) {
    if (!gpu_matmul_available()) {
        printf("SKIP: no usable CUDA GPU detected (see gpu_probe.out for details) - not a failure.\n");
        return 0;
    }

    printf("GPU backward kernels vs the CPU reference, accumulating twice into "
           "a non-zero destination:\n");

    float worst = 0.0f;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        float d = check_shape(&shapes[i]);
        if (d > worst) worst = d;
    }
    check_single_pass_from_zero();

    printf("\nworst difference from the CPU backward reference: %.8g\n", (double)worst);
    gpu_matmul_shutdown();

    if (failures != 0) {
        printf("\nGPU BACKWARD CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nGPU BACKWARD CHECK PASSED\n");
    return 0;
}
