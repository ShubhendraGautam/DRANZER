/* Compares gpu_matmul() (hand-written PTX, via the CUDA Driver API) against
 * core/matmul.c's matrix_multiply() (the CPU reference used everywhere else
 * in this project) across a range of shapes. Self-skips (reports success,
 * not failure) on any machine without a usable CUDA GPU - this test is
 * inherently hardware-dependent, unlike every other test in this suite, and
 * its absence of a GPU is not a regression.
 *
 * The shipped kernel stages each 16x16 output tile through shared memory, so
 * the shapes below are chosen around that tile rather than arbitrarily. A
 * tiled kernel has three distinct ways to be wrong and each needs its own
 * shape:
 *
 *   - the ragged edge, where a tile hangs off the end of the matrix and the
 *     out-of-range lanes must contribute zero rather than garbage;
 *   - the k tail, where the last tile along the reduction axis is partial;
 *   - degenerate extents (1, or below one tile), where whole tiles are edge.
 *
 * A kernel that mishandles any of these still passes on a shape whose extents
 * are all multiples of 16, which is why none of the shapes here are.
 *
 * DRANZER_GPU_MATMUL=naive selects the older non-tiled kernel, kept as the
 * comparison baseline; running this test with it set checks that one too.
 */
#include "core/matmul.h"
#include "backends/gpu/gpu_matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct { size_t m, k, n; const char *what; } shape_t;

static const shape_t shapes[] = {
    { 37,  53,  41,  "no extent a multiple of the tile" },
    { 16,  16,  16,  "exactly one tile, no edge at all" },
    { 17,  17,  17,  "one past a tile in every extent" },
    { 15,  15,  15,  "one short of a tile: every tile is edge" },
    { 1,   64,  64,  "single row, as single-token decode issues" },
    { 64,  64,  1,   "single column" },
    { 1,   1,   1,   "degenerate" },
    { 3,   5,   7,   "smaller than one tile in every extent" },
    { 64,  256, 64,  "prefill FFN down" },
    { 128, 256, 129, "large, with a one-wide column of edge tiles" },
    { 33,  1,   33,  "k of 1: a single partial reduction tile" },
};

static int failures;

int main(void) {
    if (!gpu_matmul_available()) {
        printf("SKIP: no usable CUDA GPU detected (see gpu_probe.out for details) - not a failure.\n");
        return 0;
    }

    float worst_overall = 0.0f;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        size_t m = shapes[s].m, k = shapes[s].k, n = shapes[s].n;
        float *A = malloc(m * k * sizeof(float));
        float *B = malloc(k * n * sizeof(float));
        float *C_gpu = malloc(m * n * sizeof(float));
        float *C_cpu = malloc(m * n * sizeof(float));
        if (!A || !B || !C_gpu || !C_cpu) {
            fprintf(stderr, "FAIL: allocation for %s\n", shapes[s].what);
            failures++;
            free(A); free(B); free(C_gpu); free(C_cpu);
            continue;
        }

        for (size_t i = 0; i < m * k; i++) A[i] = (float)((int)(i % 7) - 3) * 0.5f;
        for (size_t i = 0; i < k * n; i++) B[i] = (float)((int)(i % 5) - 2) * 0.25f;
        /* Poison the output: a kernel that never writes an element, which is
         * exactly what a mishandled edge tile does, must not pass by leaving
         * a plausible-looking value behind. */
        for (size_t i = 0; i < m * n; i++) C_gpu[i] = 12345.0f;

        matrix_multiply(A, B, C_cpu, m, k, n);

        if (gpu_matmul(A, B, C_gpu, m, k, n) != 0) {
            fprintf(stderr, "FAIL: gpu_matmul() failed on %s (%zux%zux%zu)\n",
                    shapes[s].what, m, k, n);
            failures++;
            free(A); free(B); free(C_gpu); free(C_cpu);
            continue;
        }

        float magnitude = 0.0f, worst = 0.0f;
        for (size_t i = 0; i < m * n; i++) {
            float d = fabsf(C_gpu[i] - C_cpu[i]);
            if (d > worst) worst = d;
            if (fabsf(C_cpu[i]) > magnitude) magnitude = fabsf(C_cpu[i]);
        }
        float tolerance = 1e-4f * (1.0f + magnitude);
        if (!(worst <= tolerance)) {
            fprintf(stderr, "FAIL: %s (%zux%zux%zu): worst %g exceeds %g\n",
                    shapes[s].what, m, k, n, (double)worst, (double)tolerance);
            failures++;
        }
        if (worst > worst_overall) worst_overall = worst;
        printf("  %-42s %zux%zux%zu  worst %.3g\n", shapes[s].what, m, k, n, (double)worst);

        free(A); free(B); free(C_gpu); free(C_cpu);
    }

    printf("\nshapes=%zu, worst difference from the CPU reference: %.8g\n",
           sizeof(shapes) / sizeof(shapes[0]), (double)worst_overall);
    gpu_matmul_shutdown();

    if (failures != 0) {
        printf("\nGPU MATMUL CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nGPU MATMUL CHECK PASSED\n");
    return 0;
}
