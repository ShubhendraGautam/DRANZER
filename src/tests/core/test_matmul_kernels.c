/*
 * Every selectable CPU matmul kernel, at every tile size the benchmark can
 * sweep, must agree with the portable scalar reference - including on the
 * awkward shapes real models produce (single rows, k or n of 1, extents that
 * are not multiples of the tile, extents larger than the tile).
 *
 * This is what makes the measurement work in tools/bench_matmul.c meaningful:
 * a kernel is only allowed to be faster, never different. It also pins the
 * configuration contract (defaults, validation, name round-trip) that the
 * benchmark and the model dispatch both depend on.
 */

#include "core/matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { size_t m, k, n; } shape_t;

static const shape_t shapes[] = {
    { 1, 1, 1 },        /* degenerate */
    { 1, 16, 16 },      /* single-token decode projection */
    { 1, 16, 260 },     /* single-token output head */
    { 3, 5, 7 },        /* below every tile size, no dimension a multiple */
    { 4, 64, 16 },      /* exactly the register-block row count */
    { 5, 73, 67 },      /* prime-ish, straddles the 64 tile in k and n */
    { 7, 1, 9 },        /* k == 1: one pass, no accumulation */
    { 9, 11, 1 },       /* n == 1: single-column output */
    { 32, 16, 64 },     /* prefill FFN up-projection */
    { 33, 65, 129 },    /* one past the tile in all three extents */
    { 64, 64, 64 },     /* exactly one tile */
};

static const size_t tiles[] = { 1, 3, 16, 64, 128, 256 };

static const matmul_kernel_t kernels[] = {
    MATMUL_KERNEL_AUTO,
    MATMUL_KERNEL_SCALAR,
    MATMUL_KERNEL_ROWWISE,
    MATMUL_KERNEL_TILED,
    MATMUL_KERNEL_TILED_MR4,
};

static int failures;

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    failures++;
}

/* Compare against the scalar reference. Kernels differ only in the order
 * they sum the same products, so the tolerance scales with the magnitude of
 * the result rather than being an absolute epsilon. */
static float compare_all_kernels(const shape_t *shape, float *worst_out) {
    size_t m = shape->m, k = shape->k, n = shape->n;
    float *a = malloc(m * k * sizeof(float));
    float *b = malloc(k * n * sizeof(float));
    float *reference = malloc(m * n * sizeof(float));
    float *output = malloc(m * n * sizeof(float));
    if (!a || !b || !reference || !output) {
        fail("allocation");
        free(a); free(b); free(reference); free(output);
        return -1.0f;
    }

    for (size_t i = 0; i < m * k; i++) a[i] = (float)((int)(i % 37) - 18) / 7.0f;
    for (size_t i = 0; i < k * n; i++) b[i] = (float)((int)((i * 13 + 5) % 41) - 20) / 9.0f;

    matrix_multiply_scalar(a, b, reference, m, k, n);
    float magnitude = 0.0f;
    for (size_t i = 0; i < m * n; i++) {
        if (fabsf(reference[i]) > magnitude) magnitude = fabsf(reference[i]);
    }
    float tolerance = 1e-4f * (1.0f + magnitude);

    float worst = 0.0f;
    for (size_t t = 0; t < sizeof(tiles) / sizeof(tiles[0]); t++) {
        if (matmul_set_tile_size(tiles[t]) != 0) {
            fail("matmul_set_tile_size rejected a positive tile");
            continue;
        }
        for (size_t ki = 0; ki < sizeof(kernels) / sizeof(kernels[0]); ki++) {
            for (size_t i = 0; i < m * n; i++) output[i] = 12345.0f; /* poison */
            matmul_run(kernels[ki], a, b, output, m, k, n);
            for (size_t i = 0; i < m * n; i++) {
                float diff = fabsf(reference[i] - output[i]);
                if (diff > worst) worst = diff;
                if (!(diff <= tolerance)) {
                    fprintf(stderr,
                            "  %s tile %zu shape %zux%zux%zu element %zu: "
                            "%.9g vs reference %.9g\n",
                            matmul_kernel_name(kernels[ki]), tiles[t], m, k, n, i,
                            (double)output[i], (double)reference[i]);
                    fail("kernel diverged from the scalar reference");
                    goto done;
                }
            }
        }
    }

done:
    if (worst > *worst_out) *worst_out = worst;
    free(a); free(b); free(reference); free(output);
    return worst;
}

static void check_configuration_contract(void) {
    if (matmul_get_kernel() != MATMUL_KERNEL_AUTO) {
        fail("the default kernel is not auto");
    }
    if (matmul_tile_size() != DRANZER_MATMUL_BLOCK_SIZE) {
        fail("the default tile is not DRANZER_MATMUL_BLOCK_SIZE");
    }

    size_t before = matmul_tile_size();
    if (matmul_set_tile_size(0) != -1 || matmul_tile_size() != before) {
        fail("a zero tile was accepted or changed the active tile");
    }
    if (matmul_set_kernel((matmul_kernel_t)MATMUL_KERNEL_COUNT) != -1 ||
        matmul_set_kernel((matmul_kernel_t)-1) != -1) {
        fail("an out-of-range kernel was accepted");
    }
    if (matmul_get_kernel() != MATMUL_KERNEL_AUTO) {
        fail("a rejected kernel still changed the active kernel");
    }

    for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++) {
        matmul_kernel_t parsed;
        const char *name = matmul_kernel_name(kernels[i]);
        if (matmul_kernel_from_name(name, &parsed) != 0 || parsed != kernels[i]) {
            fail("kernel name did not round-trip");
        }
    }
    matmul_kernel_t ignored;
    if (matmul_kernel_from_name("tiled_mr8", &ignored) != -1 ||
        matmul_kernel_from_name("", &ignored) != -1 ||
        matmul_kernel_from_name(NULL, &ignored) != -1) {
        fail("an unknown kernel name was accepted");
    }
}

/* The policy has to be a pure function of the shape: a training run that
 * repeats the same call must repeat the same accumulation order, or exact
 * resume stops being exact. */
static void check_selection_is_deterministic(void) {
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        matmul_kernel_t first = matmul_select(shapes[i].m, shapes[i].k, shapes[i].n);
        if (first == MATMUL_KERNEL_AUTO) {
            fail("the policy resolved to auto instead of a concrete kernel");
        }
        for (int repeat = 0; repeat < 4; repeat++) {
            if (matmul_select(shapes[i].m, shapes[i].k, shapes[i].n) != first) {
                fail("the policy is not a pure function of the shape");
            }
        }
    }
    /* The measured policy is currently one kernel for every shape (see
     * matmul.c). Pinning that here is deliberate: if a future sweep adds a
     * shape split, this assertion is what forces the documented policy in
     * docs/matmul.md to be updated with it rather than drifting silently. */
    matmul_kernel_t single_row = matmul_select(1, 256, 4000);
    if (single_row != MATMUL_KERNEL_TILED_MR4) {
        fail("the documented default kernel changed without updating the test");
    }
    if (matmul_select(128, 1024, 256) != single_row ||
        matmul_select(1, 16, 16) != single_row ||
        matmul_select(64, 64, 256) != single_row) {
        fail("selection now splits by shape but docs/matmul.md still documents one kernel");
    }
}

/* A zero-length inner dimension is a defined, if unusual, request: the
 * result is an all-zero matrix, not untouched memory. */
static void check_empty_inner_dimension(void) {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float c[4] = {9.0f, 9.0f, 9.0f, 9.0f};
    matmul_run(MATMUL_KERNEL_AUTO, a, b, c, 2, 0, 2);
    for (size_t i = 0; i < 4; i++) {
        if (c[i] != 0.0f) fail("k == 0 did not produce a zero matrix");
    }
}

int main(void) {
    check_configuration_contract();
    check_selection_is_deterministic();
    check_empty_inner_dimension();

    float worst = 0.0f;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        if (compare_all_kernels(&shapes[i], &worst) < 0.0f) break;
    }
    matmul_set_tile_size(DRANZER_MATMUL_BLOCK_SIZE);

    printf("shapes=%zu tiles=%zu kernels=%zu, worst difference from the scalar "
           "reference: %.8g\n",
           sizeof(shapes) / sizeof(shapes[0]), sizeof(tiles) / sizeof(tiles[0]),
           sizeof(kernels) / sizeof(kernels[0]), (double)worst);
    if (failures != 0) {
        printf("\nMATMUL KERNEL CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nMATMUL KERNEL CHECK PASSED\n");
    return 0;
}
