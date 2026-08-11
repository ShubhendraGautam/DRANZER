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
 *
 * Runtime-dispatched SIMD kernels are covered by exactly the same sweep. They
 * are checked against the same reference on the same shapes, and the shape
 * list carries extents around the 8- and 16-float vector widths so the
 * vectorized body, its scalar tail, and the AVX-512 masked tail are all
 * exercised. Kernels this machine cannot run are skipped and reported, never
 * silently counted as passing.
 */

#include "core/cpu_features.h"
#include "core/matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    /* Vector-width boundaries: 8 floats for AVX2, 16 for AVX-512, 4 for NEON.
     * A kernel that mishandles its tail is wrong on exactly these. */
    { 4, 8, 8 },        /* exactly one AVX2 vector, no tail */
    { 4, 8, 15 },       /* one short of an AVX-512 vector: all tail */
    { 4, 8, 17 },       /* one past an AVX-512 vector: body plus 1-wide tail */
    { 2, 3, 31 },       /* fewer than four rows and a tail in both widths */
};

static const size_t tiles[] = { 1, 3, 16, 64, 128, 256 };

static const matmul_kernel_t kernels[] = {
    MATMUL_KERNEL_AUTO,
    MATMUL_KERNEL_SCALAR,
    MATMUL_KERNEL_ROWWISE,
    MATMUL_KERNEL_TILED,
    MATMUL_KERNEL_TILED_MR4,
    MATMUL_KERNEL_AVX2_MR4,
    MATMUL_KERNEL_AVX512_MR4,
    MATMUL_KERNEL_NEON_MR4,
};

#define KERNEL_COUNT (sizeof(kernels) / sizeof(kernels[0]))

static int failures;

/* Which kernels actually ran, so the summary can distinguish "checked and
 * agreed" from "not available on this CPU". Without this a run on a machine
 * without AVX-512 looks identical to one that checked it. */
static int kernel_exercised[MATMUL_KERNEL_COUNT];

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
        for (size_t ki = 0; ki < KERNEL_COUNT; ki++) {
            /* A kernel compiled for another architecture, or for instructions
             * this CPU lacks, would be run as the portable fallback by
             * matmul_run(). Timing or comparing that would be measuring
             * tiled_mr4 twice under a different name. */
            if (!matmul_kernel_available(kernels[ki])) continue;
            kernel_exercised[kernels[ki]] = 1;

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

    /* Names must round-trip for every kernel on every architecture, including
     * the ones that cannot run here. A config file or a --kernel flag written
     * on one machine has to remain parseable on another; refusing the name
     * would turn a portable fallback into a startup failure. */
    for (size_t i = 0; i < KERNEL_COUNT; i++) {
        matmul_kernel_t parsed;
        const char *name = matmul_kernel_name(kernels[i]);
        if (matmul_kernel_from_name(name, &parsed) != 0 || parsed != kernels[i]) {
            fail("kernel name did not round-trip");
        }
        if (matmul_set_kernel(kernels[i]) != 0) {
            fail("a valid kernel was rejected because it is unavailable here");
        }
    }
    matmul_set_kernel(MATMUL_KERNEL_AUTO);

    matmul_kernel_t ignored;
    if (matmul_kernel_from_name("tiled_mr8", &ignored) != -1 ||
        matmul_kernel_from_name("", &ignored) != -1 ||
        matmul_kernel_from_name(NULL, &ignored) != -1) {
        fail("an unknown kernel name was accepted");
    }

    /* The portable kernels are unconditionally available; a SIMD kernel is
     * available only where its ISA is. */
    if (!matmul_kernel_available(MATMUL_KERNEL_SCALAR) ||
        !matmul_kernel_available(MATMUL_KERNEL_TILED_MR4)) {
        fail("a portable kernel reported itself unavailable");
    }
    if (matmul_kernel_isa(MATMUL_KERNEL_TILED_MR4) != CPU_ISA_BASELINE ||
        matmul_kernel_isa(MATMUL_KERNEL_AVX512_MR4) != CPU_ISA_AVX512) {
        fail("a kernel reported the wrong instruction-set requirement");
    }
    for (size_t i = 0; i < KERNEL_COUNT; i++) {
        if (matmul_kernel_available(kernels[i]) &&
            !cpu_isa_available(matmul_kernel_isa(kernels[i]))) {
            fail("a kernel is available but its instruction set is not");
        }
    }
}

/* What the policy should answer here, derived the same way the policy derives
 * it: with runtime dispatch the right answer depends on the CPU, so a literal
 * would only pass on the machine that wrote it. What is pinned is the rule.
 *
 * This is now the widest available kernel, which it was not always: avx2_mr4
 * was excluded while it measured slower than the portable kernel under Clang.
 * Rewriting it to keep accumulators in registers fixed that (see matmul.c), so
 * the exclusion was lifted on new measurements. neon_mr4 stays excluded for
 * want of AArch64 hardware to measure it on, and the check below pins that. */
static matmul_kernel_t expected_simd_selection(void) {
    if (matmul_kernel_available(MATMUL_KERNEL_AVX512_MR4)) {
        return MATMUL_KERNEL_AVX512_MR4;
    }
    if (matmul_kernel_available(MATMUL_KERNEL_AVX2_MR4)) {
        return MATMUL_KERNEL_AVX2_MR4;
    }
    return MATMUL_KERNEL_TILED_MR4;
}

/* neon_mr4 must stay unselected on every shape until someone measures it on
 * real AArch64 hardware. */
static void check_unselected_kernels_stay_unselected(void) {
    static const shape_t probes[] = {
        { 1, 256, 4000 }, { 128, 1024, 256 }, { 64, 64, 256 }, { 1, 16, 16 },
    };
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        matmul_kernel_t chosen =
            matmul_select(probes[i].m, probes[i].k, probes[i].n);
        if (chosen == MATMUL_KERNEL_NEON_MR4) {
            fail("neon_mr4 became the default while still unmeasured");
        }
    }
}

/* The policy has to be a pure function of the shape: a training run that
 * repeats the same call must repeat the same accumulation order, or exact
 * resume stops being exact. Runtime dispatch does not weaken this - the ISA
 * is resolved once from the hardware, not per call - but it does mean the
 * answer is only fixed within a process, which is what the exactness contract
 * in docs/matmul.md now says. */
static void check_selection_is_deterministic(void) {
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        matmul_kernel_t first = matmul_select(shapes[i].m, shapes[i].k, shapes[i].n);
        if (first == MATMUL_KERNEL_AUTO) {
            fail("the policy resolved to auto instead of a concrete kernel");
        }
        if (!matmul_kernel_available(first)) {
            fail("the policy selected a kernel this machine cannot run");
        }
        for (int repeat = 0; repeat < 4; repeat++) {
            if (matmul_select(shapes[i].m, shapes[i].k, shapes[i].n) != first) {
                fail("the policy is not a pure function of the shape");
            }
        }
    }

    /* Pin the one measured shape split and its boundaries. The small and
     * medium single-token FFN expansions use rowwise; tiny and non-FFN decode
     * shapes keep the ordinary SIMD preference. */
    matmul_kernel_t ordinary = expected_simd_selection();
    if (matmul_select(1, 64, 256) != MATMUL_KERNEL_ROWWISE ||
        matmul_select(1, 256, 1024) != MATMUL_KERNEL_ROWWISE) {
        fail("single-token FFN expansion did not select rowwise");
    }
    if (matmul_select(1, 16, 64) != ordinary) {
        fail("tiny FFN expansion crossed the measured rowwise boundary");
    }
    if (matmul_select(1, 63, 252) != ordinary ||
        matmul_select(1, 64, 255) != ordinary) {
        fail("rowwise selection escaped its measured width or shape boundary");
    }
    if (matmul_select(1, 256, 4000) != ordinary) {
        fail("the documented preference order changed without updating the test");
    }
    if (matmul_select(128, 1024, 256) != ordinary ||
        matmul_select(1, 16, 16) != ordinary ||
        matmul_select(64, 64, 256) != ordinary) {
        fail("an unmeasured shape entered the rowwise dispatch case");
    }
}

/* The acceptance gate for runtime dispatch: a binary carrying SIMD kernels
 * must still be correct on a machine that cannot execute them.
 *
 * That machine cannot be provisioned from inside a test, so the ISA cap
 * simulates it at the only layer that decides anything - the dispatch. Capping
 * to baseline is exactly what detection would have reported on such a CPU, so
 * every branch below it runs the code that machine would run. What this does
 * not prove is that the instructions themselves are absent from the reachable
 * path, which is a property of the target attributes rather than of dispatch;
 * docs/matmul.md records that distinction.
 */
static void check_fallback_without_simd(void) {
    const size_t m = 6, k = 20, n = 37; /* tails in every vector width */
    float *a = malloc(m * k * sizeof(float));
    float *b = malloc(k * n * sizeof(float));
    float *reference = malloc(m * n * sizeof(float));
    float *output = malloc(m * n * sizeof(float));
    if (!a || !b || !reference || !output) {
        fail("allocation");
        free(a); free(b); free(reference); free(output);
        return;
    }
    for (size_t i = 0; i < m * k; i++) a[i] = (float)((int)(i % 29) - 14) / 5.0f;
    for (size_t i = 0; i < k * n; i++) b[i] = (float)((int)((i * 7 + 3) % 31) - 15) / 6.0f;
    matrix_multiply_scalar(a, b, reference, m, k, n);
    float magnitude = 0.0f;
    for (size_t i = 0; i < m * n; i++) {
        if (fabsf(reference[i]) > magnitude) magnitude = fabsf(reference[i]);
    }
    const float tolerance = 1e-4f * (1.0f + magnitude);

    if (cpu_features_set_max_isa(CPU_ISA_BASELINE) != 0) {
        fail("capping the instruction set to baseline was rejected");
    }
    if (cpu_isa_best() != CPU_ISA_BASELINE) {
        fail("baseline cap did not remove every SIMD instruction set");
    }
    for (size_t i = 0; i < KERNEL_COUNT; i++) {
        if (matmul_kernel_isa(kernels[i]) == CPU_ISA_BASELINE) continue;
        if (matmul_kernel_available(kernels[i])) {
            fail("a SIMD kernel is still available under a baseline cap");
        }
    }
    if (matmul_select(m, k, n) != MATMUL_KERNEL_TILED_MR4) {
        fail("selection did not fall back to the portable kernel");
    }
    if (matmul_select(1, 64, 256) != MATMUL_KERNEL_ROWWISE) {
        fail("the measured rowwise case changed under a baseline ISA cap");
    }

    /* Every kernel, including the SIMD ones nothing can execute now, must
     * still produce the right answer through the fallback rather than
     * faulting or returning the poison. */
    for (size_t ki = 0; ki < KERNEL_COUNT; ki++) {
        for (size_t i = 0; i < m * n; i++) output[i] = 12345.0f;
        matmul_run(kernels[ki], a, b, output, m, k, n);
        for (size_t i = 0; i < m * n; i++) {
            if (!(fabsf(reference[i] - output[i]) <= tolerance)) {
                fprintf(stderr, "  %s under a baseline cap: element %zu is "
                        "%.9g, reference %.9g\n",
                        matmul_kernel_name(kernels[ki]), i,
                        (double)output[i], (double)reference[i]);
                fail("a kernel was wrong when its instruction set was unavailable");
                break;
            }
        }
    }

    cpu_features_clear_max_isa();
    if (matmul_select(m, k, n) != expected_simd_selection()) {
        fail("clearing the cap did not restore the selected kernel");
    }
    free(a); free(b); free(reference); free(output);
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
    printf("cpu: %s\n", cpu_features_summary());
    printf("auto selects: %s\n\n",
           matmul_kernel_name(matmul_select(1, 64, 64)));

    check_configuration_contract();
    check_selection_is_deterministic();
    check_unselected_kernels_stay_unselected();
    check_empty_inner_dimension();

    float worst = 0.0f;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        if (compare_all_kernels(&shapes[i], &worst) < 0.0f) break;
    }
    matmul_set_tile_size(DRANZER_MATMUL_BLOCK_SIZE);

    /* Run last: it leaves the process capped for the duration and restores
     * the cap on the way out, so nothing above can be affected by it. */
    check_fallback_without_simd();

    /* Name which kernels were actually compared, not just how many exist. A
     * pass on a machine without AVX-512 is a weaker result than a pass on one
     * with it, and the output has to say so. */
    enum { NAME_LIST_SIZE = 192 };
    size_t checked = 0, skipped = 0;
    char checked_names[NAME_LIST_SIZE] = "", skipped_names[NAME_LIST_SIZE] = "";
    for (size_t i = 0; i < KERNEL_COUNT; i++) {
        int was_run = kernel_exercised[kernels[i]];
        char *list = was_run ? checked_names : skipped_names;
        size_t *count = was_run ? &checked : &skipped;
        size_t used = strlen(list);
        snprintf(list + used, NAME_LIST_SIZE - used, "%s%s", used ? ", " : "",
                 matmul_kernel_name(kernels[i]));
        (*count)++;
    }

    printf("shapes=%zu tiles=%zu, kernels checked=%zu (%s)\n",
           sizeof(shapes) / sizeof(shapes[0]), sizeof(tiles) / sizeof(tiles[0]),
           checked, checked_names);
    if (skipped != 0) {
        printf("kernels skipped=%zu (%s) - not available on this CPU\n",
               skipped, skipped_names);
    }
    printf("worst difference from the scalar reference: %.8g\n", (double)worst);
    if (failures != 0) {
        printf("\nMATMUL KERNEL CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nMATMUL KERNEL CHECK PASSED\n");
    return 0;
}
