/*
 * The cutoff that decides whether an OpenMP parallel region is entered at all
 * (core/parallel.h), and the one property it is not allowed to have: an effect
 * on results.
 *
 * The cutoff is a performance policy, so most of what it does cannot be tested
 * by asserting a time. What can be tested, and is what makes the policy safe to
 * tune, is that flipping it changes nothing else:
 *
 *   - Every kernel, forward and backward, must produce bit-identical output
 *     with forking forced on and forced off. Not "within a tolerance" - the
 *     blocks a region distributes are disjoint and no thread reduces into
 *     another's output, so the two paths sum identical products in identical
 *     order. Anything less than bit-identity means a region is not actually
 *     disjoint, which no tolerance-based test would catch reliably.
 *   - The predicate itself refuses to fork what cannot be spread across a team,
 *     regardless of how much work it is.
 *   - The shipped threshold classifies the configurations docs/threading.md
 *     measured the way that document says it does, so moving the constant is a
 *     deliberate act that fails a test rather than a silent retune.
 *
 * In a build without OpenMP the predicate always answers "no" and the identity
 * checks compare serial code against itself, which still exercises every kernel
 * and is still worth running.
 */

#include "core/cpu_features.h"
#include "core/matmul.h"
#include "core/parallel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    failures++;
}

/* Shapes with enough blocks to fork at some tile, plus the awkward ones - a
 * region is most likely to differ from serial code exactly where the block
 * decomposition has a remainder. */
typedef struct { size_t m, k, n; } shape_t;

static const shape_t shapes[] = {
    { 1, 16, 260 },     /* single-token output head: 2 blocks at tile 256 */
    { 1, 64, 1000 },    /* the smallest shape measured to be worth forking */
    { 1, 256, 4000 },   /* medium output head: 16 blocks */
    { 3, 5, 7 },        /* smaller than every tile */
    { 33, 65, 129 },    /* one past the tile in all three extents */
    { 64, 64, 64 },     /* exactly one tile */
    { 128, 256, 300 },  /* multi-row, blocks in both i and j */
    { 130, 40, 70 },    /* row count not a multiple of the register block */
};
#define SHAPE_COUNT (sizeof(shapes) / sizeof(shapes[0]))

static const size_t tiles[] = { 16, 64, 256 };
#define TILE_COUNT (sizeof(tiles) / sizeof(tiles[0]))

static const matmul_kernel_t kernels[] = {
    MATMUL_KERNEL_AUTO,
    MATMUL_KERNEL_ROWWISE,
    MATMUL_KERNEL_TILED,
    MATMUL_KERNEL_TILED_MR4,
    MATMUL_KERNEL_AVX2_MR4,
    MATMUL_KERNEL_AVX512_MR4,
    MATMUL_KERNEL_NEON_MR4,
};
#define KERNEL_COUNT (sizeof(kernels) / sizeof(kernels[0]))

/* Force forking on or off for every shape regardless of size. A threshold of 1
 * admits everything with at least two blocks; SIZE_MAX admits nothing. */
#define FORK_ALWAYS ((size_t)1)
#define FORK_NEVER  ((size_t)-1)

static void fill(float *buffer, size_t count, int salt) {
    for (size_t i = 0; i < count; i++) {
        buffer[i] = (float)((int)((i * 13 + (size_t)salt * 7) % 37) - 18) / 7.0f;
    }
}

/* Bit comparison, not an epsilon: see the file header for why. */
static int identical(const float *left, const float *right, size_t count) {
    return memcmp(left, right, count * sizeof(float)) == 0;
}

/* How many (shape, tile) pairs actually produced a region under FORK_ALWAYS.
 * Without this the identity checks would pass just as cleanly on a run where
 * nothing ever forked - comparing serial code against serial code - and report
 * the same "PASSED" for a much weaker check. */
static size_t forkable_combinations;

static void count_forkable(void) {
    for (size_t s = 0; s < SHAPE_COUNT; s++) {
        for (size_t t = 0; t < TILE_COUNT; t++) {
            /* The blocked kernels' decomposition, from core/matmul.c. */
            size_t tile = tiles[t];
            size_t chunks = ((shapes[s].m + tile - 1) / tile) *
                            ((shapes[s].n + tile - 1) / tile);
            if (parallel_should_fork(chunks, FORK_ALWAYS)) forkable_combinations++;
        }
    }
}

static void check_forward_identity(void) {
    for (size_t s = 0; s < SHAPE_COUNT; s++) {
        size_t m = shapes[s].m, k = shapes[s].k, n = shapes[s].n;
        float *a = malloc(m * k * sizeof(float));
        float *b = malloc(k * n * sizeof(float));
        float *forked = malloc(m * n * sizeof(float));
        float *serial = malloc(m * n * sizeof(float));
        if (!a || !b || !forked || !serial) {
            fail("allocation");
            free(a); free(b); free(forked); free(serial);
            return;
        }
        fill(a, m * k, 1);
        fill(b, k * n, 2);

        for (size_t t = 0; t < TILE_COUNT; t++) {
            matmul_set_tile_size(tiles[t]);
            for (size_t ki = 0; ki < KERNEL_COUNT; ki++) {
                if (!matmul_kernel_available(kernels[ki])) continue;

                parallel_set_min_work(FORK_ALWAYS);
                matmul_run(kernels[ki], a, b, forked, m, k, n);
                parallel_set_min_work(FORK_NEVER);
                matmul_run(kernels[ki], a, b, serial, m, k, n);

                if (!identical(forked, serial, m * n)) {
                    fprintf(stderr, "  %s tile %zu shape %zux%zux%zu\n",
                            matmul_kernel_name(kernels[ki]), tiles[t], m, k, n);
                    fail("forked and serial results differ");
                }
            }
        }
        free(a); free(b); free(forked); free(serial);
    }
}

/* The backward functions accumulate into their destination, so they are run
 * twice into a non-zero buffer: a region that raced on the += would show up
 * here and not in a single fresh call. */
static void check_backward_identity(void) {
    for (size_t s = 0; s < SHAPE_COUNT; s++) {
        size_t m = shapes[s].m, k = shapes[s].k, n = shapes[s].n;
        float *dc = malloc(m * n * sizeof(float));
        float *b = malloc(k * n * sizeof(float));
        float *a = malloc(m * k * sizeof(float));
        float *forked = malloc((k * n > m * k ? k * n : m * k) * sizeof(float));
        float *serial = malloc((k * n > m * k ? k * n : m * k) * sizeof(float));
        if (!dc || !b || !a || !forked || !serial) {
            fail("allocation");
            free(dc); free(b); free(a); free(forked); free(serial);
            return;
        }
        fill(dc, m * n, 3);
        fill(b, k * n, 4);
        fill(a, m * k, 5);

        /* dA (m x k) from dC and B */
        for (size_t pass = 0; pass < 2; pass++) {
            size_t count = m * k;
            if (pass == 0) { fill(forked, count, 6); fill(serial, count, 6); }
            parallel_set_min_work(FORK_ALWAYS);
            matmul_backward_input(dc, b, forked, m, k, n);
            parallel_set_min_work(FORK_NEVER);
            matmul_backward_input(dc, b, serial, m, k, n);
            if (!identical(forked, serial, count)) {
                fprintf(stderr, "  matmul_backward_input shape %zux%zux%zu pass %zu\n",
                        m, k, n, pass);
                fail("forked and serial backward-input results differ");
                break;
            }
        }

        /* dB (k x n) from A and dC */
        for (size_t pass = 0; pass < 2; pass++) {
            size_t count = k * n;
            if (pass == 0) { fill(forked, count, 7); fill(serial, count, 7); }
            parallel_set_min_work(FORK_ALWAYS);
            matmul_backward_weight(a, dc, forked, m, k, n);
            parallel_set_min_work(FORK_NEVER);
            matmul_backward_weight(a, dc, serial, m, k, n);
            if (!identical(forked, serial, count)) {
                fprintf(stderr, "  matmul_backward_weight shape %zux%zux%zu pass %zu\n",
                        m, k, n, pass);
                fail("forked and serial backward-weight results differ");
                break;
            }
        }

        free(dc); free(b); free(a); free(forked); free(serial);
    }
}

static void check_predicate(void) {
    const size_t huge = (size_t)1 << 40;

    /* Nothing to distribute is never worth a region, at any amount of work. */
    parallel_set_min_work(FORK_ALWAYS);
    if (parallel_should_fork(0, huge)) fail("forked a loop with no iterations");
    if (parallel_should_fork(1, huge)) fail("forked a loop with one iteration");

    /* Below the threshold, no number of chunks helps. */
    parallel_set_min_work(1000);
    if (parallel_should_fork(1024, 999)) fail("forked below the work threshold");
    if (parallel_should_fork(1024, 1000) != (parallel_max_threads() > 1)) {
        fail("threshold is not inclusive, or disagrees with the team size");
    }

    /* A single-thread team has nothing to gain either. A build without OpenMP
     * reports one thread and must always answer no. */
    if (parallel_max_threads() < 1) fail("thread count below one");
#ifndef _OPENMP
    if (parallel_max_threads() != 1) fail("threads reported without OpenMP");
    if (parallel_should_fork(1024, huge)) fail("forked in a build without OpenMP");
#endif
}

static void check_setter_contract(void) {
    size_t original = parallel_min_work();

    if (parallel_set_min_work(0) != -1) fail("accepted a zero threshold");
    if (parallel_min_work() != original) {
        fail("a rejected threshold changed the setting");
    }
    if (parallel_set_min_work(4096) != 0) fail("rejected a valid threshold");
    if (parallel_min_work() != 4096) fail("threshold did not round-trip");

    parallel_set_min_work(original);
}

/* The verdicts docs/threading.md records, restated as assertions.
 *
 * These are not re-measurements - they pin which side of the shipped threshold
 * each measured configuration lands on. Changing DRANZER_PARALLEL_MIN_WORK
 * fails this until the document and the measurement behind it are updated too,
 * which is the point: the constant was bracketed by a narrow range of evidence
 * and should not drift without new evidence. */
static void check_measured_verdicts(void) {
    struct { const char *what; size_t chunks, work; int should_fork; } cases[] = {
        /* matmul: 1x16x260 at tile 256 is two blocks, and forking it measured
         * 0.21x-0.25x - the largest shape measured to lose. */
        { "tiny output-head matmul (1x16x260)", 2, 1 * 16 * 260, 0 },
        /* matmul: 1x64x1000, four blocks, measured 1.16x-1.37x - the smallest
         * shape measured to win. */
        { "small output-head matmul (1x64x1000)", 4, 1 * 64 * 1000, 1 },
        /* Decode head loop, tiny tier: 2 heads, context 32, embedding 16.
         * Serializing it was worth 2.3x on sliding decode. */
        { "tiny decode head loop", 2,
          2 * 32 * 16 + 2 * 32 * DRANZER_PARALLEL_SOFTMAX_WORK, 0 },
        /* Decode head loop, small tier: 4 heads, context 64, embedding 64.
         * Serializing this one cost 1.20x, 1.19x and 1.45x, which is what
         * moved the threshold down to admit it. */
        { "small decode head loop", 4,
          2 * 64 * 64 + 4 * 64 * DRANZER_PARALLEL_SOFTMAX_WORK, 1 },
        /* Medium tier prefill, comfortably above: 128x256x1024, measured
         * 1.31x-2.62x. */
        { "medium prefill matmul (128x256x1024)", 4, (size_t)128 * 256 * 1024, 1 },
    };
    const size_t count = sizeof(cases) / sizeof(cases[0]);

    for (size_t i = 0; i < count; i++) {
        /* The predicate also depends on the team size, which varies by machine
         * and is not what this is pinning. Compare the work/chunks decision
         * alone by asking what it would answer given more than one thread. */
        int decided = cases[i].chunks >= 2 && cases[i].work >= parallel_min_work();
        if (decided != cases[i].should_fork) {
            fprintf(stderr,
                    "  %s: work %zu against threshold %zu decides %s, "
                    "docs/threading.md measured %s\n",
                    cases[i].what, cases[i].work, parallel_min_work(),
                    decided ? "fork" : "serial",
                    cases[i].should_fork ? "fork" : "serial");
            fail("the shipped threshold contradicts a recorded measurement");
        }
    }
}

int main(void) {
    cpu_features_detect();

    const size_t original_min_work = parallel_min_work();
    const size_t original_tile = matmul_tile_size();

    printf("threads=%d cutoff=%zu simd=%s\n",
           parallel_max_threads(), parallel_min_work(), cpu_features_summary());
#ifndef _OPENMP
    printf("built without OpenMP: the predicate must always answer no, and the\n"
           "identity checks compare serial code against itself\n");
#endif

    check_measured_verdicts();
    check_predicate();
    check_setter_contract();

    parallel_set_min_work(FORK_ALWAYS);
    count_forkable();

    check_forward_identity();
    check_backward_identity();

    parallel_set_min_work(original_min_work);
    matmul_set_tile_size(original_tile);

    printf("shapes=%zu tiles=%zu kernels=%zu, forked in %zu of %zu shape/tile pairs\n",
           SHAPE_COUNT, TILE_COUNT, KERNEL_COUNT, forkable_combinations,
           SHAPE_COUNT * TILE_COUNT);
#ifdef _OPENMP
    if (forkable_combinations == 0 && parallel_max_threads() > 1) {
        fail("no shape produced a parallel region: the identity checks above "
             "compared serial code against itself and proved nothing");
    }
#endif
    if (failures != 0) {
        printf("\nPARALLEL CUTOFF CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nPARALLEL CUTOFF CHECK PASSED\n");
    return 0;
}
