/*
 * GPU latency invariants.
 *
 * The GPU backend's behaviour is dominated by a fixed per-call cost - driver
 * round-trips for the launch and the two transfers - rather than by
 * arithmetic. Every conclusion this project has drawn about when offload pays
 * rests on that cost, and until now it was only ever observed by running
 * gpu_latency.out and reading the table. This file asserts the structural
 * facts behind those conclusions so they cannot quietly stop being true.
 *
 * Same rule as tests/perf/test_perf_invariants.c: every assertion is a RATIO
 * between measurements taken back to back in this process. No absolute
 * microsecond count is asserted anywhere, because the correct value differs
 * per card, per driver, and per host. What is asserted is the *shape* of the
 * cost - that overhead is amortized by larger work, that the weight cache
 * removes a transfer, that the shipped kernel beats the one it replaced.
 *
 * Self-skips without a usable CUDA GPU, like every test in tests/gpu/.
 */

#include "backends/gpu/gpu_matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tools/timing_spread.h"

#define ROUNDS 5

static int failures;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct { float *a, *b, *c; size_t m, k, n; } case_t;

static case_t make_case(size_t m, size_t k, size_t n) {
    case_t t = { malloc(m * k * sizeof(float)), malloc(k * n * sizeof(float)),
                 malloc(m * n * sizeof(float)), m, k, n };
    if (!t.a || !t.b || !t.c) { fprintf(stderr, "allocation failed\n"); exit(1); }
    for (size_t i = 0; i < m * k; i++) t.a[i] = (float)(i % 7) * 0.01f;
    for (size_t i = 0; i < k * n; i++) t.b[i] = (float)(i % 5) * 0.01f;
    return t;
}

static void free_case(case_t *t) { free(t->a); free(t->b); free(t->c); }

/* Cases the comparisons below run on. File-static because
 * timing_measure_speedup() takes argument-less functions - it has to, so that
 * the two arms it interleaves are indistinguishable to it. */
static case_t cache_case;
static case_t kernel_case;

/* Seconds per call, fastest of ROUNDS. The warm-up call also populates the
 * weight cache, so what is timed is steady state rather than a first upload. */
static double best_call_sec(case_t *t, int calls) {
    gpu_matmul(t->a, t->b, t->c, t->m, t->k, t->n);
    double best = -1.0;
    for (int r = 0; r < ROUNDS; r++) {
        double t0 = now_sec();
        for (int i = 0; i < calls; i++) {
            gpu_matmul(t->a, t->b, t->c, t->m, t->k, t->n);
        }
        double dt = (now_sec() - t0) / calls;
        if (best < 0.0 || dt < best) best = dt;
    }
    return best;
}

static void expect_speedup(const char *what, double speedup, double minimum,
                           const char *explanation) {
    int ok = speedup >= minimum;
    printf("  %-50s %7.2fx  (min %.2fx)  %s\n",
           what, speedup, minimum, ok ? "ok" : "FAIL");
    if (!ok) {
        fprintf(stderr, "FAIL: %s is %.2fx, below the %.2fx minimum.\n       %s\n",
                what, speedup, minimum, explanation);
        failures++;
    }
}

/* A call on a 1-element matmul does essentially no arithmetic, so its cost is
 * the fixed overhead. A call doing millions of times more arithmetic must not
 * cost millions of times more - if it did, there would be no overhead to
 * amortize and the whole threshold model in core/training.c would be wrong.
 *
 * Stated as a ratio of *efficiency*, not of time: the large shape must extract
 * far more FLOP per second than the tiny one. */
static void check_overhead_is_amortized(void) {
    case_t tiny = make_case(1, 1, 1);
    case_t large = make_case(128, 256, 1024);

    double tiny_s = best_call_sec(&tiny, 50);
    double large_s = best_call_sec(&large, 20);

    double tiny_flops = 2.0 * 1 * 1 * 1 / tiny_s;
    double large_flops = 2.0 * 128 * 256 * 1024 / large_s;

    printf("\nfixed per-call overhead is amortized by larger work:\n");
    printf("    1x1x1        %8.1f us   %12.3f GFLOP/s\n", tiny_s * 1e6, tiny_flops / 1e9);
    printf("    128x256x1024 %8.1f us   %12.3f GFLOP/s\n", large_s * 1e6, large_flops / 1e9);
    expect_speedup("throughput of a large call vs a 1-element call",
                   large_flops / tiny_flops, 1000.0,
                   "A GPU call has a fixed cost independent of size. If a large "
                   "call is not vastly more efficient than a trivial one, either "
                   "the kernel is not running or the overhead has exploded.");

    /* The same fact from the other side: a call doing 33 million times the
     * arithmetic must not take anywhere near 33 million times as long. */
    expect_speedup("work grows 33.5M x while time grows less than 100x",
                   100.0 / (large_s / tiny_s), 1.0,
                   "This is the amortization the GPU dispatch thresholds depend on.");

    free_case(&tiny);
    free_case(&large);
}

/* The weight cache exists so B is uploaded once rather than per call. With it
 * working, repeated calls reusing the same B are cheaper than calls that
 * invalidate it every time. This is the only assertion here that would catch
 * the cache being disabled - a correctness test cannot, because an
 * always-uploading cache returns identical results. */
static void op_cached(void) {
    for (int i = 0; i < 20; i++) {
        gpu_matmul(cache_case.a, cache_case.b, cache_case.c,
                   cache_case.m, cache_case.k, cache_case.n);
    }
}

static void op_invalidated(void) {
    for (int i = 0; i < 20; i++) {
        gpu_matmul_invalidate_weights();      /* force a re-upload of B */
        gpu_matmul(cache_case.a, cache_case.b, cache_case.c,
                   cache_case.m, cache_case.k, cache_case.n);
    }
}

static void check_weight_cache_removes_a_transfer(void) {
    cache_case = make_case(128, 256, 1024);

    /* Interleaved and asserted at the median, for the same reason as the kernel
     * comparison below: this one has the narrowest margin of the four checks
     * here (a 1.05x floor against a ratio that has read anywhere from 1.54x to
     * 2.76x on this machine), so it is the next one that would have started
     * flaking. */
    timing_spread_t spread = timing_measure_speedup(op_cached, op_invalidated);

    printf("\nweight cache removes a per-call upload:\n");
    if (!timing_expect_median_speedup("cached vs re-uploading B every call",
                                      spread, 1.05)) {
        fprintf(stderr, "FAIL: cached vs re-uploading has a median of %.2fx over "
                        "%zu replicates (spread %.2fx-%.2fx), below the 1.05x "
                        "minimum.\n       B is 1 MB at this shape. If "
                        "invalidating the cache costs nothing, the cache is not "
                        "being consulted.\n",
                spread.median, spread.replicates, spread.minimum, spread.maximum);
        failures++;
    }

    free_case(&cache_case);
}

static void op_tiled(void) {
    gpu_matmul_set_forward_kernel("tiled");
    for (int i = 0; i < 20; i++) {
        gpu_matmul(kernel_case.a, kernel_case.b, kernel_case.c,
                   kernel_case.m, kernel_case.k, kernel_case.n);
    }
}

static void op_naive(void) {
    gpu_matmul_set_forward_kernel("naive");
    for (int i = 0; i < 20; i++) {
        gpu_matmul(kernel_case.a, kernel_case.b, kernel_case.c,
                   kernel_case.m, kernel_case.k, kernel_case.n);
    }
}

/* The shipped tiled kernel must not be slower than the naive one it replaced.
 *
 * This is the assertion that made the flakiness of the old form obvious. It
 * measured each kernel once, in its own phase, took the fastest of five rounds
 * for each, and divided - and on an unmodified tree that quotient read 0.83x
 * once and 1.01x, 1.05x, 1.11x, 1.14x, 1.20x, 1.21x, 1.31x, and 1.42x on the
 * eight runs after it. A 1.00x floor sits inside that spread, so the check
 * failed a correct tree at some rate, for reasons having nothing to do with the
 * change under review.
 *
 * Now: the two kernels are timed alternately within each replicate and the
 * median of the replicates is asserted on, with the spread printed. The kernel
 * selection is inside each timed function so it applies to the calls it is meant
 * to - one selection per 20 matmuls is not measurable against them. */
static void check_tiled_kernel_is_not_a_regression(void) {
    kernel_case = make_case(128, 256, 1024);

    if (gpu_matmul_set_forward_kernel("naive") != 0 ||
        gpu_matmul_set_forward_kernel("tiled") != 0) {
        fprintf(stderr, "FAIL: could not select both forward kernels\n");
        failures++;
        free_case(&kernel_case);
        return;
    }

    timing_spread_t spread = timing_measure_speedup(op_tiled, op_naive);

    printf("\nshipped kernel against the baseline it replaced:\n");
    if (!timing_expect_median_speedup("tiled vs naive at 128x256x1024", spread,
                                      1.0)) {
        fprintf(stderr, "FAIL: tiled vs naive has a median of %.2fx over %zu "
                        "replicates (spread %.2fx-%.2fx), below the 1.00x "
                        "minimum.\n       The tiled kernel stages operands "
                        "through shared memory and measures a median of "
                        "1.19x-1.29x here across five runs. A "
                        "median below 1.0 means the shared-memory path "
                        "regressed or stopped being selected.\n",
                spread.median, spread.replicates, spread.minimum, spread.maximum);
        failures++;
    }

    /* Leave the process on the shipped default. */
    gpu_matmul_set_forward_kernel("tiled");
    free_case(&kernel_case);
}

int main(void) {
    if (!gpu_matmul_available()) {
        printf("SKIP: no usable CUDA GPU detected (see gpu_probe.out for details) - not a failure.\n");
        return 0;
    }
    printf("forward kernel: %s, best of %d rounds per measurement, ratios only\n",
           gpu_matmul_forward_kernel_name(), ROUNDS);

    check_overhead_is_amortized();
    check_weight_cache_removes_a_transfer();
    check_tiled_kernel_is_not_a_regression();

    gpu_matmul_shutdown();

    if (failures != 0) {
        printf("\nGPU LATENCY INVARIANT CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nGPU LATENCY INVARIANT CHECK PASSED\n");
    return 0;
}
