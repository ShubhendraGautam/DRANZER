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
static void check_weight_cache_removes_a_transfer(void) {
    case_t t = make_case(128, 256, 1024);

    double cached = best_call_sec(&t, 20);

    gpu_matmul(t.a, t.b, t.c, t.m, t.k, t.n);
    double best = -1.0;
    for (int r = 0; r < ROUNDS; r++) {
        double t0 = now_sec();
        for (int i = 0; i < 20; i++) {
            gpu_matmul_invalidate_weights();      /* force a re-upload of B */
            gpu_matmul(t.a, t.b, t.c, t.m, t.k, t.n);
        }
        double dt = (now_sec() - t0) / 20;
        if (best < 0.0 || dt < best) best = dt;
    }

    printf("\nweight cache removes a per-call upload:\n");
    printf("    cached %8.1f us     invalidated every call %8.1f us\n",
           cached * 1e6, best * 1e6);
    expect_speedup("cached vs re-uploading B every call", best / cached, 1.05,
                   "B is 1 MB at this shape. If invalidating the cache costs "
                   "nothing, the cache is not being consulted.");

    free_case(&t);
}

/* The shipped tiled kernel must not be slower than the naive one it replaced.
 * Measured at 1.19-1.24x on the largest shapes; asserted only as "not a
 * regression" because the margin narrows on smaller work and this must not
 * flake. */
static void check_tiled_kernel_is_not_a_regression(void) {
    case_t t = make_case(128, 256, 1024);

    if (gpu_matmul_set_forward_kernel("naive") != 0) {
        fprintf(stderr, "FAIL: could not select the naive kernel\n");
        failures++;
        free_case(&t);
        return;
    }
    double naive = best_call_sec(&t, 20);

    if (gpu_matmul_set_forward_kernel("tiled") != 0) {
        fprintf(stderr, "FAIL: could not select the tiled kernel\n");
        failures++;
        free_case(&t);
        return;
    }
    double tiled = best_call_sec(&t, 20);

    printf("\nshipped kernel against the baseline it replaced:\n");
    printf("    naive %8.1f us     tiled %8.1f us\n", naive * 1e6, tiled * 1e6);
    expect_speedup("tiled vs naive at 128x256x1024", naive / tiled, 1.0,
                   "The tiled kernel stages operands through shared memory and "
                   "measured 1.19-1.24x here. Slower means the shared-memory "
                   "path regressed or stopped being selected.");

    /* Leave the process on the shipped default. */
    gpu_matmul_set_forward_kernel("tiled");
    free_case(&t);
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
