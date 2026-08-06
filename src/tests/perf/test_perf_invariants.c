/*
 * Performance invariants, as tests rather than as a benchmark someone
 * remembers to read.
 *
 * Everything measured in this project so far was caught by running a
 * benchmark by hand and noticing a number looked wrong. That found the
 * pathology in matmul_backward_weight() - it cost about 6.3x a forward matmul
 * of identical FLOP count because it strided both operands in its innermost
 * loop - but only after it had been shipped and only because someone went
 * looking. This file turns those observations into assertions that fail on
 * their own.
 *
 * THE RULE: every assertion here is a RATIO between two measurements taken
 * back to back in this same process, never an absolute time. A slow, busy, or
 * thermally throttled machine scales both sides equally and the ratio holds.
 * That is the same discipline tools/perf_check.py applies to CI benchmark
 * rows, and it is what makes a timing test safe to run in an ordinary suite.
 *
 * Thresholds are deliberately loose. The defects worth catching here are
 * order-of-magnitude - a quadratic memory access pattern, a dispatch that
 * silently stopped firing, a kernel that regressed to the reference - not a
 * 10% drift. A gate that tries to police 10% on a shared runner produces
 * false failures, which is worse than no gate at all.
 */

#include "core/cpu_features.h"
#include "core/matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Timing under a sanitizer measures the sanitizer.
 *
 * Every load and store is instrumented, and not evenly: portable scalar loops
 * take far more instrumentation per useful operation than vector intrinsics
 * do. Run under AddressSanitizer, the ratios below inverted and inflated - the
 * backward matmuls measured 0.69x a forward matmul of equal FLOP count
 * (faster, which they are not), and the AVX-512 speedups read 8-16x instead of
 * about 2-3x. Those assertions pass with enormous margin, which is worse than
 * failing: they look like coverage while testing nothing about performance.
 *
 * The correctness suite still runs under ASAN, which is where the sanitizer
 * earns its keep. */
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define DRANZER_UNDER_SANITIZER 1
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#  define DRANZER_UNDER_SANITIZER 1
#endif

/* Big enough that a single call clears timer noise, small enough that the
 * whole file runs in a couple of seconds. 2*64*256*256 = 8.4 MFLOP per call. */
#define M 64
#define K 256
#define N 256

#define ROUNDS 5

static int failures;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float *A, *B, *C, *dC, *dA, *dB;

static void alloc_all(void) {
    A = malloc((size_t)M * K * sizeof(float));
    B = malloc((size_t)K * N * sizeof(float));
    C = malloc((size_t)M * N * sizeof(float));
    dC = malloc((size_t)M * N * sizeof(float));
    dA = malloc((size_t)M * K * sizeof(float));
    dB = malloc((size_t)K * N * sizeof(float));
    if (!A || !B || !C || !dC || !dA || !dB) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (float)(i % 13) * 0.01f;
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (float)(i % 11) * 0.01f;
    for (size_t i = 0; i < (size_t)M * N; i++) dC[i] = (float)(i % 7) * 0.01f;
    memset(dA, 0, (size_t)M * K * sizeof(float));
    memset(dB, 0, (size_t)K * N * sizeof(float));
}

/* Fastest round wins: contention can only ever make a round slower, so the
 * minimum is the closest estimate of the true cost. Same rule as the matmul
 * sweep (docs/matmul.md). One discarded warm-up round per operation. */
static double best_of(void (*op)(void)) {
    op();
    double best = -1.0;
    for (int r = 0; r < ROUNDS; r++) {
        double t0 = now_sec();
        op();
        double dt = now_sec() - t0;
        if (best < 0.0 || dt < best) best = dt;
    }
    return best;
}

static void op_forward(void)          { matrix_multiply(A, B, C, M, K, N); }
static void op_forward_scalar(void)   { matrix_multiply_scalar(A, B, C, M, K, N); }
static void op_backward_input(void)   { matmul_backward_input(dC, B, dA, M, K, N); }
static void op_backward_weight(void)  { matmul_backward_weight(A, dC, dB, M, K, N); }

static void expect_ratio(const char *what, double ratio, double limit,
                         const char *explanation) {
    int ok = ratio <= limit;
    printf("  %-52s %6.2fx  (limit %.2fx)  %s\n",
           what, ratio, limit, ok ? "ok" : "FAIL");
    if (!ok) {
        fprintf(stderr, "FAIL: %s is %.2fx, above the %.2fx limit.\n       %s\n",
                what, ratio, limit, explanation);
        failures++;
    }
}

static void expect_speedup(const char *what, double speedup, double minimum,
                           const char *explanation) {
    int ok = speedup >= minimum;
    printf("  %-52s %6.2fx  (min   %.2fx)  %s\n",
           what, speedup, minimum, ok ? "ok" : "FAIL");
    if (!ok) {
        fprintf(stderr, "FAIL: %s is only %.2fx, below the %.2fx minimum.\n       %s\n",
                what, speedup, minimum, explanation);
        failures++;
    }
}

/* The backward matmuls do the same 2*m*k*n floating-point operations as a
 * forward matmul of the same shape. They traverse memory differently, so they
 * are allowed to be somewhat slower - but not by an order of magnitude. This
 * is the assertion that would have caught the loop-order pathology the day it
 * was written instead of months later. */
static void check_backward_within_reach_of_forward(void) {
    double forward = best_of(op_forward);
    double bwd_in = best_of(op_backward_input);
    double bwd_wt = best_of(op_backward_weight);

    printf("\nbackward cost against a forward matmul of identical FLOP count:\n");
    expect_ratio("matmul_backward_input / matrix_multiply", bwd_in / forward, 3.0,
                 "Both do 2*m*k*n FLOPs. A ratio this high means the traversal "
                 "is fighting the cache, not that the work is harder.");
    expect_ratio("matmul_backward_weight / matrix_multiply", bwd_wt / forward, 3.0,
                 "Both do 2*m*k*n FLOPs. This is exactly the shape of the "
                 "l/j/i loop-order defect fixed in core/matmul.c. Rebuilding "
                 "this test against that old kernel reports 24.7x at this "
                 "shape, so the 3.0x limit has ample margin either way.");
}

/* The tuned forward kernel must beat the portable scalar reference. If this
 * ever fails, either the kernel selection stopped working or the tuning was
 * undone. */
static void check_forward_beats_scalar(void) {
    double tuned = best_of(op_forward);
    double scalar = best_of(op_forward_scalar);
    printf("\nselected forward kernel against the scalar reference:\n");
    expect_speedup("matrix_multiply vs matrix_multiply_scalar", scalar / tuned, 1.5,
                   "matmul_select() resolves to a tuned kernel; if it is no "
                   "faster than the unblocked reference, dispatch is broken.");
}

/* Where the CPU has AVX-512, the vector paths must actually be faster than
 * the portable ones. A dispatch that silently stops firing - a bad
 * cpu_isa_available(), a missing DRANZER_HAVE_X86_SIMD - shows up here as a
 * ratio near 1.0 rather than as a wrong answer, which no correctness test
 * would catch. */
static void check_simd_dispatch_is_worth_something(void) {
    if (!cpu_isa_available(CPU_ISA_AVX512)) {
        printf("\nSIMD dispatch: no AVX-512 on this CPU, nothing to compare - skipped.\n");
        return;
    }

    cpu_features_clear_max_isa();
    double simd_fwd = best_of(op_forward);
    double simd_in = best_of(op_backward_input);
    double simd_wt = best_of(op_backward_weight);

    cpu_features_set_max_isa(CPU_ISA_BASELINE);
    double base_fwd = best_of(op_forward);
    double base_in = best_of(op_backward_input);
    double base_wt = best_of(op_backward_weight);
    cpu_features_clear_max_isa();

    printf("\nAVX-512 paths against the portable paths, same process:\n");
    expect_speedup("forward matmul", base_fwd / simd_fwd, 1.2,
                   "matmul_select() should resolve to avx512_mr4 here.");
    expect_speedup("matmul_backward_input", base_in / simd_in, 1.2,
                   "core/matmul.c dispatches this to matmul_backward_input_avx512.");
    expect_speedup("matmul_backward_weight", base_wt / simd_wt, 1.2,
                   "core/matmul.c dispatches this to matmul_backward_weight_avx512.");
}

int main(void) {
#ifdef DRANZER_UNDER_SANITIZER
    printf("SKIP: built with a sanitizer, so these timings would measure "
           "instrumentation rather than the kernels - not a failure.\n");
    return 0;
#else
    printf("cpu: %s\n", cpu_features_summary());
    printf("shape %dx%dx%d, best of %d rounds per operation, ratios only\n",
           M, K, N, ROUNDS);
    alloc_all();

    check_backward_within_reach_of_forward();
    check_forward_beats_scalar();
    check_simd_dispatch_is_worth_something();

    free(A); free(B); free(C); free(dC); free(dA); free(dB);

    if (failures != 0) {
        printf("\nPERFORMANCE INVARIANT CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nPERFORMANCE INVARIANT CHECK PASSED\n");
    return 0;
#endif
}
