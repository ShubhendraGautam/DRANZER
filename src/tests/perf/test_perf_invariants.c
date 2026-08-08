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
 *
 * THE SECOND RULE, learned the hard way: every ratio here is a MEDIAN over
 * replicates, and the spread is printed beside it. One measurement of a ratio is
 * not evidence about a machine that can deschedule a thread mid-call. The
 * AVX-512 backward-weight check below used to assert on a single best-of-5 ratio
 * against a 1.20x floor, and on an unmodified tree it read between 1.74x and
 * 3.69x on more than a dozen runs and 0.67x on one - so it failed a correct tree
 * at some rate and blocked merges for reasons unrelated to the change under
 * review. See include/tools/timing_spread.h for why the median and the printed
 * spread fix that, and why it matters beyond convenience.
 *
 * How unstable the old form really was is worth recording, because it is worse
 * than "occasionally flaky". The AVX-512 matmul_backward_input comparison read
 * 2.19x on one run of an untouched tree and 18.66x and 14.98x on the next two.
 * The true ratio on this machine is about 17x, confirmed by phase-separated
 * best-of-10 timings of each side. So a single reading was wrong by a factor of
 * eight on a quantity the test claimed to be measuring, and it PASSED, because
 * the floor was 1.20x and 2.19x clears it. A threshold set anywhere near the
 * real value would have flipped at random. The number was never evidence.
 *
 * The medians are stable. Six consecutive runs of this file on the same idle
 * machine gave:
 *
 *   matmul_backward_input / matrix_multiply     2.02 - 2.21   (limit 3.00)
 *   matmul_backward_weight / matrix_multiply    1.56 - 1.79   (limit 3.00)
 *   matrix_multiply vs matrix_multiply_scalar  35.39 - 37.97  (min   1.50)
 *   AVX-512 forward matmul                      3.76 - 3.82   (min   1.20)
 *   AVX-512 matmul_backward_input              16.51 - 17.63  (min   1.20)
 *   AVX-512 matmul_backward_weight              2.05 - 2.15   (min   1.20)
 *
 * every median within 10% of its neighbours, while the worst single replicate
 * inside those runs went as low as 0.96x for a comparison whose median never
 * left 1.56-1.79. Every threshold sits far outside the spread of the statistic
 * it gates, which is the property that was missing.
 *
 * The thresholds stay loose anyway, and deliberately: the spreads above are for
 * one CPU, and the ratios themselves legitimately differ between machines - a
 * different cache hierarchy changes how much the vector paths win. The printed
 * spread is what makes drift visible; the threshold only catches collapse.
 */

#include "core/cpu_features.h"
#include "core/matmul.h"
#include "tools/timing_spread.h"
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

static int failures;

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

static void op_forward(void)          { matrix_multiply(A, B, C, M, K, N); }
/* ISA-pinned variants for the dispatch comparison. Setting the cap inside the
 * timed function means each of the four interleaved calls in a replicate runs
 * under the ISA it is meant to measure - the cap is a cheap store, orders of
 * magnitude below the matmul it precedes. Timing the two arms in separate
 * phases instead, as this test used to, lets any drift between the phases land
 * entirely on one arm. */
static void op_forward_scalar(void)   { matrix_multiply_scalar(A, B, C, M, K, N); }
static void op_backward_input(void)   { matmul_backward_input(dC, B, dA, M, K, N); }
static void op_backward_weight(void)  { matmul_backward_weight(A, dC, dB, M, K, N); }

static void op_forward_simd(void) {
    cpu_features_clear_max_isa();
    matrix_multiply(A, B, C, M, K, N);
}
static void op_forward_baseline(void) {
    cpu_features_set_max_isa(CPU_ISA_BASELINE);
    matrix_multiply(A, B, C, M, K, N);
}
static void op_backward_input_simd(void) {
    cpu_features_clear_max_isa();
    matmul_backward_input(dC, B, dA, M, K, N);
}
static void op_backward_input_baseline(void) {
    cpu_features_set_max_isa(CPU_ISA_BASELINE);
    matmul_backward_input(dC, B, dA, M, K, N);
}
static void op_backward_weight_simd(void) {
    cpu_features_clear_max_isa();
    matmul_backward_weight(A, dC, dB, M, K, N);
}
static void op_backward_weight_baseline(void) {
    cpu_features_set_max_isa(CPU_ISA_BASELINE);
    matmul_backward_weight(A, dC, dB, M, K, N);
}

/* `numerator` must cost no more than `limit` times `denominator`. */
static void expect_cost_ratio(const char *what, void (*numerator)(void),
                              void (*denominator)(void), double limit,
                              const char *explanation) {
    /* timing_measure_speedup(fast, slow) returns slow/fast, so passing the
     * denominator as "fast" gives numerator/denominator. */
    timing_spread_t spread = timing_measure_speedup(denominator, numerator);
    if (!timing_expect_median_below(what, spread, limit)) {
        fprintf(stderr, "FAIL: %s has a median of %.2fx over %zu replicates "
                        "(spread %.2fx-%.2fx), above the %.2fx limit.\n       %s\n",
                what, spread.median, spread.replicates, spread.minimum,
                spread.maximum, limit, explanation);
        failures++;
    }
}

/* `fast` must beat `slow` by at least `minimum`, at the median. */
static void expect_speedup(const char *what, void (*fast)(void),
                           void (*slow)(void), double minimum,
                           const char *explanation) {
    timing_spread_t spread = timing_measure_speedup(fast, slow);
    if (!timing_expect_median_speedup(what, spread, minimum)) {
        fprintf(stderr, "FAIL: %s has a median of only %.2fx over %zu replicates "
                        "(spread %.2fx-%.2fx), below the %.2fx minimum.\n       %s\n",
                what, spread.median, spread.replicates, spread.minimum,
                spread.maximum, minimum, explanation);
        failures++;
    }
}

/* The backward matmuls do the same 2*m*k*n floating-point operations as a
 * forward matmul of the same shape. They traverse memory differently, so they
 * are allowed to be somewhat slower - but not by an order of magnitude. This
 * is the assertion that would have caught the loop-order pathology the day it
 * was written instead of months later. */
static void check_backward_within_reach_of_forward(void) {
    printf("\nbackward cost against a forward matmul of identical FLOP count:\n");
    expect_cost_ratio("matmul_backward_input / matrix_multiply",
                      op_backward_input, op_forward, 3.0,
                 "Both do 2*m*k*n FLOPs. A ratio this high means the traversal "
                 "is fighting the cache, not that the work is harder.");
    expect_cost_ratio("matmul_backward_weight / matrix_multiply",
                      op_backward_weight, op_forward, 3.0,
                 "Both do 2*m*k*n FLOPs. This is exactly the shape of the "
                 "l/j/i loop-order defect fixed in core/matmul.c. Rebuilding "
                 "this test against that old kernel reports 24.7x at this "
                 "shape, so the 3.0x limit has ample margin either way.");
}

/* The tuned forward kernel must beat the portable scalar reference. If this
 * ever fails, either the kernel selection stopped working or the tuning was
 * undone. */
static void check_forward_beats_scalar(void) {
    printf("\nselected forward kernel against the scalar reference:\n");
    expect_speedup("matrix_multiply vs matrix_multiply_scalar",
                   op_forward, op_forward_scalar, 1.5,
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

    /* Each arm has to be timed with the ISA cap in the right state, so the two
     * sides cannot simply be handed to timing_measure_speedup() as function
     * pointers - the cap has to move between them. These wrappers carry it, so
     * the interleaving still applies to the pair that is being compared. */
    printf("\nAVX-512 paths against the portable paths, same process:\n");
    expect_speedup("forward matmul", op_forward_simd, op_forward_baseline, 1.2,
                   "matmul_select() should resolve to avx512_mr4 here.");
    expect_speedup("matmul_backward_input", op_backward_input_simd,
                   op_backward_input_baseline, 1.2,
                   "core/matmul.c dispatches this to matmul_backward_input_avx512.");
    expect_speedup("matmul_backward_weight", op_backward_weight_simd,
                   op_backward_weight_baseline, 1.2,
                   "core/matmul.c dispatches this to matmul_backward_weight_avx512.");
    cpu_features_clear_max_isa();
}

int main(void) {
#ifdef DRANZER_UNDER_SANITIZER
    printf("SKIP: built with a sanitizer, so these timings would measure "
           "instrumentation rather than the kernels - not a failure.\n");
    return 0;
#else
    printf("cpu: %s\n", cpu_features_summary());
    printf("shape %dx%dx%d, median of %d ABBA-interleaved ratio replicates, "
           "ratios only\n", M, K, N, TIMING_SPREAD_REPLICATES);
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
