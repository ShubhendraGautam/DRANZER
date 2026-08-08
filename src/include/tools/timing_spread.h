#ifndef TIMING_SPREAD_H
#define TIMING_SPREAD_H

/* clock_gettime() and CLOCK_MONOTONIC are POSIX, not C, so a strict -std=c11
 * translation unit does not see them without this. Requested before any header
 * is included, which is where the feature-test macro has to be. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Measuring a speedup as a distribution instead of as a number.
 *
 * Why this exists
 * ---------------
 * The timing tests used to take the fastest of five rounds for each side of a
 * comparison and divide once. That is a good estimator of each side's cost and a
 * bad way to gate a build, because a single collapsed round on the numerator
 * ruins the one ratio the test ever computes, and the test then reports a
 * pass/fail flip with no indication of how close it was.
 *
 * It happened. tests/perf/test_perf_invariants.c required the AVX-512
 * matmul_backward_weight to beat the portable path by 1.20x; across more than a
 * dozen runs of an unmodified tree, every reading but one fell between 1.74x and
 * 3.69x, and one collapsed to 0.67x. CI hit the same collapse at 0.74x on a
 * different machine. tests/gpu/test_gpu_latency_invariants.c had the same shape
 * of failure against a 1.00x floor: eight consecutive readings of 1.01x to 1.42x
 * and one of 0.83x.
 *
 * This is a research-integrity problem and not just an annoyance. A suite that
 * cries wolf teaches its maintainer to re-run until green, and that is the same
 * habit that turns a noisy experiment into a published effect.
 *
 * What this does instead
 * ----------------------
 * Take the ratio several times and keep the whole distribution.
 *
 *  - Each replicate times both sides ABBA - fast, slow, slow, fast - so a
 *    machine that is warming up or being descheduled biases both arms equally,
 *    and takes the faster of each side's two readings, since contention can only
 *    ever make a reading slower.
 *  - The assertion is on the MEDIAN of the replicates. One collapsed replicate
 *    cannot move it; a real regression moves all of them.
 *  - The minimum and maximum are reported next to it, always, so the spread is
 *    visible in the log of every run. A future collapse then shows up as a
 *    distribution that has changed shape - a max that fell, a spread that
 *    widened - rather than as a boolean that flipped.
 *
 * The threshold still has to be set outside the measured spread. The number
 * belongs beside the assertion, with the readings it came from.
 */

typedef struct {
    double minimum;
    double median;
    double maximum;
    size_t replicates;
} timing_spread_t;

/* How many ratio replicates. Nine keeps the median robust to one collapsed
 * reading (it would take five to move it) while costing four timed calls per
 * replicate, which at the shapes these tests use is a couple of seconds. */
#define TIMING_SPREAD_REPLICATES 9

static inline double timing_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline double timing_once(void (*op)(void)) {
    double start = timing_now_sec();
    op();
    return timing_now_sec() - start;
}

static inline int timing_compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* The distribution of `slow`'s cost divided by `fast`'s, over
 * TIMING_SPREAD_REPLICATES replicates. A returned median above 1.0 means `fast`
 * is indeed faster. */
static inline timing_spread_t timing_measure_speedup(void (*fast)(void),
                                                     void (*slow)(void)) {
    double ratios[TIMING_SPREAD_REPLICATES];

    /* Warm-up, discarded: first-touch page faults and cold caches belong to
     * neither arm. */
    fast();
    slow();

    for (size_t r = 0; r < TIMING_SPREAD_REPLICATES; r++) {
        double fast_first = timing_once(fast);
        double slow_first = timing_once(slow);
        double slow_second = timing_once(slow);
        double fast_second = timing_once(fast);

        double fast_best = fast_first < fast_second ? fast_first : fast_second;
        double slow_best = slow_first < slow_second ? slow_first : slow_second;
        ratios[r] = fast_best > 0.0 ? slow_best / fast_best : 0.0;
    }

    qsort(ratios, TIMING_SPREAD_REPLICATES, sizeof(ratios[0]),
          timing_compare_double);

    timing_spread_t spread;
    spread.minimum = ratios[0];
    spread.median = ratios[TIMING_SPREAD_REPLICATES / 2];
    spread.maximum = ratios[TIMING_SPREAD_REPLICATES - 1];
    spread.replicates = TIMING_SPREAD_REPLICATES;
    return spread;
}

/* Print one line carrying the whole distribution, not just the statistic being
 * asserted on. Returns nonzero when the median clears `minimum`. */
static inline int timing_expect_median_speedup(const char *what,
                                               timing_spread_t spread,
                                               double minimum) {
    int ok = spread.median >= minimum;
    printf("  %-44s median %5.2fx  [%5.2fx - %5.2fx, n=%zu]  (min %.2fx)  %s\n",
           what, spread.median, spread.minimum, spread.maximum,
           spread.replicates, minimum, ok ? "ok" : "FAIL");
    return ok;
}

/* The same, for a ratio that must stay BELOW a ceiling (a cost, not a speedup).
 * The median is compared, and the maximum is still reported. */
static inline int timing_expect_median_below(const char *what,
                                             timing_spread_t spread,
                                             double limit) {
    int ok = spread.median <= limit;
    printf("  %-44s median %5.2fx  [%5.2fx - %5.2fx, n=%zu]  (max %.2fx)  %s\n",
           what, spread.median, spread.minimum, spread.maximum,
           spread.replicates, limit, ok ? "ok" : "FAIL");
    return ok;
}

#endif /* TIMING_SPREAD_H */
