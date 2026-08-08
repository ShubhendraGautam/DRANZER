/* The shared statistics, pinned against values computed by hand.
 *
 * Every number in this file was worked out independently of the implementation -
 * the arithmetic is written into the comments so a reader can check it without
 * running anything. That matters more here than in most tests: a statistics
 * module that is subtly wrong does not crash, it produces confident intervals
 * around the wrong answer, and every result the project reports inherits the
 * error.
 *
 * The verdict logic gets the most attention, because it is the part that is easy
 * to write in a way that is technically defensible and practically useless. An
 * interval that excludes zero but sits inside the noise floor must come back
 * UNRESOLVED, and the case is constructed here explicitly.
 */
#include "tools/statistics.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;

static void expect_close(const char *what, double got, double want, double tolerance) {
    double gap = fabs(got - want);
    int ok = gap <= tolerance;
    printf("  %-46s %12.6f  (expected %.6f)  %s\n", what, got, want,
           ok ? "ok" : "FAIL");
    if (!ok) {
        fprintf(stderr, "FAIL: %s is %.9f, expected %.9f (gap %.2e > %.2e)\n",
                what, got, want, gap, tolerance);
        failures++;
    }
}

static void expect_verdict(const char *what, stat_verdict_t got,
                           stat_verdict_t want) {
    int ok = got == want;
    printf("  %-46s %-24s  %s\n", what, stat_verdict_name(got),
           ok ? "ok" : "FAIL");
    if (!ok) {
        fprintf(stderr, "FAIL: %s gave \"%s\", expected \"%s\"\n", what,
                stat_verdict_name(got), stat_verdict_name(want));
        failures++;
    }
}

int main(void) {
    /* ---- 1. Summary statistics ----
     *
     * values = {2, 4, 4, 4, 5, 5, 7, 9}, n = 8
     *   sum      = 40, so mean = 5
     *   sorted   = same; the two middle values are 4 and 5, so median = 4.5
     *   deviations from the mean: -3, -1, -1, -1, 0, 0, 2, 4
     *   squares:                   9,  1,  1,  1, 0, 0, 4, 16  -> sum 32
     *   sample variance = 32 / (8 - 1) = 4.571428571...
     *   sample stddev   = 2.13808993...
     *
     * The n-1 denominator is the thing worth pinning: with n it would give
     * exactly 2.0, which is the population standard deviation and the wrong
     * quantity for a sample drawn from a process. A test that accepted 2.0 would
     * accept a module that understates every interval it reports. */
    const double values[] = {2, 4, 4, 4, 5, 5, 7, 9};
    stat_summary_t summary = stat_summarize(values, 8);
    printf("summary of {2,4,4,4,5,5,7,9}:\n");
    expect_close("count", (double)summary.count, 8.0, 0.0);
    expect_close("mean", summary.mean, 5.0, 1e-12);
    expect_close("median (even n: mean of the middle two)", summary.median, 4.5, 1e-12);
    expect_close("minimum", summary.minimum, 2.0, 0.0);
    expect_close("maximum", summary.maximum, 9.0, 0.0);
    expect_close("stddev (n-1 denominator, not n)", summary.stddev,
                 2.1380899352993947, 1e-12);
    if (fabs(summary.stddev - 2.0) < 1e-9) {
        fprintf(stderr, "FAIL: stddev used the population denominator\n");
        failures++;
    }

    /* Odd n takes the middle element outright: {1,2,3,4,5} -> 3. */
    const double odd[] = {5, 1, 4, 2, 3};
    stat_summary_t odd_summary = stat_summarize(odd, 5);
    expect_close("median (odd n)", odd_summary.median, 3.0, 1e-12);
    /* And the input must be untouched - a function that sorts its caller's array
     * would silently break the pairing in stat_paired_compare(). */
    if (odd[0] != 5.0 || odd[1] != 1.0 || odd[4] != 3.0) {
        fprintf(stderr, "FAIL: stat_summarize reordered its caller's array\n");
        failures++;
    }
    printf("  %-46s %s\n", "input array left unmodified", "ok");

    /* Degenerate inputs return a defined answer rather than reading off the end.
     * n = 1 has no sample standard deviation; zero is the documented stand-in. */
    const double single[] = {42.0};
    stat_summary_t one = stat_summarize(single, 1);
    expect_close("n=1 mean", one.mean, 42.0, 0.0);
    expect_close("n=1 stddev (undefined, reported as 0)", one.stddev, 0.0, 0.0);
    stat_summary_t none = stat_summarize(values, 0);
    expect_close("n=0 count", (double)none.count, 0.0, 0.0);

    /* ---- 2. Bootstrap interval ----
     *
     * A constant sample has no sampling variation: every resample of
     * {3, 3, 3, 3, 3} is {3, 3, 3, 3, 3}, so every resample mean is 3 and the
     * interval must be exactly [3, 3]. This is the one bootstrap case with an
     * analytically certain answer, which makes it the one worth pinning
     * exactly - it catches an off-by-one in the percentile indexing that a
     * wider sample would hide. */
    const double constant[] = {3, 3, 3, 3, 3};
    stat_interval_t flat = stat_bootstrap_mean(constant, 5, 500, 0.95, 7);
    printf("\nbootstrap of a constant sample {3,3,3,3,3}:\n");
    expect_close("point", flat.point, 3.0, 1e-12);
    expect_close("interval low", flat.low, 3.0, 1e-12);
    expect_close("interval high", flat.high, 3.0, 1e-12);

    /* On a real sample the interval must contain the point estimate and be
     * ordered. Also: the same seed must give the same interval twice, because a
     * reported interval that moves when the analysis is re-run is not a
     * reportable number. */
    const double spread[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    stat_interval_t a = stat_bootstrap_mean(spread, 8, 2000, 0.95, 99);
    stat_interval_t b = stat_bootstrap_mean(spread, 8, 2000, 0.95, 99);
    printf("\nbootstrap of {1..8}, 2000 resamples, seed 99:\n");
    printf("  point %.4f, interval [%.4f, %.4f]\n", a.point, a.low, a.high);
    expect_close("point is the sample mean", a.point, 4.5, 1e-12);
    if (!(a.low <= a.point && a.point <= a.high)) {
        fprintf(stderr, "FAIL: interval does not contain its point estimate\n");
        failures++;
    } else {
        printf("  %-46s %s\n", "interval contains the point estimate", "ok");
    }
    if (a.low != b.low || a.high != b.high) {
        fprintf(stderr, "FAIL: the same seed gave two different intervals\n");
        failures++;
    } else {
        printf("  %-46s %s\n", "same seed reproduces the interval exactly", "ok");
    }
    /* The interval has to be narrower than the sample's own range, or the
     * bootstrap is not estimating the mean's sampling distribution at all. */
    if (!(a.high - a.low < 7.0)) {
        fprintf(stderr, "FAIL: the interval is as wide as the whole sample\n");
        failures++;
    }

    /* ---- 3. Paired comparison, resolved ----
     *
     * B is A plus 1.0 in every pair, exactly. The paired differences are all
     * 1.0, so every resample mean is 1.0 and the interval is [1, 1]: it excludes
     * zero and clears a small floor, so the verdict is resolved. All five pairs
     * favour B. */
    const double arm_a[] = {10.0, 12.0, 11.0, 13.0, 9.0};
    const double arm_b[] = {11.0, 13.0, 12.0, 14.0, 10.0};
    stat_paired_t resolved = stat_paired_compare(arm_a, arm_b, 5, 0.1, 2000, 5);
    printf("\npaired: B = A + 1 exactly, noise floor 0.1:\n");
    expect_close("difference point", resolved.difference.point, 1.0, 1e-12);
    expect_close("difference low", resolved.difference.low, 1.0, 1e-12);
    expect_close("pairs favouring B", (double)resolved.pairs_favouring_b, 5.0, 0.0);
    expect_verdict("verdict", resolved.verdict, STAT_RESOLVED_B_GREATER);
    expect_close("pairs needed (already resolved)",
                 (double)stat_pairs_needed(&resolved), 0.0, 0.0);

    /* The ratio is reported too, on the same pairs: 11/10, 13/12, 12/11, 14/13,
     * 10/9 = 1.1, 1.08333..., 1.090909..., 1.076923..., 1.111111...
     * mean = (1.1 + 1.0833333 + 1.0909091 + 1.0769231 + 1.1111111) / 5
     *      = 5.4622766 / 5 = 1.09245532... */
    expect_close("ratio point (mean of per-pair ratios)", resolved.ratio.point,
                 1.0924553, 1e-6);

    /* ---- 4. Paired comparison, unresolved because it straddles zero ---- */
    const double noisy_a[] = {10.0, 12.0, 11.0, 13.0, 9.0};
    const double noisy_b[] = {11.0, 11.0, 12.0, 12.0, 9.5};
    stat_paired_t straddles = stat_paired_compare(noisy_a, noisy_b, 5, 0.0, 2000, 5);
    printf("\npaired: differences +1, -1, +1, -1, +0.5, no floor:\n");
    printf("  point %.4f, interval [%.4f, %.4f]\n", straddles.difference.point,
           straddles.difference.low, straddles.difference.high);
    expect_close("difference point (mean of +1,-1,+1,-1,+0.5)",
                 straddles.difference.point, 0.1, 1e-12);
    expect_verdict("verdict", straddles.verdict, STAT_UNRESOLVED);
    /* And it should say roughly how many pairs would settle it, rather than
     * leaving the reader to guess. */
    size_t needed = stat_pairs_needed(&straddles);
    printf("  pairs needed to resolve an effect this size: %s\n",
           needed == SIZE_MAX ? "no N would help" : "estimated");
    if (needed == 0) {
        fprintf(stderr, "FAIL: an unresolved comparison needs more than 0 pairs\n");
        failures++;
    }

    /* ---- 5. Paired comparison, unresolved because of the NOISE FLOOR ----
     *
     * This is the case the module exists for. B is A plus 0.002 in every pair,
     * exactly, so the interval is [0.002, 0.002] and excludes zero with total
     * confidence. But the caller's floor is 0.01: the effect is five times
     * smaller than the noise of the thing being measured on. A module that
     * reported this as resolved would be arithmetically right and would license
     * a claim that means nothing - which is precisely the failure the 12-seed
     * quantization withdrawal was. */
    const double tiny_a[] = {2.500, 2.510, 2.490, 2.505, 2.495, 2.502};
    double tiny_b[6];
    for (size_t i = 0; i < 6; i++) tiny_b[i] = tiny_a[i] + 0.002;
    stat_paired_t below_floor = stat_paired_compare(tiny_a, tiny_b, 6, 0.01, 2000, 5);
    printf("\npaired: B = A + 0.002 exactly, but the noise floor is 0.01:\n");
    expect_close("difference point", below_floor.difference.point, 0.002, 1e-12);
    expect_close("difference low (excludes zero with certainty)",
                 below_floor.difference.low, 0.002, 1e-12);
    expect_close("pairs favouring B", (double)below_floor.pairs_favouring_b, 6.0, 0.0);
    expect_verdict("verdict must be unresolved anyway", below_floor.verdict,
                   STAT_UNRESOLVED);
    if (stat_pairs_needed(&below_floor) != SIZE_MAX) {
        fprintf(stderr, "FAIL: an effect below the noise floor should report "
                        "that no N helps\n");
        failures++;
    } else {
        printf("  %-46s %s\n", "and that no sample size would help", "ok");
    }

    /* ---- 6. Degenerate paired input ---- */
    stat_paired_t one_pair = stat_paired_compare(arm_a, arm_b, 1, 0.0, 100, 5);
    expect_verdict("one pair is insufficient", one_pair.verdict,
                   STAT_INSUFFICIENT_DATA);
    stat_paired_t no_pairs = stat_paired_compare(arm_a, arm_b, 0, 0.0, 100, 5);
    expect_verdict("zero pairs is insufficient", no_pairs.verdict,
                   STAT_INSUFFICIENT_DATA);

    /* A zero in arm A cannot produce a ratio; it must be skipped and counted,
     * not turned into an infinity that the interval then propagates. */
    const double with_zero_a[] = {0.0, 2.0, 4.0, 5.0};
    const double with_zero_b[] = {1.0, 4.0, 8.0, 10.0};
    stat_paired_t skipped = stat_paired_compare(with_zero_a, with_zero_b, 4, 0.0,
                                                500, 5);
    expect_close("ratio pairs (one denominator was zero)",
                 (double)skipped.ratio_pairs, 3.0, 0.0);
    expect_close("ratio point over the usable pairs", skipped.ratio.point, 2.0, 1e-12);

    printf("\n%s\n", failures == 0 ? "STATISTICS CHECK PASSED"
                                   : "STATISTICS CHECK FAILED");
    return failures == 0 ? 0 : 1;
}
