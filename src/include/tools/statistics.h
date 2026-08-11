#ifndef STATISTICS_H
#define STATISTICS_H

#include <stddef.h>
#include <stdint.h>

/*
 * The error bars, in one place.
 *
 * Why this exists
 * ---------------
 * This project's measurement discipline is unusually careful - ABBA
 * interleaving, best-of-N with the median as a noise indicator, a claim
 * withdrawn when a larger sample failed to sharpen it - and all of it lived in
 * prose and in individual tools. `tools/bench_quant.c` computed paired
 * differences and spreads by hand. `tools/bench_matmul.c` had its own notion of
 * a noise floor. Each new experiment re-derived the same arithmetic, and the one
 * that forgets a step is the one that gets published.
 *
 * So: one module, tested against hand-computed values
 * (tests/core/test_statistics.c), used by every experiment tool. An experiment
 * that computes its own error bars is a bug in the same sense that an experiment
 * that computes its own matmul would be.
 *
 * The verdict is part of the output
 * ---------------------------------
 * Every comparison here returns whether it resolved, not just an estimate. A
 * difference smaller than the noise of the thing being compared is not a small
 * effect - it is an absence of evidence, and the only honest report of it says
 * so. stat_paired_verdict() gives that answer in words, and "unresolvable at
 * this N" is a first-class outcome rather than a caveat someone might add to
 * the prose later.
 *
 * That is not a stylistic preference. The 12-seed to 60-seed quantization
 * withdrawal recorded in docs/design-checklist.md is exactly the failure this
 * prevents: an effect that looked real at one sample size and did not survive a
 * larger one. Making the verdict structural means the next such comparison
 * reports "unresolvable" by itself instead of depending on someone remembering.
 *
 * Bootstrap rather than a t-test
 * ------------------------------
 * The quantities this project compares - throughput ratios, held-out
 * cross-entropy across seeds, weight-space error - are not normally distributed
 * and there is no reason to pretend otherwise. A percentile bootstrap makes no
 * distributional assumption, works on a ratio as readily as on a difference, and
 * is trivially explainable: resample the pairs with replacement, recompute the
 * statistic, and report the middle 95% of what comes out.
 *
 * It is also deterministic here. The resampling draws from core/rng.h with a
 * caller-supplied seed, so a reported interval is reproducible from the recorded
 * seed rather than being a number that moves every time the analysis is re-run.
 */

/* Order statistics and moments of one sample. */
typedef struct {
    size_t count;
    double mean;
    double median;
    double minimum;
    double maximum;
    /* Sample standard deviation, with the n-1 denominator: these are samples
     * from a process, not a population that has been enumerated. Zero when
     * count < 2, where the quantity is undefined rather than zero - callers that
     * care should check count. */
    double stddev;
} stat_summary_t;

/* Summarize `count` values. Does not modify the input: the median needs sorted
 * data and copies internally, because a function that quietly reorders its
 * caller's array is how a paired design stops being paired. */
stat_summary_t stat_summarize(const double *values, size_t count);

/* A point estimate with an interval around it. */
typedef struct {
    double point;
    double low;
    double high;
    size_t resamples;
    double level;      /* e.g. 0.95 */
} stat_interval_t;

/* Percentile bootstrap interval for the mean of one sample.
 *
 * `resamples` of 2000 is enough for a 95% interval to be stable to about two
 * significant figures, which is more precision than any claim in this project
 * should be quoted to anyway. `seed` fixes the resampling. */
stat_interval_t stat_bootstrap_mean(const double *values, size_t count,
                                    size_t resamples, double level,
                                    uint64_t seed);

/* The outcome of a paired comparison. */
typedef enum {
    /* The interval excludes zero (and clears the noise floor): arm B differs
     * from arm A by more than the measurement can explain. */
    STAT_RESOLVED_B_GREATER = 0,
    STAT_RESOLVED_B_SMALLER,
    /* The interval contains zero, or lies inside the noise floor. There is no
     * evidence of a difference at this sample size. This is not "no
     * difference" - it is "this experiment cannot tell", and the two must not
     * be reported as the same thing. */
    STAT_UNRESOLVED,
    /* Fewer than two pairs: nothing to say. */
    STAT_INSUFFICIENT_DATA
} stat_verdict_t;

typedef struct {
    size_t pairs;
    /* B - A, per pair, bootstrapped. The paired difference is the statistic
     * rather than the difference of the two means, because the design is
     * paired: the same seed, the same shape, the same machine on both arms. A
     * per-arm interval would carry the between-seed variance that pairing was
     * introduced to remove, and would report "unresolved" for effects the
     * experiment can actually see. */
    stat_interval_t difference;
    /* B / A, per pair, bootstrapped. Reported alongside the difference because
     * a speedup is naturally multiplicative and a loss delta is naturally
     * additive, and forcing either into the other's form loses meaning. Zero
     * denominators are skipped and reported in `ratio_pairs`. */
    stat_interval_t ratio;
    size_t ratio_pairs;
    /* The smallest difference the caller considers meaningful - a seed-variance
     * floor, a measured timing noise band. An interval that clears zero but sits
     * inside this is still unresolved. */
    double noise_floor;
    stat_verdict_t verdict;
    /* How many pairs favoured B, as a plain count. Kept because a sign count is
     * the one summary a reader can check by eye against the raw table, and
     * because 5/5 in one direction says something an interval straddling zero
     * does not. */
    size_t pairs_favouring_b;
} stat_paired_t;

/* Compare two arms measured on the same units in the same order.
 *
 * arm_a[i] and arm_b[i] must be the same seed, shape, or configuration measured
 * both ways - that is what makes this paired. Passing unpaired samples produces
 * a number, and it is the wrong number. */
stat_paired_t stat_paired_compare(const double *arm_a, const double *arm_b,
                                  size_t pairs, double noise_floor,
                                  size_t resamples, uint64_t seed);

/* One line, suitable for printing directly into a results table. Never elides
 * the verdict: an unresolved comparison says so in words. */
const char *stat_verdict_name(stat_verdict_t verdict);

/* How many pairs would be needed to resolve an effect of the observed size,
 * estimated from the observed per-pair spread. Returns 0 when the effect is
 * already resolved, and SIZE_MAX when the observed effect is zero or smaller
 * than the noise floor, where no sample size helps.
 *
 * This is a rough guide for deciding whether to run more seeds, not a power
 * calculation to be quoted. It exists so that "unresolvable at this N" comes
 * with an idea of what N would be enough - which is the next question a reader
 * asks, and answering it with a number beats answering it with a shrug. */
size_t stat_pairs_needed(const stat_paired_t *comparison);

/* Adaptive estimate of the between-seed quality floor.
 *
 * The floor is the sample standard deviation of one fixed configuration's
 * held-out cross-entropy across seeds. It is deliberately not the observed
 * min/max range (which grows as seeds are added) or the confidence interval of
 * the mean (which answers how precisely the mean is known, not how much seeds
 * move an individual run). Architecture deltas narrower than this value must
 * be passed to stat_paired_compare() as its noise_floor.
 *
 * N is selected from the observed sample rather than fixed in advance. After
 * minimum_samples, collection is ready when the 95% bootstrap interval's
 * half-width for the mean is at most target_precision_ratio times the observed
 * standard deviation. Otherwise recommended_total estimates the N needed under
 * the ordinary 1/sqrt(N) interval-width scaling, capped at maximum_samples.
 */
typedef enum {
    STAT_SEED_FLOOR_READY = 0,
    STAT_SEED_FLOOR_COLLECT_MORE,
    STAT_SEED_FLOOR_LIMIT_REACHED,
    STAT_SEED_FLOOR_INVALID
} stat_seed_floor_status_t;

typedef struct {
    stat_summary_t losses;
    stat_interval_t mean_interval;
    double noise_floor;
    double precision_ratio;
    double target_precision_ratio;
    size_t minimum_samples;
    size_t maximum_samples;
    size_t recommended_total;
    stat_seed_floor_status_t status;
} stat_seed_floor_t;

stat_seed_floor_t stat_seed_floor(const double *losses, size_t count,
                                  size_t minimum_samples,
                                  size_t maximum_samples,
                                  double target_precision_ratio,
                                  size_t resamples, uint64_t seed);

const char *stat_seed_floor_status_name(stat_seed_floor_status_t status);

#endif /* STATISTICS_H */
