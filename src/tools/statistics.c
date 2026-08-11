/*
 * The shared statistics - see include/tools/statistics.h for why they live in
 * one module and why every comparison returns a verdict rather than only an
 * estimate.
 */

#include "tools/statistics.h"
#include "core/rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Sorting a copy, never the caller's array. */
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_of_sorted(const double *sorted, size_t count) {
    if (count == 0) return 0.0;
    if (count % 2 == 1) return sorted[count / 2];
    /* Even count: the mean of the two middle values. Averaging rather than
     * taking the lower keeps the median of {1,2} at 1.5, which is what a reader
     * comparing it against the mean expects. */
    return 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]);
}

stat_summary_t stat_summarize(const double *values, size_t count) {
    stat_summary_t summary;
    memset(&summary, 0, sizeof(summary));
    if (!values || count == 0) return summary;

    summary.count = count;

    double sum = 0.0;
    summary.minimum = values[0];
    summary.maximum = values[0];
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
        if (values[i] < summary.minimum) summary.minimum = values[i];
        if (values[i] > summary.maximum) summary.maximum = values[i];
    }
    summary.mean = sum / (double)count;

    if (count >= 2) {
        /* Two passes, not the sum-of-squares shortcut. The one-pass form
         * subtracts two large nearly equal numbers and loses most of its
         * precision on samples with a small spread relative to their mean -
         * which is exactly the shape of every timing measurement here. */
        double sum_sq = 0.0;
        for (size_t i = 0; i < count; i++) {
            double deviation = values[i] - summary.mean;
            sum_sq += deviation * deviation;
        }
        summary.stddev = sqrt(sum_sq / (double)(count - 1));
    }

    double *sorted = malloc(count * sizeof(double));
    if (sorted) {
        memcpy(sorted, values, count * sizeof(double));
        qsort(sorted, count, sizeof(double), compare_double);
        summary.median = median_of_sorted(sorted, count);
        free(sorted);
    } else {
        /* Out of memory for a copy of a sample this small means something is
         * badly wrong, but a zero median would be silently wrong. The mean is a
         * defensible substitute and the count still tells the caller what it
         * has. */
        summary.median = summary.mean;
    }

    return summary;
}

/* The percentile bootstrap, shared by every interval below.
 *
 * `statistic` is applied to each resample. The resample indices come from the
 * project generator so a reported interval is reproducible from its seed. */
static stat_interval_t bootstrap(const double *values, size_t count,
                                 size_t resamples, double level, uint64_t seed) {
    stat_interval_t interval;
    memset(&interval, 0, sizeof(interval));
    interval.resamples = resamples;
    interval.level = level;
    if (!values || count == 0) return interval;

    stat_summary_t observed = stat_summarize(values, count);
    interval.point = observed.mean;
    interval.low = observed.mean;
    interval.high = observed.mean;
    if (count < 2 || resamples == 0) return interval;

    double *means = malloc(resamples * sizeof(double));
    if (!means) return interval;

    uint64_t rng = dranzer_rng_stream(seed, DRANZER_RNG_STREAM_TESTING);
    for (size_t r = 0; r < resamples; r++) {
        double sum = 0.0;
        for (size_t i = 0; i < count; i++) {
            /* With replacement: that is what makes the spread of the resample
             * means an estimate of the sampling distribution. */
            size_t pick = (size_t)(dranzer_rng_unit(&rng) * (double)count);
            if (pick >= count) pick = count - 1;   /* only if unit rounds to 1.0 */
            sum += values[pick];
        }
        means[r] = sum / (double)count;
    }

    qsort(means, resamples, sizeof(double), compare_double);
    double tail = (1.0 - level) / 2.0;
    size_t low_index = (size_t)(tail * (double)resamples);
    size_t high_index = (size_t)((1.0 - tail) * (double)resamples);
    if (high_index >= resamples) high_index = resamples - 1;
    interval.low = means[low_index];
    interval.high = means[high_index];

    free(means);
    return interval;
}

stat_interval_t stat_bootstrap_mean(const double *values, size_t count,
                                    size_t resamples, double level,
                                    uint64_t seed) {
    return bootstrap(values, count, resamples, level, seed);
}

stat_paired_t stat_paired_compare(const double *arm_a, const double *arm_b,
                                  size_t pairs, double noise_floor,
                                  size_t resamples, uint64_t seed) {
    stat_paired_t result;
    memset(&result, 0, sizeof(result));
    result.noise_floor = fabs(noise_floor);
    result.pairs = pairs;
    result.verdict = STAT_INSUFFICIENT_DATA;
    if (!arm_a || !arm_b || pairs < 2) return result;

    double *differences = malloc(pairs * sizeof(double));
    double *ratios = malloc(pairs * sizeof(double));
    if (!differences || !ratios) {
        free(differences);
        free(ratios);
        return result;
    }

    size_t ratio_count = 0;
    for (size_t i = 0; i < pairs; i++) {
        differences[i] = arm_b[i] - arm_a[i];
        if (differences[i] > 0.0) result.pairs_favouring_b++;
        if (arm_a[i] != 0.0) {
            ratios[ratio_count++] = arm_b[i] / arm_a[i];
        }
    }
    result.ratio_pairs = ratio_count;

    /* Different seeds for the two bootstraps so the difference interval and the
     * ratio interval are not resampled along identical index sequences - which
     * would make them look more consistent with each other than they are. */
    result.difference = bootstrap(differences, pairs, resamples, 0.95, seed);
    if (ratio_count >= 2) {
        result.ratio = bootstrap(ratios, ratio_count, resamples, 0.95, seed + 1);
    }

    /* The verdict. Two conditions, both required:
     *
     *   1. The interval excludes zero. An interval containing zero is consistent
     *      with no difference at all, whatever the point estimate says.
     *   2. The whole interval lies outside the noise floor. An interval of
     *      [0.001, 0.004] excludes zero and is still meaningless when the
     *      seed-variance floor is 0.01 - the experiment is resolving something
     *      smaller than the thing it is measuring on.
     *
     * Anything else is STAT_UNRESOLVED, which is a result and gets reported as
     * one. */
    int excludes_zero = (result.difference.low > 0.0 && result.difference.high > 0.0) ||
                        (result.difference.low < 0.0 && result.difference.high < 0.0);
    double smallest_magnitude = fmin(fabs(result.difference.low),
                                     fabs(result.difference.high));
    int clears_floor = smallest_magnitude > result.noise_floor;

    if (!excludes_zero || !clears_floor) {
        result.verdict = STAT_UNRESOLVED;
    } else {
        result.verdict = result.difference.point > 0.0 ? STAT_RESOLVED_B_GREATER
                                                       : STAT_RESOLVED_B_SMALLER;
    }

    free(differences);
    free(ratios);
    return result;
}

const char *stat_verdict_name(stat_verdict_t verdict) {
    switch (verdict) {
        case STAT_RESOLVED_B_GREATER:  return "resolved: B greater";
        case STAT_RESOLVED_B_SMALLER:  return "resolved: B smaller";
        case STAT_UNRESOLVED:          return "unresolvable at this N";
        case STAT_INSUFFICIENT_DATA:   return "insufficient data";
        default:                       return "unknown";
    }
}

size_t stat_pairs_needed(const stat_paired_t *comparison) {
    if (!comparison || comparison->pairs < 2) return SIZE_MAX;
    if (comparison->verdict == STAT_RESOLVED_B_GREATER ||
        comparison->verdict == STAT_RESOLVED_B_SMALLER) {
        return 0;
    }

    double effect = fabs(comparison->difference.point);
    if (effect <= comparison->noise_floor) return SIZE_MAX;

    /* The bootstrap interval half-width scales as 1/sqrt(n), so the n at which
     * it would shrink to the effect size is the current n times the square of
     * the ratio. Rough by construction - it assumes the spread does not change
     * with n, which is why the header says this is a guide and not a power
     * calculation. */
    double half_width = 0.5 * (comparison->difference.high - comparison->difference.low);
    if (half_width <= 0.0) return SIZE_MAX;
    double margin = effect - comparison->noise_floor;
    double factor = half_width / margin;
    double needed = (double)comparison->pairs * factor * factor;
    if (!(needed > 0.0) || needed > 1e9) return SIZE_MAX;
    return (size_t)ceil(needed);
}

stat_seed_floor_t stat_seed_floor(const double *losses, size_t count,
                                  size_t minimum_samples,
                                  size_t maximum_samples,
                                  double target_precision_ratio,
                                  size_t resamples, uint64_t seed) {
    stat_seed_floor_t floor;
    memset(&floor, 0, sizeof(floor));
    floor.status = STAT_SEED_FLOOR_INVALID;
    floor.minimum_samples = minimum_samples;
    floor.maximum_samples = maximum_samples;
    floor.target_precision_ratio = target_precision_ratio;

    if (!losses || count == 0 || minimum_samples < 2 ||
        maximum_samples < minimum_samples || count > maximum_samples ||
        !(target_precision_ratio > 0.0) || resamples == 0) {
        return floor;
    }

    floor.losses = stat_summarize(losses, count);
    floor.mean_interval = stat_bootstrap_mean(losses, count, resamples, 0.95, seed);
    floor.noise_floor = floor.losses.stddev;

    if (count < minimum_samples) {
        floor.status = STAT_SEED_FLOOR_COLLECT_MORE;
        floor.recommended_total = minimum_samples;
        return floor;
    }

    if (floor.noise_floor == 0.0) {
        /* Identical losses across the minimum sample are a real measured zero,
         * not a divide-by-zero failure. Additional seeds may falsify it later,
         * but the same is true of any finite sample. */
        floor.precision_ratio = 0.0;
        floor.recommended_total = count;
        floor.status = STAT_SEED_FLOOR_READY;
        return floor;
    }

    double half_width = 0.5 * (floor.mean_interval.high - floor.mean_interval.low);
    floor.precision_ratio = half_width / floor.noise_floor;
    if (floor.precision_ratio <= target_precision_ratio) {
        floor.recommended_total = count;
        floor.status = STAT_SEED_FLOOR_READY;
        return floor;
    }

    if (count >= maximum_samples) {
        floor.recommended_total = maximum_samples;
        floor.status = STAT_SEED_FLOOR_LIMIT_REACHED;
        return floor;
    }

    double scale = floor.precision_ratio / target_precision_ratio;
    double estimated = ceil((double)count * scale * scale);
    size_t recommended = estimated >= (double)SIZE_MAX ? maximum_samples
                                                        : (size_t)estimated;
    if (recommended <= count) recommended = count + 1;
    if (recommended > maximum_samples) recommended = maximum_samples;
    floor.recommended_total = recommended;
    floor.status = STAT_SEED_FLOOR_COLLECT_MORE;
    return floor;
}

const char *stat_seed_floor_status_name(stat_seed_floor_status_t status) {
    switch (status) {
        case STAT_SEED_FLOOR_READY:         return "ready";
        case STAT_SEED_FLOOR_COLLECT_MORE:  return "collect_more";
        case STAT_SEED_FLOOR_LIMIT_REACHED: return "limit_reached";
        case STAT_SEED_FLOOR_INVALID:       return "invalid";
        default:                            return "unknown";
    }
}
