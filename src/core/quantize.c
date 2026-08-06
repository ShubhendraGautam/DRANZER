/*
 * Symmetric weight-only quantization. See core/quantize.h for the contract and
 * docs/quantization.md for what the measurements say it costs.
 */

#include "core/quantize.h"
#include <math.h>
#include <string.h>

/* Levels for a b-bit symmetric grid: [-qmax, qmax] with qmax = 2^(b-1) - 1.
 *
 * The negative end could hold one more level (two's complement runs to
 * -2^(b-1)), and dropping it costs one level out of 128 at 8 bits and one out
 * of 8 at 4 bits. It is dropped anyway, because a grid symmetric about zero is
 * what makes `value == scale * q` exactly invertible in both directions and
 * keeps the reconstruction free of a sign-dependent bias. At 4 bits that is a
 * real 12% of the range given up, and it is the kind of choice that belongs in
 * a comment rather than in a reader's head. */
static int quant_levels(int bits) {
    return (1 << (bits - 1)) - 1;
}

static int granularity_in_range(quant_granularity_t granularity) {
    int value = (int)granularity;
    return value >= 0 && value < (int)QUANT_GRANULARITY_COUNT;
}

static int arguments_valid(const float *values, size_t rows, size_t cols,
                           int bits, quant_granularity_t granularity) {
    if (!values) return 0;
    if (rows == 0 || cols == 0) return 0;
    if (bits < QUANT_MIN_BITS || bits > QUANT_MAX_BITS) return 0;
    if (!granularity_in_range(granularity)) return 0;
    return 1;
}

static void error_reset(quant_error_t *error) {
    memset(error, 0, sizeof(*error));
}

/* Round half away from zero, matching what the integer grid's midpoints imply
 * and what nearbyintf() would give under the default rounding mode only by
 * coincidence. Written explicitly so the tie behaviour is not a property of the
 * floating-point environment: -ffast-math and a changed rounding mode are both
 * in play in this project's builds. */
static float round_half_away(float value) {
    return (value >= 0.0f) ? floorf(value + 0.5f) : ceilf(value - 0.5f);
}

/* One group of `count` contiguous values sharing a scale. Returns the number
 * of values that landed on the extreme level, and accumulates squared error. */
static void quantize_group(const float *src, float *dst, size_t count,
                           size_t stride, int qmax, double *sum_squared_error,
                           double *sum_squared_original, float *max_abs_error,
                           size_t *clipped, unsigned char *level_seen,
                           size_t level_seen_size) {
    float peak = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float magnitude = fabsf(src[i * stride]);
        if (magnitude > peak) peak = magnitude;
    }

    /* An all-zero group has no scale to derive. Reconstructing it as zeros is
     * both exact and what any scale would have produced, so it is copied
     * through rather than given an invented one. */
    if (!(peak > 0.0f)) {
        for (size_t i = 0; i < count; i++) dst[i * stride] = src[i * stride];
        if (level_seen && level_seen_size > 0) level_seen[qmax] = 1; /* the zero level */
        return;
    }

    const float scale = peak / (float)qmax;
    const float inverse = (float)qmax / peak;

    for (size_t i = 0; i < count; i++) {
        /* Read once, up front. The in-place form passes src == dst, so every
         * read after the store below would return the reconstruction rather
         * than the original - which made the measured error identically zero
         * while the quantization itself was perfectly correct. An operation
         * that overwrites its input cannot also measure against it afterwards.
         * test_quantize.c pins this by requiring the in-place and out-of-place
         * forms to report the same error, which is the only comparison that
         * would have caught it. */
        const float original = src[i * stride];

        float level = round_half_away(original * inverse);
        if (level > (float)qmax) level = (float)qmax;
        if (level < -(float)qmax) level = -(float)qmax;

        int index = (int)level;
        if (index == qmax || index == -qmax) (*clipped)++;
        if (level_seen && level_seen_size > 0) {
            size_t slot = (size_t)(index + qmax);
            if (slot < level_seen_size) level_seen[slot] = 1;
        }

        float reconstructed = scale * level;
        dst[i * stride] = reconstructed;

        double difference = (double)original - (double)reconstructed;
        *sum_squared_error += difference * difference;
        *sum_squared_original += (double)original * (double)original;
        float absolute = fabsf((float)difference);
        if (absolute > *max_abs_error) *max_abs_error = absolute;
    }
}

int quantize_dequantize_into(const float *src, float *dst,
                             size_t rows, size_t cols,
                             int bits, quant_granularity_t granularity,
                             quant_error_t *error_out) {
    if (!dst) return -1;
    if (!arguments_valid(src, rows, cols, bits, granularity)) return -1;

    const int qmax = quant_levels(bits);
    const size_t total = rows * cols;

    /* One byte per representable level, so "how much of the grid did this
     * tensor actually use" is a count rather than an estimate. 2^16 bytes at
     * the widest supported width, on the stack of a function called once per
     * tensor. */
    unsigned char level_seen[1 << QUANT_MAX_BITS];
    const size_t level_seen_size = (size_t)(2 * qmax + 1);
    memset(level_seen, 0, level_seen_size);

    double sum_squared_error = 0.0;
    double sum_squared_original = 0.0;
    float max_abs_error = 0.0f;
    size_t clipped = 0;

    switch (granularity) {
        case QUANT_GRANULARITY_TENSOR:
            quantize_group(src, dst, total, 1, qmax, &sum_squared_error,
                           &sum_squared_original, &max_abs_error, &clipped,
                           level_seen, level_seen_size);
            break;
        case QUANT_GRANULARITY_ROW:
            for (size_t r = 0; r < rows; r++) {
                quantize_group(src + r * cols, dst + r * cols, cols, 1, qmax,
                               &sum_squared_error, &sum_squared_original,
                               &max_abs_error, &clipped, level_seen,
                               level_seen_size);
            }
            break;
        case QUANT_GRANULARITY_COLUMN:
        default:
            /* One group per column, walked with a `cols` stride. */
            for (size_t c = 0; c < cols; c++) {
                quantize_group(src + c, dst + c, rows, cols, qmax,
                               &sum_squared_error, &sum_squared_original,
                               &max_abs_error, &clipped, level_seen,
                               level_seen_size);
            }
            break;
    }

    if (error_out) {
        error_reset(error_out);
        error_out->count = total;
        error_out->max_abs_error = max_abs_error;
        error_out->rms_error = sqrt(sum_squared_error / (double)total);
        error_out->original_rms = sqrt(sum_squared_original / (double)total);
        error_out->rms_relative = (error_out->original_rms > 0.0)
                                ? error_out->rms_error / error_out->original_rms
                                : 0.0;
        error_out->clipped = clipped;
        size_t used = 0;
        for (size_t i = 0; i < level_seen_size; i++) used += level_seen[i] ? 1u : 0u;
        error_out->levels_used = used;
    }
    return 0;
}

int quantize_dequantize(float *values, size_t rows, size_t cols,
                        int bits, quant_granularity_t granularity,
                        quant_error_t *error_out) {
    return quantize_dequantize_into(values, values, rows, cols, bits,
                                    granularity, error_out);
}

void quant_error_accumulate(quant_error_t *total, const quant_error_t *tensor) {
    if (!total || !tensor || tensor->count == 0) return;

    /* Quadratic mean weighted by value count. Averaging the per-tensor RMS
     * figures instead would give a 256-element bias vector the same weight as a
     * million-element embedding matrix, which is how a model-wide error figure
     * ends up dominated by the tensors that matter least. */
    double combined = (double)total->count + (double)tensor->count;
    double total_squared = total->rms_error * total->rms_error * (double)total->count;
    double tensor_squared = tensor->rms_error * tensor->rms_error * (double)tensor->count;
    double original_squared =
        total->original_rms * total->original_rms * (double)total->count +
        tensor->original_rms * tensor->original_rms * (double)tensor->count;

    total->rms_error = sqrt((total_squared + tensor_squared) / combined);
    total->original_rms = sqrt(original_squared / combined);
    total->rms_relative = (total->original_rms > 0.0)
                        ? total->rms_error / total->original_rms
                        : 0.0;
    if (tensor->max_abs_error > total->max_abs_error) {
        total->max_abs_error = tensor->max_abs_error;
    }
    total->clipped += tensor->clipped;
    total->count = (size_t)combined;
    /* Level usage does not combine meaningfully across tensors with different
     * scales, so the aggregate reports the largest any single tensor reached
     * rather than a sum that would exceed the grid. */
    if (tensor->levels_used > total->levels_used) {
        total->levels_used = tensor->levels_used;
    }
}

static const char *const granularity_names[QUANT_GRANULARITY_COUNT] = {
    "tensor", "row", "column"
};

const char *quant_granularity_name(quant_granularity_t granularity) {
    if (!granularity_in_range(granularity)) return "unknown";
    return granularity_names[granularity];
}

int quant_granularity_from_name(const char *name, quant_granularity_t *out) {
    if (!name || !out) return -1;
    for (int i = 0; i < (int)QUANT_GRANULARITY_COUNT; i++) {
        if (strcmp(name, granularity_names[i]) == 0) {
            *out = (quant_granularity_t)i;
            return 0;
        }
    }
    return -1;
}
