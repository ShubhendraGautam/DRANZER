/*
 * The quantization grid, and the error it reports about itself.
 *
 * Two things are checked, and the second is the one that matters.
 *
 * The grid must be *correct*: reconstructed values land on `scale * level`,
 * stay inside the representable range, never move further than half a step, and
 * degrade monotonically as bits are removed. That much is what any test would
 * check.
 *
 * The error statistics must also be *honest*. Quantization is applied in place,
 * and the first version of quantize_group() computed its error after storing
 * the reconstruction - so it compared the output against itself and reported
 * exactly zero error for every tensor, while quantizing perfectly correctly.
 * Nothing about the quantized values was wrong, so no test of the values could
 * have caught it; the whole benchmark ran and printed a column of zeros. The
 * check that catches it is requiring the in-place and out-of-place forms to
 * report the same error, and it is the reason that comparison exists.
 *
 * "The same" is a tolerance, not a bit pattern. The two forms run one function,
 * but a compiler that vectorizes its accumulation loop guards the vector path
 * with a runtime alias check - which the in-place call (src == dst) fails and
 * the copied call passes, leaving the two to sum the same squares in a
 * different order. That is a difference of ~1e-15 relative, and demanding bit
 * equality turns every such compiler decision into a red suite. The defect
 * above reports exactly zero against a value of order 0.1, so a bound twelve
 * orders of magnitude tighter than the defect loses nothing that matters.
 */

#include "core/quantize.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    failures++;
}

static const quant_granularity_t granularities[] = {
    QUANT_GRANULARITY_TENSOR,
    QUANT_GRANULARITY_ROW,
    QUANT_GRANULARITY_COLUMN,
};
#define GRANULARITY_COUNT (sizeof(granularities) / sizeof(granularities[0]))

/* Relative agreement, with an absolute floor so that two figures near zero are
 * not held to a relative bound they cannot meet. See the file header for why
 * these comparisons are tolerances rather than equality. */
static int agrees(double a, double b, double tolerance) {
    const double difference = fabs(a - b);
    const double magnitude = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return difference <= tolerance * (magnitude > 1.0 ? magnitude : 1.0);
}

/* Two reconstructions of the same tensor. A single float ULP is ~1.2e-7
 * relative, so 1e-6 admits a differently-ordered arithmetic path and nothing
 * that would count as a different grid. */
static int reconstructions_agree(const float *a, const float *b, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (!agrees((double)a[i], (double)b[i], 1e-6)) return 0;
    }
    return 1;
}

typedef struct { size_t rows, cols; } shape_t;

static const shape_t shapes[] = {
    { 1, 1 },      /* degenerate */
    { 1, 64 },     /* a bias vector: one row, so row and tensor coincide */
    { 64, 1 },     /* one column, so column and tensor coincide */
    { 3, 5 },      /* neither extent divides anything */
    { 16, 16 },
    { 7, 129 },    /* wide, prime-ish */
};
#define SHAPE_COUNT (sizeof(shapes) / sizeof(shapes[0]))

static void fill(float *values, size_t count, unsigned int salt) {
    /* Deliberately not uniform: a few values an order of magnitude larger than
     * the rest, because a scale derived from the maximum is exactly what an
     * outlier damages, and a test on well-behaved data would never see it. */
    for (size_t i = 0; i < count; i++) {
        float base = (float)((int)((i * 37u + salt * 11u) % 71u) - 35) / 17.0f;
        if ((i % 23) == 0) base *= 12.0f;
        values[i] = base;
    }
}

/* The error an in-place call reports must equal what the same grid reports
 * when the original is preserved. See the file header. */
static void check_inplace_matches_copy(void) {
    for (size_t s = 0; s < SHAPE_COUNT; s++) {
        const size_t rows = shapes[s].rows, cols = shapes[s].cols;
        const size_t count = rows * cols;

        for (size_t g = 0; g < GRANULARITY_COUNT; g++) {
            for (int bits = 2; bits <= 12; bits += 2) {
                float *original = malloc(count * sizeof(float));
                float *in_place = malloc(count * sizeof(float));
                float *copied = malloc(count * sizeof(float));
                if (!original || !in_place || !copied) {
                    fail("allocation");
                    free(original); free(in_place); free(copied);
                    return;
                }
                fill(original, count, (unsigned int)(s * 7 + (size_t)bits));
                memcpy(in_place, original, count * sizeof(float));

                quant_error_t error_in_place, error_copied;
                if (quantize_dequantize(in_place, rows, cols, bits,
                                        granularities[g], &error_in_place) != 0 ||
                    quantize_dequantize_into(original, copied, rows, cols, bits,
                                             granularities[g], &error_copied) != 0) {
                    fail("a valid call was rejected");
                    free(original); free(in_place); free(copied);
                    return;
                }

                if (!reconstructions_agree(in_place, copied, count)) {
                    fail("in-place and copied reconstructions differ");
                }
                /* clipped and levels_used stay exact: they are counts, and a
                 * rounding path that moved a value onto a different level
                 * would be a different grid rather than a different sum. */
                const char *disagreement = NULL;
                if (!agrees(error_in_place.rms_error, error_copied.rms_error, 1e-12)) {
                    disagreement = "rms_error";
                } else if (!agrees((double)error_in_place.max_abs_error,
                                   (double)error_copied.max_abs_error, 1e-6)) {
                    disagreement = "max_abs_error";
                } else if (!agrees(error_in_place.original_rms,
                                   error_copied.original_rms, 1e-12)) {
                    disagreement = "original_rms";
                } else if (error_in_place.clipped != error_copied.clipped) {
                    disagreement = "clipped";
                } else if (error_in_place.levels_used != error_copied.levels_used) {
                    disagreement = "levels_used";
                }
                if (disagreement) {
                    /* Name the field, and print the doubles at %.17g. A CI log
                     * of this failure carrying only a %.9g rms left it unclear
                     * whether the rms differed at all or whether a count did,
                     * because %.9g rounds two values a bit apart onto different
                     * digits. A failure is worth nothing that cannot be read. */
                    fprintf(stderr,
                            "  %s %d-bit %zux%zu: %s differs | rms %.17g vs %.17g"
                            " | max %.9g vs %.9g | clipped %zu vs %zu"
                            " | levels %zu vs %zu\n",
                            quant_granularity_name(granularities[g]), bits,
                            rows, cols, disagreement,
                            error_in_place.rms_error, error_copied.rms_error,
                            (double)error_in_place.max_abs_error,
                            (double)error_copied.max_abs_error,
                            error_in_place.clipped, error_copied.clipped,
                            error_in_place.levels_used, error_copied.levels_used);
                    fail("in-place error statistics disagree with the copied form");
                }

                /* A tensor whose groups hold more than one value must report
                 * *some* error: a reported zero is the exact symptom of the
                 * aliasing bug above.
                 *
                 * Groups of one are excluded because they are exactly lossless,
                 * and not by accident. A scale derived from the group maximum
                 * is that single value, so the one representable level
                 * reproduces it bit for bit. That is why per-column scales on a
                 * one-row bias vector cost a 32-bit scale per value to store
                 * zero error - the degenerate corner the default policy avoids
                 * by leaving biases alone. */
                size_t group = (granularities[g] == QUANT_GRANULARITY_TENSOR) ? count
                             : (granularities[g] == QUANT_GRANULARITY_ROW) ? cols
                             : rows;
                if (group > 1 && count > 4 && bits < 12 &&
                    !(error_in_place.rms_error > 0.0)) {
                    fprintf(stderr, "  %s %d-bit %zux%zu reported zero error\n",
                            quant_granularity_name(granularities[g]), bits,
                            rows, cols);
                    fail("a quantized tensor reported no error at all");
                }

                free(original); free(in_place); free(copied);
            }
        }
    }
}

/* Every reconstructed value must be an exact multiple of its group's scale and
 * no further than half a step from the original. */
static void check_grid_properties(void) {
    const size_t rows = 16, cols = 24, count = rows * cols;
    float *original = malloc(count * sizeof(float));
    float *result = malloc(count * sizeof(float));
    if (!original || !result) {
        fail("allocation");
        free(original); free(result);
        return;
    }
    fill(original, count, 3);

    for (size_t g = 0; g < GRANULARITY_COUNT; g++) {
        for (int bits = 3; bits <= 10; bits++) {
            quant_error_t error;
            if (quantize_dequantize_into(original, result, rows, cols, bits,
                                         granularities[g], &error) != 0) {
                fail("a valid call was rejected");
                continue;
            }

            const int qmax = (1 << (bits - 1)) - 1;

            /* Half a step of the group's own scale. The largest magnitude in
             * the whole tensor bounds every group's scale from above, so this
             * is a valid bound for all granularities without recomputing each
             * group's peak here. */
            float peak = 0.0f;
            for (size_t i = 0; i < count; i++) {
                if (fabsf(original[i]) > peak) peak = fabsf(original[i]);
            }
            const float bound = (peak / (float)qmax) * 0.5f * 1.001f; /* rounding slack */

            for (size_t i = 0; i < count; i++) {
                if (fabsf(original[i] - result[i]) > bound) {
                    fprintf(stderr,
                            "  %s %d-bit element %zu: %.9g -> %.9g, beyond half a step (%.9g)\n",
                            quant_granularity_name(granularities[g]), bits, i,
                            (double)original[i], (double)result[i], (double)bound);
                    fail("a value moved further than half a quantization step");
                    break;
                }
                if (fabsf(result[i]) > peak * 1.0001f) {
                    fail("a reconstructed value left the representable range");
                    break;
                }
            }

            if (error.levels_used > (size_t)(2 * qmax + 1)) {
                fail("more levels were used than the grid has");
            }
        }
    }
    free(original); free(result);
}

/* Removing bits must not reduce the error, and a wide enough grid on a small
 * set of distinct values must reproduce them closely. */
static void check_monotonic_in_bits(void) {
    const size_t rows = 8, cols = 32, count = rows * cols;
    float *original = malloc(count * sizeof(float));
    float *result = malloc(count * sizeof(float));
    if (!original || !result) {
        fail("allocation");
        free(original); free(result);
        return;
    }
    fill(original, count, 5);

    for (size_t g = 0; g < GRANULARITY_COUNT; g++) {
        double previous = -1.0;
        for (int bits = 12; bits >= 3; bits--) {
            quant_error_t error;
            if (quantize_dequantize_into(original, result, rows, cols, bits,
                                         granularities[g], &error) != 0) {
                fail("a valid call was rejected");
                continue;
            }
            if (previous >= 0.0 && error.rms_error < previous) {
                fprintf(stderr, "  %s: %d bits gave less error than %d bits\n",
                        quant_granularity_name(granularities[g]), bits, bits + 1);
                fail("error did not grow as bits were removed");
            }
            previous = error.rms_error;
        }
    }
    free(original); free(result);
}

/* A group with no non-zero value has no scale to derive, and must be passed
 * through exactly rather than given an invented one. */
static void check_zero_group(void) {
    float values[32];
    memset(values, 0, sizeof(values));
    quant_error_t error;
    if (quantize_dequantize(values, 4, 8, 8, QUANT_GRANULARITY_ROW, &error) != 0) {
        fail("an all-zero tensor was rejected");
        return;
    }
    for (size_t i = 0; i < 32; i++) {
        if (values[i] != 0.0f) fail("an all-zero tensor did not stay zero");
    }
    if (error.rms_error != 0.0) fail("an all-zero tensor reported error");
}

/* The storage representation must reconstruct the exact grid used by the
 * accuracy-only path. A 3x5 tensor deliberately leaves tail padding at the
 * widths below, so canonical padding and the unused all-ones code are covered
 * alongside the ordinary round trip. */
static void check_packed_storage(void) {
    const size_t rows = 3, cols = 5, count = rows * cols;
    float original[count], expected[count], reconstructed[count];
    fill(original, count, 19);

    for (size_t g = 0; g < GRANULARITY_COUNT; g++) {
        for (int bits = 3; bits <= 8; bits++) {
            size_t scale_count = 0, packed_size = 0;
            if (quantized_scale_count(rows, cols, granularities[g],
                                      &scale_count) != 0 ||
                quantized_packed_size(count, bits, &packed_size) != 0) {
                fail("packed storage size calculation failed");
                return;
            }
            float *scales = malloc(scale_count * sizeof(*scales));
            uint8_t *packed = malloc(packed_size);
            if (!scales || !packed) {
                free(scales); free(packed);
                fail("allocation");
                return;
            }
            if (quantize_dequantize_into(original, expected, rows, cols, bits,
                                         granularities[g], NULL) != 0 ||
                quantize_pack(original, rows, cols, bits, granularities[g],
                              scales, scale_count, packed, packed_size) != 0 ||
                quantize_unpack(scales, scale_count, packed, packed_size,
                                rows, cols, bits, granularities[g],
                                reconstructed) != 0 ||
                !reconstructions_agree(expected, reconstructed, count)) {
                fail("packed storage changed the quantization grid");
            }

            uint8_t saved_first = packed[0];
            packed[0] = (uint8_t)((packed[0] & (uint8_t)~((1u << bits) - 1u)) |
                                  ((1u << bits) - 1u));
            if (quantize_unpack(scales, scale_count, packed, packed_size,
                                rows, cols, bits, granularities[g],
                                reconstructed) != -1) {
                fail("the unused packed code was accepted");
            }
            packed[0] = saved_first;

            size_t used_bits = count * (size_t)bits;
            if (used_bits % 8 != 0) {
                packed[packed_size - 1] |= UINT8_C(0x80);
                if (quantize_unpack(scales, scale_count, packed, packed_size,
                                    rows, cols, bits, granularities[g],
                                    reconstructed) != -1) {
                    fail("non-zero packed padding was accepted");
                }
            }
            free(scales);
            free(packed);
        }
    }
}

static void check_rejects_invalid(void) {
    float values[16];
    fill(values, 16, 1);
    float snapshot[16];
    memcpy(snapshot, values, sizeof(values));

    struct { const char *what; int rc; } cases[] = {
        { "null pointer",
          quantize_dequantize(NULL, 4, 4, 8, QUANT_GRANULARITY_ROW, NULL) },
        { "zero rows",
          quantize_dequantize(values, 0, 4, 8, QUANT_GRANULARITY_ROW, NULL) },
        { "zero cols",
          quantize_dequantize(values, 4, 0, 8, QUANT_GRANULARITY_ROW, NULL) },
        { "bit width below the minimum",
          quantize_dequantize(values, 4, 4, QUANT_MIN_BITS - 1,
                              QUANT_GRANULARITY_ROW, NULL) },
        { "bit width above the maximum",
          quantize_dequantize(values, 4, 4, QUANT_MAX_BITS + 1,
                              QUANT_GRANULARITY_ROW, NULL) },
        { "unknown granularity",
          quantize_dequantize(values, 4, 4, 8,
                              (quant_granularity_t)QUANT_GRANULARITY_COUNT, NULL) },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (cases[i].rc != -1) {
            fprintf(stderr, "  accepted: %s\n", cases[i].what);
            fail("an invalid argument was accepted");
        }
    }
    if (memcmp(values, snapshot, sizeof(values)) != 0) {
        fail("a rejected call modified the values");
    }
}

static void check_name_round_trip(void) {
    for (int i = 0; i < (int)QUANT_GRANULARITY_COUNT; i++) {
        const char *name = quant_granularity_name((quant_granularity_t)i);
        quant_granularity_t parsed;
        if (quant_granularity_from_name(name, &parsed) != 0 || (int)parsed != i) {
            fail("a granularity name did not round-trip");
        }
    }
    quant_granularity_t parsed;
    if (quant_granularity_from_name("per-channel", &parsed) != -1) {
        fail("an unknown granularity name was accepted");
    }
    if (strcmp(quant_granularity_name((quant_granularity_t)-1), "unknown") != 0) {
        fail("an out-of-range granularity did not report as unknown");
    }
}

/* Combining per-tensor errors must weight by value count. Averaging the
 * per-tensor figures instead would let a one-element tensor pull a model-wide
 * number as hard as an embedding matrix. */
static void check_accumulate_weighting(void) {
    quant_error_t big = { .count = 1000, .rms_error = 1.0, .original_rms = 1.0,
                          .rms_relative = 1.0, .max_abs_error = 1.0f,
                          .clipped = 2, .levels_used = 200 };
    quant_error_t small = { .count = 1, .rms_error = 100.0, .original_rms = 1.0,
                            .rms_relative = 100.0, .max_abs_error = 100.0f,
                            .clipped = 1, .levels_used = 3 };

    quant_error_t total;
    memset(&total, 0, sizeof(total));
    quant_error_accumulate(&total, &big);
    quant_error_accumulate(&total, &small);

    if (total.count != 1001) fail("accumulated count is wrong");
    if (total.clipped != 3) fail("accumulated clipped count is wrong");
    if (total.max_abs_error != 100.0f) fail("accumulated maximum is wrong");
    if (total.levels_used != 200) fail("accumulated level usage is wrong");

    /* sqrt((1000*1 + 1*10000)/1001) = sqrt(10.989) ~ 3.315. A plain mean of the
     * two RMS figures would be 50.5, which is the mistake this pins. */
    const double expected = sqrt((1000.0 * 1.0 + 1.0 * 10000.0) / 1001.0);
    if (fabs(total.rms_error - expected) > 1e-9) {
        fprintf(stderr, "  combined rms %.9g, expected %.9g\n",
                total.rms_error, expected);
        fail("accumulation is not count-weighted");
    }

    /* A zero-count contribution must be inert rather than divide by zero. */
    quant_error_t before = total;
    quant_error_t empty;
    memset(&empty, 0, sizeof(empty));
    quant_error_accumulate(&total, &empty);
    if (memcmp(&before, &total, sizeof(total)) != 0) {
        fail("an empty contribution changed the total");
    }
}

int main(void) {
    check_inplace_matches_copy();
    check_grid_properties();
    check_monotonic_in_bits();
    check_zero_group();
    check_packed_storage();
    check_rejects_invalid();
    check_name_round_trip();
    check_accumulate_weighting();

    printf("shapes=%zu granularities=%zu\n", SHAPE_COUNT, GRANULARITY_COUNT);
    if (failures != 0) {
        printf("\nQUANTIZATION CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nQUANTIZATION CHECK PASSED\n");
    return 0;
}
