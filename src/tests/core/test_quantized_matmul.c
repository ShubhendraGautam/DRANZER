/* INT8/INT4 weight-only matmul correctness.
 *
 * The reference is float32 matmul over the values decoded from the packed
 * matrix, not over the original weights. That isolates kernel correctness
 * from the accuracy cost already covered by test_quantize.c. */

#include "core/cpu_features.h"
#include "core/matmul.h"
#include "core/quantized_matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

typedef struct { size_t m, k, n; const char *name; } shape_t;
static const shape_t shapes[] = {
    {1, 64, 257, "single-token projection"},
    {17, 33, 47, "unaligned extents"},
    {5, 7, 3, "sub-vector extents"},
    {16, 16, 16, "one AVX-512 vector"},
};

static void fill(float *values, size_t count, size_t salt) {
    for (size_t i = 0; i < count; i++) {
        values[i] = (float)((int)((i * 29u + salt * 17u) % 97u) - 48) / 31.0f;
    }
}

static void check_shape(const shape_t *shape, int bits,
                        quant_granularity_t granularity, cpu_isa_t isa) {
    size_t a_count = shape->m * shape->k;
    size_t b_count = shape->k * shape->n;
    size_t c_count = shape->m * shape->n;
    float *a = malloc(a_count * sizeof(*a));
    float *b = malloc(b_count * sizeof(*b));
    float *decoded = malloc(b_count * sizeof(*decoded));
    float *reference = malloc(c_count * sizeof(*reference));
    float *actual = malloc(c_count * sizeof(*actual));
    quantized_weight_matrix_t weights = {0};
    if (!a || !b || !decoded || !reference || !actual) {
        fprintf(stderr, "FAIL: allocation for %s\n", shape->name);
        failures++;
        goto cleanup;
    }
    fill(a, a_count, 1);
    fill(b, b_count, 2);
    if (quantized_weight_matrix_encode(&weights, b, shape->k, shape->n,
                                       bits, granularity) != 0 ||
        quantize_unpack(weights.scales, weights.scale_count,
                        weights.packed, weights.packed_size,
                        shape->k, shape->n, bits, granularity, decoded) != 0) {
        fprintf(stderr, "FAIL: encode/decode for %s, INT%d/%s\n", shape->name,
                bits, quant_granularity_name(granularity));
        failures++;
        goto cleanup;
    }

    cpu_features_set_max_isa(isa);
    matrix_multiply(a, decoded, reference, shape->m, shape->k, shape->n);
    if (matmul_quantized_weight(a, &weights, actual,
                                shape->m, shape->k, 0) != 0) {
        fprintf(stderr, "FAIL: kernel rejected %s, INT%d/%s\n", shape->name,
                bits, quant_granularity_name(granularity));
        failures++;
        cpu_features_clear_max_isa();
        goto cleanup;
    }
    cpu_features_clear_max_isa();

    float worst = 0.0f, magnitude = 0.0f;
    for (size_t i = 0; i < c_count; i++) {
        float difference = fabsf(reference[i] - actual[i]);
        if (difference > worst) worst = difference;
        float value_magnitude = fabsf(reference[i]);
        if (value_magnitude > magnitude) magnitude = value_magnitude;
    }
    float tolerance = 1e-4f * (1.0f + magnitude);
    if (!(worst <= tolerance)) {
        fprintf(stderr,
                "FAIL: %s INT%d/%s diverged: %.8g > %.8g\n",
                shape->name, bits, quant_granularity_name(granularity),
                (double)worst, (double)tolerance);
        failures++;
    }

cleanup:
    cpu_features_clear_max_isa();
    quantized_weight_matrix_free(&weights);
    free(a); free(b); free(decoded); free(reference); free(actual);
}

static void check_storage_and_rejection(void) {
    float values[15], snapshot[15];
    fill(values, 15, 5);
    memcpy(snapshot, values, sizeof(values));
    for (int bits = 4; bits <= 8; bits += 4) {
        quantized_weight_matrix_t weights = {0};
        size_t expected_bytes = (15u * (size_t)bits + 7u) / 8u;
        if (quantized_weight_matrix_encode(&weights, values, 3, 5, bits,
                                           QUANT_GRANULARITY_COLUMN) != 0 ||
            weights.packed_size != expected_bytes || weights.scale_count != 5) {
            fprintf(stderr, "FAIL: INT%d storage accounting\n", bits);
            failures++;
        }
        quantized_weight_matrix_free(&weights);
    }
    if (memcmp(values, snapshot, sizeof(values)) != 0) {
        fprintf(stderr, "FAIL: encoding modified source weights\n");
        failures++;
    }
    quantized_weight_matrix_t invalid = {0};
    float output[5] = {0};
    if (quantized_weight_matrix_encode(&invalid, values, 3, 5, 3,
                                       QUANT_GRANULARITY_COLUMN) != -1 ||
        matmul_quantized_weight(values, &invalid, output, 1, 3, 0) != -1) {
        fprintf(stderr, "FAIL: unsupported runtime format was accepted\n");
        failures++;
    }
    quantized_weight_matrix_free(&invalid);
}

int main(void) {
    printf("=== INT8/INT4 quantized-weight matmul ===\n");
    printf("cpu: %s\n", cpu_features_summary());
    check_storage_and_rejection();

    const struct { cpu_isa_t isa; const char *name; } rungs[] = {
        {CPU_ISA_AVX512, "AVX-512"},
        {CPU_ISA_AVX2, "AVX2"},
        {CPU_ISA_BASELINE, "portable"},
    };
    int usable[3];
    for (size_t i = 0; i < 3; i++) usable[i] = cpu_isa_available(rungs[i].isa);
    const quant_granularity_t granularities[] = {
        QUANT_GRANULARITY_TENSOR,
        QUANT_GRANULARITY_ROW,
        QUANT_GRANULARITY_COLUMN,
    };

    for (size_t r = 0; r < 3; r++) {
        if (!usable[r]) {
            printf("%s unavailable: skipped\n", rungs[r].name);
            continue;
        }
        printf("%s:\n", rungs[r].name);
        for (int bits = 4; bits <= 8; bits += 4) {
            for (size_t g = 0; g < 3; g++) {
                for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
                    check_shape(&shapes[s], bits, granularities[g], rungs[r].isa);
                }
            }
        }
    }
    cpu_features_clear_max_isa();
    printf("%s\n", failures ? "QUANTIZED MATMUL CHECK FAILED"
                              : "QUANTIZED MATMUL CHECK PASSED");
    return failures ? 1 : 0;
}
