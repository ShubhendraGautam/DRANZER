/* Isolated bandwidth benchmark for fp32, bf16, INT8, and packed INT4 weights.
 * Conversion happens once before timing; every timed call widens weights in
 * the kernel and accumulates in float32. */

#include "core/bf16.h"
#include "core/matmul.h"
#include "core/quantized_matmul.h"
#include "tools/bench_support.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RESULT_PATH "quantized_matmul_results_v1.csv"
#define REPEATS 5

typedef struct { const char *name; size_t m, k, n; } shape_t;
static const shape_t full_shapes[] = {
    {"decode_head", 1, 64, 1000},
    {"prefill_ffn_down", 32, 256, 64},
    {"prefill_ffn_up", 64, 64, 256},
    {"medium_ffn_up", 128, 256, 1024},
    {"medium_ffn_down", 128, 1024, 256},
    {"all_position_head", 128, 256, 4000},
    {"large_square", 128, 1024, 1024},
    {"past_l3", 128, 2048, 2048},
};

typedef enum { FORMAT_FP32, FORMAT_BF16, FORMAT_INT8, FORMAT_INT4 } format_t;
static const char *const format_names[] = {"fp32", "bf16", "int8", "int4"};
static volatile float result_sink;

typedef struct {
    const float *a;
    const float *b;
    const bf16_t *bf16;
    const quantized_weight_matrix_t *int8;
    const quantized_weight_matrix_t *int4;
    float *c;
    size_t m, k, n;
} inputs_t;

static void run_format(format_t format, const inputs_t *inputs) {
    switch (format) {
        case FORMAT_FP32:
            matrix_multiply(inputs->a, inputs->b, inputs->c,
                            inputs->m, inputs->k, inputs->n);
            break;
        case FORMAT_BF16:
            matmul_bf16_weight(inputs->a, inputs->bf16, inputs->c,
                               inputs->m, inputs->k, inputs->n, 0);
            break;
        case FORMAT_INT8:
            (void)matmul_quantized_weight(inputs->a, inputs->int8, inputs->c,
                                          inputs->m, inputs->k, 0);
            break;
        case FORMAT_INT4:
            (void)matmul_quantized_weight(inputs->a, inputs->int4, inputs->c,
                                          inputs->m, inputs->k, 0);
            break;
    }
}

static double time_format(format_t format, const inputs_t *inputs,
                          size_t iterations) {
    double started = bench_now_sec();
    for (size_t i = 0; i < iterations; i++) run_format(format, inputs);
    double elapsed = bench_now_sec() - started;
    result_sink += inputs->c[(iterations * 131u) % (inputs->m * inputs->n)];
    return elapsed / (double)iterations;
}

static size_t choose_iterations(const inputs_t *inputs, int quick) {
    const double target = quick ? 0.015 : 0.08;
    size_t iterations = 1;
    for (;;) {
        double elapsed = time_format(FORMAT_FP32, inputs, iterations) *
                         (double)iterations;
        if (elapsed >= target || iterations >= (1u << 18)) return iterations;
        double multiplier = elapsed > 0.0 ? target / elapsed : 2.0;
        if (multiplier < 2.0) multiplier = 2.0;
        size_t next = (size_t)((double)iterations * multiplier);
        if (next <= iterations) next = iterations + 1;
        iterations = next > (1u << 18) ? (1u << 18) : next;
    }
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static int validate_format(format_t format, const inputs_t *inputs,
                           size_t b_count, size_t c_count) {
    if (format == FORMAT_FP32) return 0;
    float *decoded = malloc(b_count * sizeof(*decoded));
    float *reference = malloc(c_count * sizeof(*reference));
    if (!decoded || !reference) {
        free(decoded); free(reference);
        return -1;
    }
    int decode_rc = 0;
    if (format == FORMAT_BF16) {
        bf16_decode_array(inputs->bf16, decoded, b_count);
    } else {
        const quantized_weight_matrix_t *weights =
            format == FORMAT_INT8 ? inputs->int8 : inputs->int4;
        decode_rc = quantize_unpack(weights->scales, weights->scale_count,
                                    weights->packed, weights->packed_size,
                                    weights->rows, weights->cols, weights->bits,
                                    weights->granularity, decoded);
    }
    if (decode_rc != 0) {
        free(decoded); free(reference);
        return -1;
    }
    matrix_multiply(inputs->a, decoded, reference,
                    inputs->m, inputs->k, inputs->n);
    run_format(format, inputs);
    float worst = 0.0f, magnitude = 0.0f;
    for (size_t i = 0; i < c_count; i++) {
        float difference = fabsf(reference[i] - inputs->c[i]);
        if (difference > worst) worst = difference;
        float value_magnitude = fabsf(reference[i]);
        if (value_magnitude > magnitude) magnitude = value_magnitude;
    }
    free(decoded); free(reference);
    return worst <= 1e-4f * (1.0f + magnitude) ? 0 : -1;
}

static int run_shape(const shape_t *shape, int quick, FILE *csv,
                     const char *timestamp, const bench_metadata_t *metadata) {
    size_t a_count = shape->m * shape->k;
    size_t b_count = shape->k * shape->n;
    size_t c_count = shape->m * shape->n;
    float *a = malloc(a_count * sizeof(*a));
    float *b = malloc(b_count * sizeof(*b));
    float *c = malloc(c_count * sizeof(*c));
    bf16_t *b16 = malloc(b_count * sizeof(*b16));
    quantized_weight_matrix_t int8 = {0}, int4 = {0};
    if (!a || !b || !c || !b16) goto fail;
    for (size_t i = 0; i < a_count; i++) {
        a[i] = (float)((int)((i * 13u + 7u) % 101u) - 50) / 53.0f;
    }
    for (size_t i = 0; i < b_count; i++) {
        b[i] = (float)((int)((i * 17u + 11u) % 103u) - 51) / 47.0f;
    }
    bf16_encode_array(b, b16, b_count);
    if (quantized_weight_matrix_encode(&int8, b, shape->k, shape->n, 8,
                                       QUANT_GRANULARITY_COLUMN) != 0 ||
        quantized_weight_matrix_encode(&int4, b, shape->k, shape->n, 4,
                                       QUANT_GRANULARITY_COLUMN) != 0) goto fail;

    inputs_t inputs = {a, b, b16, &int8, &int4, c,
                       shape->m, shape->k, shape->n};
    for (int format = FORMAT_BF16; format <= FORMAT_INT4; format++) {
        if (validate_format((format_t)format, &inputs, b_count, c_count) != 0) {
            fprintf(stderr, "incorrect %s kernel on %s; refusing to time it\n",
                    format_names[format], shape->name);
            goto fail;
        }
    }
    size_t iterations = choose_iterations(&inputs, quick);
    double samples[4][REPEATS];
    for (int format = 0; format < 4; format++) run_format((format_t)format, &inputs);
    for (size_t repeat = 0; repeat < REPEATS; repeat++) {
        if ((repeat & 1u) == 0) {
            for (int format = 0; format < 4; format++) {
                samples[format][repeat] = time_format((format_t)format, &inputs, iterations);
            }
        } else {
            for (int format = 3; format >= 0; format--) {
                samples[format][repeat] = time_format((format_t)format, &inputs, iterations);
            }
        }
    }
    for (int format = 0; format < 4; format++) {
        qsort(samples[format], REPEATS, sizeof(double), compare_double);
    }
    double baseline = samples[FORMAT_FP32][REPEATS / 2];
    printf("%-20s %4zux%-4zux%-4zu B=%7.2f MiB", shape->name,
           shape->m, shape->k, shape->n,
           (double)(b_count * sizeof(float)) / (1024.0 * 1024.0));
    for (int format = 0; format < 4; format++) {
        double median = samples[format][REPEATS / 2];
        double speedup = median > 0.0 ? baseline / median : 0.0;
        printf("  %s %.3fx", format_names[format], speedup);
        if (csv) {
            fprintf(csv, "%s,%s,%zu,%zu,%zu,%s,%zu,%.9g,%.6f",
                    timestamp, shape->name, shape->m, shape->k, shape->n,
                    format_names[format], iterations, median, speedup);
            bench_csv_metadata(csv, metadata);
        }
    }
    putchar('\n');
    quantized_weight_matrix_free(&int8);
    quantized_weight_matrix_free(&int4);
    free(a); free(b); free(c); free(b16);
    return 0;

fail:
    quantized_weight_matrix_free(&int8);
    quantized_weight_matrix_free(&int4);
    free(a); free(b); free(c); free(b16);
    return -1;
}

int main(int argc, char **argv) {
    int quick = 0;
    const char *csv_path = RESULT_PATH;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) quick = 1;
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csv_path = argv[++i];
        else {
            fprintf(stderr, "usage: %s [--quick] [--csv PATH]\n", argv[0]);
            return 2;
        }
    }
    bench_metadata_t metadata;
    bench_collect_metadata(&metadata);
    bench_print_metadata(&metadata);
    FILE *csv = fopen(csv_path, "a+");
    if (!csv) {
        perror(csv_path);
        return 1;
    }
    if (fseek(csv, 0, SEEK_END) == 0 && ftell(csv) == 0) {
        fprintf(csv, "timestamp,shape,m,k,n,format,iterations,seconds,speedup_vs_fp32,%s\n",
                BENCH_METADATA_CSV_HEADER);
    }
    char timestamp[32] = "unknown";
    time_t now = time(NULL);
    struct tm broken_down;
    if (gmtime_r(&now, &broken_down)) {
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &broken_down);
    }
    size_t shape_count = quick ? 3 : sizeof(full_shapes) / sizeof(full_shapes[0]);
    int failed = 0;
    for (size_t i = 0; i < shape_count; i++) {
        if (run_shape(&full_shapes[i], quick, csv, timestamp, &metadata) != 0) failed = 1;
    }
    fclose(csv);
    return failed;
}
