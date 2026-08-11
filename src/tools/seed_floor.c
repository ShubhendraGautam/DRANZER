/* Turn per-seed held-out losses into the machine-readable floor consumed by
 * architecture comparisons. Collection policy lives in tools/statistics.c;
 * this file is deliberately only strict CSV input and artifact output. */

#include "common/fp_bits.h"
#include "tools/statistics.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SEED_ROWS 1024
#define LINE_CAPACITY 4096

typedef struct {
    uint64_t seed;
    double loss;
} seed_loss_t;

typedef struct {
    const char *input;
    const char *output;
    size_t minimum;
    size_t maximum;
    size_t resamples;
    double target_ratio;
    uint64_t bootstrap_seed;
} options_t;

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s INPUT.csv [--output FILE] [--minimum N] [--maximum N]\n"
            "       [--target-ratio R] [--resamples N] [--bootstrap-seed N]\n\n"
            "CSV header: seed,validation_cross_entropy,model_sha256\n",
            program);
}

static int parse_size(const char *text, size_t minimum, size_t maximum,
                      size_t *out) {
    if (!text || !text[0] || !out || text[0] == '-') return 0;
    for (const unsigned char *cursor = (const unsigned char *)text; *cursor; cursor++) {
        if (!isdigit(*cursor)) return 0;
    }
    errno = 0;
    char *end = NULL;
    uintmax_t value = strtoumax(text, &end, 10);
    if (errno || !end || *end || value < minimum || value > maximum) return 0;
    *out = (size_t)value;
    return 1;
}

static int parse_u64(const char *text, uint64_t *out) {
    if (!text || !text[0] || !out || text[0] == '-') return 0;
    for (const unsigned char *cursor = (const unsigned char *)text; *cursor; cursor++) {
        if (!isdigit(*cursor)) return 0;
    }
    errno = 0;
    char *end = NULL;
    uintmax_t value = strtoumax(text, &end, 10);
    if (errno || !end || *end || value > UINT64_MAX) return 0;
    *out = (uint64_t)value;
    return 1;
}

static int parse_positive_double(const char *text, double *out) {
    if (!text || !text[0] || !out) return 0;
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno || !end || *end || !(value > 0.0) ||
        !dranzer_double_is_finite(value)) return 0;
    *out = value;
    return 1;
}

static int parse_options(int argc, char **argv, options_t *options) {
    if (!options || argc < 2) return 0;
    *options = (options_t){
        .input = argv[1],
        .minimum = 8,
        .maximum = 32,
        .resamples = 10000,
        .target_ratio = 0.5,
        .bootstrap_seed = UINT64_C(0x53454544464c4f4f),
    };
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            options->output = argv[++i];
        } else if (strcmp(argv[i], "--minimum") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], 2, MAX_SEED_ROWS, &options->minimum)) return 0;
        } else if (strcmp(argv[i], "--maximum") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], 2, MAX_SEED_ROWS, &options->maximum)) return 0;
        } else if (strcmp(argv[i], "--target-ratio") == 0 && i + 1 < argc) {
            if (!parse_positive_double(argv[++i], &options->target_ratio)) return 0;
        } else if (strcmp(argv[i], "--resamples") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], 1, 10000000, &options->resamples)) return 0;
        } else if (strcmp(argv[i], "--bootstrap-seed") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &options->bootstrap_seed)) return 0;
        } else {
            return 0;
        }
    }
    return options->maximum >= options->minimum &&
           (!options->output || options->output[0]);
}

static int valid_sha256(const char *text) {
    if (!text || strlen(text) != 64) return 0;
    for (size_t i = 0; i < 64; i++) {
        if (!isxdigit((unsigned char)text[i])) return 0;
    }
    return 1;
}

static int parse_row(char *line, seed_loss_t *row) {
    char *first = strchr(line, ',');
    if (!first) return 0;
    *first++ = '\0';
    char *second = strchr(first, ',');
    if (!second) return 0;
    *second++ = '\0';
    if (strchr(second, ',')) return 0;

    size_t length = strlen(second);
    while (length > 0 && (second[length - 1] == '\n' || second[length - 1] == '\r')) {
        second[--length] = '\0';
    }

    if (!parse_u64(line, &row->seed) || !valid_sha256(second)) return 0;
    errno = 0;
    char *end = NULL;
    row->loss = strtod(first, &end);
    return !errno && end && *end == '\0' && row->loss >= 0.0 &&
           dranzer_double_is_finite(row->loss);
}

static int read_rows(const char *path, seed_loss_t *rows, size_t *out_count) {
    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "seed-floor: cannot open %s\n", path);
        return 0;
    }

    char line[LINE_CAPACITY];
    size_t count = 0, line_number = 0;
    int ok = 1, saw_header = 0;
    while (fgets(line, sizeof(line), file)) {
        line_number++;
        if (!strchr(line, '\n') && !feof(file)) {
            fprintf(stderr, "seed-floor: line %zu is too long\n", line_number);
            ok = 0;
            break;
        }
        if (line[0] == '#') continue;
        if (!saw_header) {
            size_t length = strlen(line);
            while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) {
                line[--length] = '\0';
            }
            if (strcmp(line, "seed,validation_cross_entropy,model_sha256") != 0) {
                fprintf(stderr, "seed-floor: invalid CSV header\n");
                ok = 0;
                break;
            }
            saw_header = 1;
            continue;
        }
        if (count >= MAX_SEED_ROWS || !parse_row(line, &rows[count])) {
            fprintf(stderr, "seed-floor: invalid row at line %zu\n", line_number);
            ok = 0;
            break;
        }
        for (size_t i = 0; i < count; i++) {
            if (rows[i].seed == rows[count].seed) {
                fprintf(stderr, "seed-floor: duplicate seed %" PRIu64 "\n", rows[count].seed);
                ok = 0;
                break;
            }
        }
        if (!ok) break;
        count++;
    }
    if (ferror(file)) ok = 0;
    if (fclose(file) != 0) ok = 0;
    if (!saw_header || count == 0) ok = 0;
    *out_count = count;
    return ok;
}

static int write_artifact(FILE *output, const options_t *options,
                          const stat_seed_floor_t *floor) {
    return fprintf(output,
                   "format_version = 1\n"
                   "metric = validation_cross_entropy_nats\n"
                   "status = %s\n"
                   "sample_count = %zu\n"
                   "minimum_samples = %zu\n"
                   "maximum_samples = %zu\n"
                   "recommended_total = %zu\n"
                   "target_precision_ratio = %.12f\n"
                   "observed_precision_ratio = %.12f\n"
                   "bootstrap_level = %.6f\n"
                   "bootstrap_resamples = %zu\n"
                   "bootstrap_seed = %" PRIu64 "\n"
                   "mean = %.12f\n"
                   "mean_interval_low = %.12f\n"
                   "mean_interval_high = %.12f\n"
                   "median = %.12f\n"
                   "minimum = %.12f\n"
                   "maximum = %.12f\n"
                   "sample_stddev = %.12f\n"
                   "noise_floor = %.12f\n",
                   stat_seed_floor_status_name(floor->status), floor->losses.count,
                   floor->minimum_samples, floor->maximum_samples,
                   floor->recommended_total, floor->target_precision_ratio,
                   floor->precision_ratio, floor->mean_interval.level,
                   floor->mean_interval.resamples, options->bootstrap_seed,
                   floor->losses.mean, floor->mean_interval.low,
                   floor->mean_interval.high, floor->losses.median,
                   floor->losses.minimum, floor->losses.maximum,
                   floor->losses.stddev, floor->noise_floor) >= 0;
}

int main(int argc, char **argv) {
    options_t options;
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return 2;
    }

    seed_loss_t rows[MAX_SEED_ROWS];
    size_t count = 0;
    if (!read_rows(options.input, rows, &count) || count > options.maximum) {
        if (count > options.maximum) {
            fprintf(stderr, "seed-floor: %zu rows exceed declared maximum %zu\n",
                    count, options.maximum);
        }
        return 1;
    }

    double losses[MAX_SEED_ROWS];
    for (size_t i = 0; i < count; i++) losses[i] = rows[i].loss;
    stat_seed_floor_t floor = stat_seed_floor(
        losses, count, options.minimum, options.maximum, options.target_ratio,
        options.resamples, options.bootstrap_seed);
    if (floor.status == STAT_SEED_FLOOR_INVALID) {
        fprintf(stderr, "seed-floor: invalid analysis configuration\n");
        return 1;
    }

    FILE *output = stdout;
    if (options.output) {
        output = fopen(options.output, "w");
        if (!output) {
            fprintf(stderr, "seed-floor: cannot write %s\n", options.output);
            return 1;
        }
    }
    int ok = write_artifact(output, &options, &floor) && fflush(output) == 0;
    if (options.output && fclose(output) != 0) ok = 0;
    return ok ? 0 : 1;
}
