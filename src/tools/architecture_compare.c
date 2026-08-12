/* Analyze paired baseline/feature held-out losses against a measured seed
 * floor. Collection and provenance live in the research runner; this tool is
 * deliberately only strict CSV parsing plus the shared statistical verdict. */

#include "common/fp_bits.h"
#include "tools/statistics.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PAIRS 1024
#define LINE_CAPACITY 4096

typedef struct {
    const char *input;
    const char *output;
    double noise_floor;
    size_t resamples;
    uint64_t bootstrap_seed;
} options_t;

typedef struct {
    uint64_t seed;
    double baseline_loss;
    double feature_loss;
} pair_t;

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s INPUT.csv --noise-floor N [--output FILE]\n"
            "       [--resamples N] [--bootstrap-seed N]\n\n"
            "CSV header: seed,baseline_validation_cross_entropy,"
            "feature_validation_cross_entropy,baseline_model_sha256,"
            "feature_model_sha256\n",
            program);
}

static int parse_size(const char *text, size_t minimum, size_t maximum,
                      size_t *out) {
    if (!text || !text[0] || !out || text[0] == '-') return 0;
    for (const unsigned char *cursor = (const unsigned char *)text; *cursor;
         cursor++) {
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
    for (const unsigned char *cursor = (const unsigned char *)text; *cursor;
         cursor++) {
        if (!isdigit(*cursor)) return 0;
    }
    errno = 0;
    char *end = NULL;
    uintmax_t value = strtoumax(text, &end, 10);
    if (errno || !end || *end || value > UINT64_MAX) return 0;
    *out = (uint64_t)value;
    return 1;
}

static int parse_nonnegative_double(const char *text, double *out) {
    if (!text || !text[0] || !out) return 0;
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno || !end || *end || value < 0.0 ||
        !dranzer_double_is_finite(value)) {
        return 0;
    }
    *out = value;
    return 1;
}

static int parse_options(int argc, char **argv, options_t *options) {
    if (!options || argc < 2) return 0;
    *options = (options_t){
        .input = argv[1],
        .resamples = 10000,
        .bootstrap_seed = UINT64_C(0x41524348434f4d50),
    };
    int have_noise_floor = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--noise-floor") == 0 && i + 1 < argc) {
            if (!parse_nonnegative_double(argv[++i], &options->noise_floor))
                return 0;
            have_noise_floor = 1;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            options->output = argv[++i];
        } else if (strcmp(argv[i], "--resamples") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], 1, 10000000, &options->resamples))
                return 0;
        } else if (strcmp(argv[i], "--bootstrap-seed") == 0 &&
                   i + 1 < argc) {
            if (!parse_u64(argv[++i], &options->bootstrap_seed)) return 0;
        } else {
            return 0;
        }
    }
    return have_noise_floor && (!options->output || options->output[0]);
}

static int valid_sha256(const char *text) {
    if (!text || strlen(text) != 64) return 0;
    for (size_t i = 0; i < 64; i++) {
        if (!isxdigit((unsigned char)text[i])) return 0;
    }
    return 1;
}

static int parse_row(char *line, pair_t *pair) {
    char *fields[5];
    fields[0] = line;
    for (size_t i = 1; i < 5; i++) {
        fields[i] = strchr(fields[i - 1], ',');
        if (!fields[i]) return 0;
        *fields[i]++ = '\0';
    }
    if (strchr(fields[4], ',')) return 0;
    size_t length = strlen(fields[4]);
    while (length > 0 &&
           (fields[4][length - 1] == '\n' || fields[4][length - 1] == '\r')) {
        fields[4][--length] = '\0';
    }
    return parse_u64(fields[0], &pair->seed) &&
           parse_nonnegative_double(fields[1], &pair->baseline_loss) &&
           parse_nonnegative_double(fields[2], &pair->feature_loss) &&
           valid_sha256(fields[3]) && valid_sha256(fields[4]);
}

static int read_pairs(const char *path, pair_t *pairs, size_t *out_count) {
    static const char expected_header[] =
        "seed,baseline_validation_cross_entropy,feature_validation_cross_entropy,"
        "baseline_model_sha256,feature_model_sha256";
    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "architecture-compare: cannot open %s\n", path);
        return 0;
    }
    char line[LINE_CAPACITY];
    size_t count = 0, line_number = 0;
    int ok = 1, saw_header = 0;
    while (fgets(line, sizeof(line), file)) {
        line_number++;
        if (!strchr(line, '\n') && !feof(file)) {
            fprintf(stderr, "architecture-compare: line %zu is too long\n",
                    line_number);
            ok = 0;
            break;
        }
        if (line[0] == '#') continue;
        if (!saw_header) {
            size_t length = strlen(line);
            while (length > 0 &&
                   (line[length - 1] == '\n' || line[length - 1] == '\r')) {
                line[--length] = '\0';
            }
            if (strcmp(line, expected_header) != 0) {
                fprintf(stderr, "architecture-compare: invalid CSV header\n");
                ok = 0;
                break;
            }
            saw_header = 1;
            continue;
        }
        if (count >= MAX_PAIRS || !parse_row(line, &pairs[count])) {
            fprintf(stderr, "architecture-compare: invalid row at line %zu\n",
                    line_number);
            ok = 0;
            break;
        }
        for (size_t i = 0; i < count; i++) {
            if (pairs[i].seed == pairs[count].seed) {
                fprintf(stderr, "architecture-compare: duplicate seed %" PRIu64
                                "\n", pairs[count].seed);
                ok = 0;
                break;
            }
        }
        if (!ok) break;
        count++;
    }
    if (ferror(file) || fclose(file) != 0) ok = 0;
    if (!saw_header || count < 2) ok = 0;
    *out_count = count;
    return ok;
}

static int write_result(FILE *output, const options_t *options,
                        const stat_paired_t *comparison) {
    size_t recommended = stat_pairs_needed(comparison);
    const char *direction = comparison->verdict == STAT_RESOLVED_B_SMALLER
                                ? "feature_better"
                            : comparison->verdict == STAT_RESOLVED_B_GREATER
                                ? "feature_worse"
                                : "unresolved";
    int ok = fprintf(output,
                   "format_version = 1\n"
                   "metric = validation_cross_entropy_nats\n"
                   "verdict = %s\n"
                   "direction = %s\n"
                   "pair_count = %zu\n"
                   "pairs_feature_loss_greater = %zu\n"
                   "noise_floor = %.12f\n"
                   "bootstrap_level = %.6f\n"
                   "bootstrap_resamples = %zu\n"
                   "bootstrap_seed = %" PRIu64 "\n"
                   "mean_feature_minus_baseline = %.12f\n"
                   "difference_interval_low = %.12f\n"
                   "difference_interval_high = %.12f\n"
                   "mean_feature_over_baseline = %.12f\n"
                   "ratio_interval_low = %.12f\n"
                   "ratio_interval_high = %.12f\n",
                   stat_verdict_name(comparison->verdict), direction,
                   comparison->pairs, comparison->pairs_favouring_b,
                   comparison->noise_floor, comparison->difference.level,
                   comparison->difference.resamples, options->bootstrap_seed,
                   comparison->difference.point, comparison->difference.low,
                   comparison->difference.high, comparison->ratio.point,
                   comparison->ratio.low, comparison->ratio.high) >= 0;
    if (!ok) return 0;
    if (recommended == SIZE_MAX) {
        return fprintf(output, "recommended_total = unbounded\n") >= 0;
    }
    return fprintf(output, "recommended_total = %zu\n", recommended) >= 0;
}

int main(int argc, char **argv) {
    options_t options;
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return 2;
    }
    pair_t pairs[MAX_PAIRS];
    size_t count = 0;
    if (!read_pairs(options.input, pairs, &count)) return 1;
    double baseline[MAX_PAIRS], feature[MAX_PAIRS];
    for (size_t i = 0; i < count; i++) {
        baseline[i] = pairs[i].baseline_loss;
        feature[i] = pairs[i].feature_loss;
    }
    stat_paired_t comparison = stat_paired_compare(
        baseline, feature, count, options.noise_floor, options.resamples,
        options.bootstrap_seed);
    FILE *output = stdout;
    if (options.output) {
        output = fopen(options.output, "w");
        if (!output) {
            fprintf(stderr, "architecture-compare: cannot write %s\n",
                    options.output);
            return 1;
        }
    }
    int ok = write_result(output, &options, &comparison);
    if (options.output && fclose(output) != 0) ok = 0;
    return ok ? 0 : 1;
}
