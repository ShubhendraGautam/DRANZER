/* Measure the two version-1 bundle loading strategies in separate process
 * runs so peak RSS is attributable to one strategy. The loader still checks
 * every payload byte; this measures allocation/initialization/copy overhead,
 * not an unsafe checksum-skipping fast path. */

#include "byte_pair_encoding.h"
#include "core/bundle.h"
#include "core/model.h"
#include "tools/bench_support.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define DEFAULT_REPEATS 7
#define DEFAULT_CSV_PATH "bundle_load_results_v1.csv"

typedef enum { LOAD_COPY, LOAD_MMAP } load_mode_t;
static volatile float result_sink;

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static int parse_repeats(const char *text, size_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value > 1000) return 0;
    *out = (size_t)value;
    return 1;
}

static void release_load(neural_model_t *model, bpe_encoder_t *encoder) {
    model_free(model);
    if (encoder) {
        bpe_encoder_free(encoder);
        free(encoder);
    }
}

static int append_csv(const char *path, const char *bundle_path,
                      const char *mode, uint64_t artifact_bytes,
                      size_t parameter_count, size_t repeats,
                      double median_ms, long peak_rss_kb,
                      const bench_metadata_t *metadata) {
    FILE *csv = fopen(path, "a+");
    if (!csv) return 0;
    if (fseek(csv, 0, SEEK_END) != 0) {
        fclose(csv);
        return 0;
    }
    long size = ftell(csv);
    if (size < 0) {
        fclose(csv);
        return 0;
    }
    if (size == 0) {
        fprintf(csv, "bundle,mode,artifact_bytes,parameter_count,repeats,"
                     "median_startup_ms,peak_rss_kb,%s\n",
                BENCH_METADATA_CSV_HEADER);
    }
    bench_csv_field(csv, bundle_path);
    fputc(',', csv);
    bench_csv_field(csv, mode);
    fprintf(csv, ",%llu,%zu,%zu,%.6f,%ld",
            (unsigned long long)artifact_bytes, parameter_count, repeats,
            median_ms, peak_rss_kb);
    bench_csv_metadata(csv, metadata);
    return fclose(csv) == 0;
}

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s BUNDLE --mode copy|mmap [--repeats N] [--csv PATH]\n",
            program);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 2;
    }
    const char *bundle_path = argv[1];
    const char *csv_path = DEFAULT_CSV_PATH;
    const char *mode_name = NULL;
    load_mode_t mode = LOAD_COPY;
    size_t repeats = DEFAULT_REPEATS;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode_name = argv[++i];
            if (strcmp(mode_name, "copy") == 0) mode = LOAD_COPY;
            else if (strcmp(mode_name, "mmap") == 0) mode = LOAD_MMAP;
            else {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            if (!parse_repeats(argv[++i], &repeats)) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!mode_name) {
        usage(argv[0]);
        return 2;
    }

    struct stat status;
    if (stat(bundle_path, &status) != 0 || status.st_size < 0) {
        fprintf(stderr, "Could not stat bundle: %s\n", bundle_path);
        return 1;
    }
    double *samples = malloc(repeats * sizeof(*samples));
    if (!samples) return 1;
    size_t parameter_count = 0;
    for (size_t repeat = 0; repeat < repeats; repeat++) {
        neural_model_t model = {0};
        bpe_encoder_t *encoder = NULL;
        model_bundle_metadata_t bundle_metadata = {0};
        double started = bench_now_sec();
        bundle_errors_t rc = mode == LOAD_MMAP
            ? model_bundle_load_mmap(&model, &encoder, &bundle_metadata,
                                     bundle_path)
            : model_bundle_load(&model, &encoder, &bundle_metadata,
                                bundle_path);
        samples[repeat] = bench_now_sec() - started;
        if (rc != BUNDLE_SUCCESS) {
            fprintf(stderr, "%s load failed with bundle status %d\n",
                    mode_name, (int)rc);
            release_load(&model, encoder);
            free(samples);
            return 1;
        }
        parameter_count = model.total_param_count;
        size_t page_stride = 4096 / sizeof(float);
        for (size_t i = 0; i < model.total_param_count; i += page_stride) {
            result_sink += model.params[i];
        }
        release_load(&model, encoder);
    }
    qsort(samples, repeats, sizeof(*samples), compare_double);
    double median_ms = samples[repeats / 2] * 1000.0;
    if ((repeats & 1u) == 0) {
        median_ms = (samples[repeats / 2 - 1] + samples[repeats / 2]) * 500.0;
    }
    free(samples);

    bench_metadata_t metadata;
    bench_collect_metadata(&metadata);
    bench_print_metadata(&metadata);
    long peak_rss_kb = bench_peak_rss_kb();
    printf("Bundle: %s\nMode: %s\nArtifact: %lld bytes\n"
           "Parameters: %zu\nRepeats: %zu\nMedian startup: %.3f ms\n"
           "Peak RSS: %ld KB\n",
           bundle_path, mode_name, (long long)status.st_size, parameter_count,
           repeats, median_ms, peak_rss_kb);
    if (!append_csv(csv_path, bundle_path, mode_name,
                    (uint64_t)status.st_size, parameter_count, repeats,
                    median_ms, peak_rss_kb, &metadata)) {
        fprintf(stderr, "Could not append benchmark CSV: %s\n", csv_path);
        return 1;
    }
    printf("CSV: %s\n", csv_path);
    return 0;
}
