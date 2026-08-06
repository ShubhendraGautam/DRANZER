/*
 * Isolated matrix-multiplication measurements - see tools/bench_matmul.h and
 * docs/matmul.md.
 *
 * Every candidate kernel is timed on the same buffers, in the same process,
 * against shapes taken directly from the model's own call sites, and each is
 * checked against the portable scalar reference before it is timed at all: a
 * faster kernel that computes something else is not a candidate.
 */

#include "tools/bench_matmul.h"
#include "core/cpu_features.h"
#include "core/matmul.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEFAULT_CSV_PATH "matmul_results_v3.csv"
#define MAX_REPEATS 16

typedef struct {
    const char *name;
    size_t m, k, n;
} matmul_case_t;

typedef struct {
    matmul_kernel_t kernel;
    /* Tile edge for the blocked kernels; 0 for kernels that ignore it, which
     * is also what the CSV records so unrelated rows never look tuned. */
    size_t tile;
} candidate_t;

typedef struct {
    size_t iterations;
    size_t repeats;
    double best_us;
    double median_us;
    double gflops;
    double speedup_vs_scalar;
    float max_abs_error;
    float max_relative_error;
    int correct;
} candidate_result_t;

/* Read after every timed region so an optimizing compiler cannot consider
 * the matrix results dead. The kernels live in another translation unit, but
 * keeping an explicit observable consumer also makes that contract survive a
 * future link-time-optimization build. */
static volatile float matmul_sink;

/* Tile candidates: two either side of the 64 the project shipped before any
 * of this was measured, so a better setting is visible as a trend rather
 * than a single lucky point. */
static const size_t sweep_tiles[] = { 16, 32, 64, 128, 256 };

void bench_matmul_default_options(bench_matmul_options_t *options) {
    if (!options) return;
    options->sweep = 0;
    options->quick = 0;
    options->repeats = 3;
    options->csv_path = DEFAULT_CSV_PATH;
}

/* Every blocked kernel gets the full tile sweep. The SIMD kernels are only
 * added where they can actually execute: on a machine without AVX-512,
 * matmul_run() would run tiled_mr4 in their place and the row would be a
 * duplicate measurement under a misleading name. */
static size_t add_tiled_candidates(matmul_kernel_t kernel, candidate_t *out,
                                   size_t count, size_t capacity) {
    if (!matmul_kernel_available(kernel)) return count;
    const size_t tile_count = sizeof(sweep_tiles) / sizeof(sweep_tiles[0]);
    for (size_t i = 0; i < tile_count && count < capacity; i++) {
        out[count++] = (candidate_t){ kernel, sweep_tiles[i] };
    }
    return count;
}

static size_t build_candidates(int sweep, candidate_t *out, size_t capacity) {
    size_t count = 0;

    /* Scalar first, always: every other row's speedup is relative to it. */
    if (count < capacity) out[count++] = (candidate_t){ MATMUL_KERNEL_SCALAR, 0 };
    if (!sweep) {
        if (count < capacity) {
            out[count++] = (candidate_t){ MATMUL_KERNEL_AUTO, matmul_tile_size() };
        }
        return count;
    }

    if (count < capacity) out[count++] = (candidate_t){ MATMUL_KERNEL_ROWWISE, 0 };
    count = add_tiled_candidates(MATMUL_KERNEL_TILED, out, count, capacity);
    count = add_tiled_candidates(MATMUL_KERNEL_TILED_MR4, out, count, capacity);
    count = add_tiled_candidates(MATMUL_KERNEL_AVX2_MR4, out, count, capacity);
    count = add_tiled_candidates(MATMUL_KERNEL_AVX512_MR4, out, count, capacity);
    count = add_tiled_candidates(MATMUL_KERNEL_NEON_MR4, out, count, capacity);
    if (count < capacity) {
        out[count++] = (candidate_t){ MATMUL_KERNEL_AUTO, matmul_tile_size() };
    }
    return count;
}

static int compare_doubles(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

/* One auto-scaled measurement round: grow the iteration count until the
 * timed region is long enough to be meaningful against clock resolution and
 * per-call noise, then report seconds per call. */
static double timed_round(matmul_kernel_t kernel, const float *a, const float *b,
                          float *c, size_t m, size_t k, size_t n,
                          double target_seconds, size_t *iterations_out) {
    const size_t max_iterations = 1U << 20;
    size_t iterations = 1;
    double elapsed = 0.0;

    for (;;) {
        double started = bench_now_sec();
        for (size_t i = 0; i < iterations; i++) {
            matmul_run(kernel, a, b, c, m, k, n);
        }
        elapsed = bench_now_sec() - started;
        matmul_sink += c[(iterations * 131U) % (m * n)];

        if (elapsed >= target_seconds || iterations >= max_iterations) break;
        double multiplier = elapsed > 0.0 ? target_seconds / elapsed : 2.0;
        if (multiplier < 2.0) multiplier = 2.0;
        size_t next = (size_t)((double)iterations * multiplier);
        if (next <= iterations) next = iterations + 1;
        iterations = next > max_iterations ? max_iterations : next;
    }

    *iterations_out = iterations;
    return elapsed / (double)iterations;
}

/* Check one candidate against the scalar reference. A kernel that computes
 * something else is not a candidate, so this runs before any timing. */
static void check_candidate(const candidate_t *candidate,
                            const matmul_case_t *test_case,
                            const float *a, const float *b,
                            const float *reference, float *output,
                            float reference_magnitude,
                            candidate_result_t *result) {
    memset(result, 0, sizeof(*result));
    if (candidate->tile != 0) matmul_set_tile_size(candidate->tile);

    size_t count = test_case->m * test_case->n;
    matmul_run(candidate->kernel, a, b, output, test_case->m, test_case->k,
               test_case->n);
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(output[i])) {
            result->max_abs_error = INFINITY;
            result->max_relative_error = INFINITY;
            break;
        }
        float abs_error = fabsf(reference[i] - output[i]);
        float relative_error = abs_error / fmaxf(fabsf(reference[i]), 1e-6f);
        if (abs_error > result->max_abs_error) result->max_abs_error = abs_error;
        if (relative_error > result->max_relative_error) {
            result->max_relative_error = relative_error;
        }
    }
    result->correct =
        result->max_abs_error <= 1e-4f * (1.0f + reference_magnitude);
}

/* Summarise one candidate's rounds. The comparison statistic is the fastest
 * round, not the median: contention and frequency scaling can only ever make
 * a round slower, so the minimum is the closest thing to "what this kernel
 * costs when it gets the machine". The median is kept alongside it as a
 * noise indicator - when the two are far apart the session was not quiet
 * enough to retune from, which the sweep script prints and docs/matmul.md
 * tells the reader to check. */
static void summarize_rounds(double *samples, size_t repeats,
                             const matmul_case_t *test_case,
                             candidate_result_t *result) {
    qsort(samples, repeats, sizeof(samples[0]), compare_doubles);
    double best = samples[0];
    double median = samples[repeats / 2];
    result->repeats = repeats;
    result->best_us = best * 1e6;
    result->median_us = median * 1e6;
    double operations = 2.0 * (double)test_case->m * (double)test_case->k *
                        (double)test_case->n;
    result->gflops = best > 0.0 ? operations / best / 1e9 : 0.0;
}

static void csv_log(FILE *csv, const char *timestamp, const char *tier,
                    const matmul_case_t *test_case, const candidate_t *candidate,
                    const char *auto_kernel, const candidate_result_t *result,
                    const bench_metadata_t *metadata) {
    if (!csv) return;
    bench_csv_field(csv, timestamp);
    fputc(',', csv);
    bench_csv_field(csv, tier);
    fputc(',', csv);
    bench_csv_field(csv, test_case->name);
    fprintf(csv, ",%zu,%zu,%zu,", test_case->m, test_case->k, test_case->n);
    bench_csv_field(csv, matmul_kernel_name(candidate->kernel));
    fprintf(csv, ",%zu,", candidate->tile);
    bench_csv_field(csv, auto_kernel);
    fprintf(csv, ",%zu,%zu,%.4f,%.4f,%.6f,%.4f,%.8g,%.8g",
            result->repeats, result->iterations, result->best_us,
            result->median_us, result->gflops, result->speedup_vs_scalar,
            result->max_abs_error, result->max_relative_error);
    bench_csv_metadata(csv, metadata);
}

static int run_case(const matmul_case_t *test_case, const candidate_t *candidates,
                    size_t candidate_count, double target_seconds, size_t repeats,
                    FILE *csv, const char *timestamp, const char *tier,
                    const bench_metadata_t *metadata) {
    if (test_case->m == 0 || test_case->k == 0 || test_case->n == 0 ||
        test_case->k > SIZE_MAX / test_case->m ||
        test_case->n > SIZE_MAX / test_case->k ||
        test_case->n > SIZE_MAX / test_case->m) {
        return -1;
    }
    size_t a_count = test_case->m * test_case->k;
    size_t b_count = test_case->k * test_case->n;
    size_t c_count = test_case->m * test_case->n;
    if (a_count > SIZE_MAX / sizeof(float) ||
        b_count > SIZE_MAX / sizeof(float) ||
        c_count > SIZE_MAX / sizeof(float)) {
        return -1;
    }

    float *a = malloc(a_count * sizeof(float));
    float *b = malloc(b_count * sizeof(float));
    float *reference = malloc(c_count * sizeof(float));
    float *output = malloc(c_count * sizeof(float));
    if (!a || !b || !reference || !output) {
        free(a); free(b); free(reference); free(output);
        return -1;
    }

    /* Deterministic, bounded, and sign-mixed: identical inputs across
     * compilers and runs, with cancellation in the sums so a kernel that
     * reorders accumulation cannot look exact by accident. */
    for (size_t i = 0; i < a_count; i++) {
        a[i] = (float)((int)(i % 101U) - 50) / 51.0f;
    }
    for (size_t i = 0; i < b_count; i++) {
        b[i] = (float)((int)((i * 17U + 13U) % 103U) - 51) / 52.0f;
    }
    matrix_multiply_scalar(a, b, reference, test_case->m, test_case->k,
                           test_case->n);
    float reference_magnitude = 0.0f;
    for (size_t i = 0; i < c_count; i++) {
        float magnitude = fabsf(reference[i]);
        if (magnitude > reference_magnitude) reference_magnitude = magnitude;
    }

    const char *auto_kernel = matmul_kernel_name(
        matmul_select(test_case->m, test_case->k, test_case->n));
    printf("  %s  %zux%zux%zu  (auto selects %s)\n", test_case->name,
           test_case->m, test_case->k, test_case->n, auto_kernel);

    size_t saved_tile = matmul_tile_size();
    int failed = 0;

    candidate_result_t *results = calloc(candidate_count, sizeof(*results));
    double *samples = malloc(candidate_count * MAX_REPEATS * sizeof(double));
    if (!results || !samples) {
        free(results); free(samples);
        free(a); free(b); free(reference); free(output);
        return -1;
    }

    for (size_t i = 0; i < candidate_count; i++) {
        check_candidate(&candidates[i], test_case, a, b, reference, output,
                        reference_magnitude, &results[i]);
        if (!results[i].correct) failed = 1;
    }

    /* Rounds are interleaved across candidates rather than run candidate by
     * candidate: on a machine that is not perfectly idle, a slow window then
     * lands on every candidate instead of penalising whichever one happened
     * to be measured while another process was busy. The first pass is
     * discarded for the same reason a single cold round is untrustworthy -
     * CPU frequency and cache state both need one pass to settle. */
    for (size_t i = 0; i < candidate_count; i++) {
        if (candidates[i].tile != 0) matmul_set_tile_size(candidates[i].tile);
        size_t warmup_iterations = 0;
        (void)timed_round(candidates[i].kernel, a, b, output, test_case->m,
                          test_case->k, test_case->n, target_seconds,
                          &warmup_iterations);
    }
    for (size_t round = 0; round < repeats; round++) {
        for (size_t i = 0; i < candidate_count; i++) {
            if (candidates[i].tile != 0) matmul_set_tile_size(candidates[i].tile);
            samples[i * MAX_REPEATS + round] =
                timed_round(candidates[i].kernel, a, b, output, test_case->m,
                            test_case->k, test_case->n, target_seconds,
                            &results[i].iterations);
        }
    }

    double scalar_us = 0.0;
    double best_us = 0.0;
    size_t best_index = 0;
    for (size_t i = 0; i < candidate_count; i++) {
        summarize_rounds(&samples[i * MAX_REPEATS], repeats, test_case, &results[i]);
        if (i == 0) scalar_us = results[i].best_us;
        results[i].speedup_vs_scalar =
            results[i].best_us > 0.0 ? scalar_us / results[i].best_us : 0.0;
        if (i == 0 || results[i].best_us < best_us) {
            best_us = results[i].best_us;
            best_index = i;
        }

        char label[48];
        if (candidates[i].tile != 0) {
            snprintf(label, sizeof(label), "%s/tile%zu",
                     matmul_kernel_name(candidates[i].kernel), candidates[i].tile);
        } else {
            snprintf(label, sizeof(label), "%s",
                     matmul_kernel_name(candidates[i].kernel));
        }
        printf("    %-20s best %10.3f us  median %10.3f us  %7.2f GFLOP/s  "
               "%6.2fx  error %.3g%s\n",
               label, results[i].best_us, results[i].median_us, results[i].gflops,
               results[i].speedup_vs_scalar, (double)results[i].max_abs_error,
               results[i].correct ? "" : "  DIVERGED");
        csv_log(csv, timestamp, tier, test_case, &candidates[i], auto_kernel,
                &results[i], metadata);
    }

    if (candidates[best_index].tile != 0) {
        printf("    fastest: %s/tile%zu\n",
               matmul_kernel_name(candidates[best_index].kernel),
               candidates[best_index].tile);
    } else {
        printf("    fastest: %s\n",
               matmul_kernel_name(candidates[best_index].kernel));
    }

    matmul_set_tile_size(saved_tile);
    free(results); free(samples);
    free(a); free(b); free(reference); free(output);
    return failed ? 1 : 0;
}

int bench_matmul_run(const bench_matmul_tier_t *tiers, size_t tier_count,
                     const char *selected_tier,
                     const bench_matmul_options_t *options,
                     const char *timestamp, const bench_metadata_t *metadata) {
    if (!tiers || !options || !metadata) return 1;

    const char *csv_path = options->csv_path ? options->csv_path : DEFAULT_CSV_PATH;
    int csv_existed = (access(csv_path, F_OK) == 0);
    FILE *csv = fopen(csv_path, "a");
    if (!csv) {
        fprintf(stderr, "Error: cannot open %s\n", csv_path);
        return 1;
    }
    if (!csv_existed) {
        fprintf(csv, "timestamp,tier,case,m,k,n,kernel,tile,auto_kernel,repeats,"
                     "iterations,best_us,median_us,gflops,speedup_vs_scalar,"
                     "max_abs_error,max_relative_error," BENCH_METADATA_CSV_HEADER "\n");
    }

    /* scalar + rowwise + auto, plus a full tile sweep for each of the five
     * blocked kernels (tiled, tiled_mr4, and the three SIMD ones). Sized for
     * all of them even though only this architecture's SIMD kernels are ever
     * added, so the bound never depends on the build target. */
    candidate_t candidates[3 + 5 * (sizeof(sweep_tiles) / sizeof(sweep_tiles[0]))];
    size_t candidate_count = build_candidates(options->sweep, candidates,
                                              sizeof(candidates) / sizeof(candidates[0]));
    double target_seconds = options->quick ? 0.01 : 0.15;
    size_t repeats = options->quick ? 1 : options->repeats;
    if (repeats == 0) repeats = 1;
    if (repeats > MAX_REPEATS) repeats = MAX_REPEATS;

    printf("Isolated matmul comparison: %zu candidate kernels, %zu interleaved "
           "timing round(s) each, %.0f ms per round.\n",
           candidate_count, repeats, target_seconds * 1000.0);
    printf("Candidates are ranked by their fastest round; the median is "
           "reported beside it as a noise indicator.\n");
    printf("Default kernel: %s (resolves to %s here), default tile: %zu.\n",
           matmul_kernel_name(matmul_get_kernel()),
           matmul_kernel_name(matmul_select(1, 64, 64)), matmul_tile_size());
    printf("Instruction set: %s.\n\n", cpu_features_summary());

    int failed = 0;
    for (size_t i = 0; i < tier_count; i++) {
        const bench_matmul_tier_t *tier = &tiers[i];
        if (selected_tier && strcmp(selected_tier, tier->key) != 0) continue;
        size_t embedding_dim = tier->embedding_dim;
        size_t sequence = tier->max_seq_len;
        size_t ffn_dim = embedding_dim * 4U;

        /* Every shape below is a live call site in core/transformer.c: the
         * decode rows come from model_forward_token(), the prefill rows from
         * model_forward()'s attention/FFN projections, and the training row
         * from the FFN down-projection that dominates the backward pass. */
        matmul_case_t cases[] = {
            { "decode_attention_projection", 1, embedding_dim, embedding_dim },
            { "decode_ffn_up",               1, embedding_dim, ffn_dim },
            { "decode_output_head",          1, embedding_dim, tier->vocab_size },
            { "prefill_attention_projection", sequence, embedding_dim, embedding_dim },
            { "prefill_ffn_up",              sequence, embedding_dim, ffn_dim },
            { "training_ffn_down",           sequence, ffn_dim, embedding_dim },
        };
        size_t case_count = sizeof(cases) / sizeof(cases[0]);

        printf("--- isolated matmul: %s ---\n", tier->name);
        for (size_t j = 0; j < case_count; j++) {
            int rc = run_case(&cases[j], candidates, candidate_count,
                              target_seconds, repeats, csv, timestamp,
                              tier->key, metadata);
            if (rc != 0) failed = 1;
        }
        printf("\n");
    }

    printf("Matmul results appended to %s\n", csv_path);
    fclose(csv);
    return failed;
}
