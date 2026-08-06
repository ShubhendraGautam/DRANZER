#ifndef BENCH_MATMUL_H
#define BENCH_MATMUL_H

#include "tools/bench_support.h"
#include <stddef.h>

/* Isolated matrix-multiplication measurements: the evidence the shipped CPU
 * kernel and tile size are selected from (docs/matmul.md).
 *
 * Deliberately separate from the full-model benchmark in tools/bench.c -
 * this one builds no model, allocates nothing but three matrices, and exists
 * to compare interchangeable kernels on the exact shapes the model issues. */

typedef struct {
    const char *key;   /* tier key, e.g. "small" - also the CSV tier column */
    const char *name;  /* human-readable tier name */
    size_t vocab_size; /* output-head width */
    size_t embedding_dim;
    size_t max_seq_len;
} bench_matmul_tier_t;

typedef struct {
    /* Measure every kernel and tile candidate rather than only the portable
     * reference against the shipped default. */
    int sweep;
    /* Shorten each measurement round; for smoke tests, not for selection. */
    int quick;
    /* Independent measurement rounds per candidate. The reported time is the
     * median, so a single scheduling hiccup cannot promote a candidate, and
     * the spread between best and median shows whether a run was quiet
     * enough to draw conclusions from. */
    size_t repeats;
    /* Results file to append to. NULL uses the default path. */
    const char *csv_path;
} bench_matmul_options_t;

/* Fill `options` with the documented defaults (no sweep, 3 repeats,
 * matmul_results_v3.csv). */
void bench_matmul_default_options(bench_matmul_options_t *options);

/* Measure every tier (or only `selected_tier`, when non-NULL), print a
 * per-shape table, and append one CSV row per candidate.
 * Returns 0 when every candidate stayed within tolerance of the scalar
 * reference, 1 when any diverged or the results file could not be opened. */
int bench_matmul_run(const bench_matmul_tier_t *tiers, size_t tier_count,
                     const char *selected_tier,
                     const bench_matmul_options_t *options,
                     const char *timestamp, const bench_metadata_t *metadata);

#endif // BENCH_MATMUL_H
