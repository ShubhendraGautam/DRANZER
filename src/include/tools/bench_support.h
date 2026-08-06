#ifndef BENCH_SUPPORT_H
#define BENCH_SUPPORT_H

#include <stddef.h>
#include <stdio.h>

/* Shared plumbing for the standalone benchmark binaries: the build/host
 * metadata every result row must carry, a monotonic clock, and CSV quoting.
 * Kept apart from any particular measurement so the full-model benchmark
 * (tools/bench.c) and the isolated matmul sweep (tools/bench_matmul.c) can
 * emit identical provenance columns without either depending on the other. */

typedef struct {
    char compiler[192];
    char os[256];
    char cpu[256];
    /* Which SIMD instruction sets this run was allowed to use, from
     * cpu_features_summary(). Load-bearing since runtime kernel dispatch:
     * the same executable picks a different matmul kernel on different
     * hardware, so "which CPU" no longer implies "which code ran". */
    char simd[160];
    const char *build_command;
    long online_cpus;
    long openmp_version;
    int max_threads;
} bench_metadata_t;

/* Compiler, OS, CPU model, core count, and OpenMP configuration of the
 * running binary. `build_command` comes from -DDRANZER_BUILD_COMMAND, which
 * the Makefile sets to the exact compile/link line. */
void bench_collect_metadata(bench_metadata_t *metadata);

/* Human-readable summary of the above, printed at the top of every run. */
void bench_print_metadata(const bench_metadata_t *metadata);

/* CLOCK_MONOTONIC seconds. */
double bench_now_sec(void);

/* Peak resident set size in KB. */
long bench_peak_rss_kb(void);

/* Write one RFC 4180-quoted CSV field (no separator). */
void bench_csv_field(FILE *csv, const char *value);

/* The trailing provenance columns shared by every results file. The header
 * text and the field order below are kept in lockstep on purpose. */
#define BENCH_METADATA_CSV_HEADER \
    "build_command,compiler,os,cpu,simd,online_cpus,openmp_version,max_threads"

/* Append the provenance columns, preceded by a separating comma, and
 * terminate the row. */
void bench_csv_metadata(FILE *csv, const bench_metadata_t *metadata);

#endif // BENCH_SUPPORT_H
