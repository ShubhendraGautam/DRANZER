/*
 * Build/host provenance, timing, and CSV helpers shared by the benchmark
 * binaries. See tools/bench_support.h.
 */

#include "tools/bench_support.h"
#include <string.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef DRANZER_BUILD_COMMAND
#define DRANZER_BUILD_COMMAND "unknown (built outside the project Makefile)"
#endif

static void trim_line(char *text) {
    size_t length = strlen(text);
    while (length > 0 &&
           (text[length - 1] == '\n' || text[length - 1] == '\r' ||
            text[length - 1] == ' ' || text[length - 1] == '\t')) {
        text[--length] = '\0';
    }
}

static void read_cpu_name(char *output, size_t capacity) {
    snprintf(output, capacity, "unknown");
    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
    if (!cpuinfo) return;
    char line[512];
    while (fgets(line, sizeof(line), cpuinfo)) {
        if (strncmp(line, "model name", 10) != 0 &&
            strncmp(line, "Hardware", 8) != 0) continue;
        char *value = strchr(line, ':');
        if (!value) continue;
        value++;
        while (*value == ' ' || *value == '\t') value++;
        trim_line(value);
        snprintf(output, capacity, "%s", value);
        break;
    }
    fclose(cpuinfo);
}

void bench_collect_metadata(bench_metadata_t *metadata) {
    if (!metadata) return;
    memset(metadata, 0, sizeof(*metadata));
#if defined(__clang__)
    snprintf(metadata->compiler, sizeof(metadata->compiler),
             "clang %s", __clang_version__);
#elif defined(__GNUC__)
    snprintf(metadata->compiler, sizeof(metadata->compiler),
             "gcc %s", __VERSION__);
#else
    snprintf(metadata->compiler, sizeof(metadata->compiler), "unknown C compiler");
#endif
    struct utsname system;
    memset(&system, 0, sizeof(system));
    if (uname(&system) == 0) {
        snprintf(metadata->os, sizeof(metadata->os), "%s %s %s",
                 system.sysname, system.release, system.machine);
    } else {
        snprintf(metadata->os, sizeof(metadata->os), "unknown");
    }
    read_cpu_name(metadata->cpu, sizeof(metadata->cpu));
    metadata->build_command = DRANZER_BUILD_COMMAND;
    metadata->online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
#ifdef _OPENMP
    metadata->openmp_version = _OPENMP;
    metadata->max_threads = omp_get_max_threads();
#else
    metadata->openmp_version = 0;
    metadata->max_threads = 1;
#endif
}

void bench_print_metadata(const bench_metadata_t *metadata) {
    if (!metadata) return;
    printf("Build: %s\nCompiler: %s\nSystem: %s\nCPU: %s\n",
           metadata->build_command, metadata->compiler, metadata->os,
           metadata->cpu);
    printf("Threads: online=%ld OpenMP=%ld max=%d\n\n",
           metadata->online_cpus, metadata->openmp_version,
           metadata->max_threads);
}

double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

long bench_peak_rss_kb(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; /* KB on Linux */
}

void bench_csv_field(FILE *csv, const char *value) {
    if (!csv) return;
    fputc('"', csv);
    for (const char *cursor = value ? value : ""; *cursor; cursor++) {
        if (*cursor == '"') fputc('"', csv);
        fputc(*cursor, csv);
    }
    fputc('"', csv);
}

void bench_csv_metadata(FILE *csv, const bench_metadata_t *metadata) {
    if (!csv || !metadata) return;
    fputc(',', csv);
    bench_csv_field(csv, metadata->build_command);
    fputc(',', csv);
    bench_csv_field(csv, metadata->compiler);
    fputc(',', csv);
    bench_csv_field(csv, metadata->os);
    fputc(',', csv);
    bench_csv_field(csv, metadata->cpu);
    fprintf(csv, ",%ld,%ld,%d\n", metadata->online_cpus,
            metadata->openmp_version, metadata->max_threads);
    fflush(csv);
}
