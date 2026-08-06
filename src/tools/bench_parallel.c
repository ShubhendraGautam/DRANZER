/*
 * What an OpenMP parallel region costs, and where it starts paying for itself.
 *
 * This is the tool behind docs/threading.md. It answers three questions that
 * the whole-model benchmark cannot separate:
 *
 *   1. How long does entering a region take, with and without OpenMP's own
 *      `if` clause? The clause turns out not to skip the cost.
 *   2. Would a persistent worker pool - threads that never sleep, dispatched
 *      through an atomic counter - charge less? The prototype below exists
 *      only to answer that, and is not part of the model.
 *   3. At what amount of work does forking start winning? That number is what
 *      DRANZER_PARALLEL_MIN_WORK in core/parallel.c is set from, and it is
 *      measured here by timing the same kernel on the same buffers with
 *      forking forced on and forced off.
 *
 * A build without OpenMP has nothing to measure and says so.
 */

#include "tools/bench_support.h"
#include "core/cpu_features.h"
#include "core/matmul.h"
#include "core/parallel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#endif

#define DEFAULT_CSV_PATH "parallel_results_v1.csv"

/* Read after every timed region so the compiler cannot delete the work. */
static volatile double sink;
static double touch[64];

typedef struct {
    const char *tier;
    const char *name;
    size_t m, k, n;
} shape_t;

/* The same call sites tools/bench_matmul.c measures, at all three tiers, so
 * the threshold is chosen against the shapes the model actually issues rather
 * than against round numbers. */
static const shape_t shapes[] = {
    { "tiny",   "decode_attention_projection",   1,   16,   16 },
    { "tiny",   "decode_ffn_up",                 1,   16,   64 },
    { "tiny",   "decode_output_head",            1,   16,  260 },
    { "tiny",   "prefill_attention_projection", 32,   16,   16 },
    { "tiny",   "prefill_ffn_up",               32,   16,   64 },
    { "tiny",   "training_ffn_down",            32,   64,   16 },
    { "small",  "decode_attention_projection",   1,   64,   64 },
    { "small",  "decode_ffn_up",                 1,   64,  256 },
    { "small",  "decode_output_head",            1,   64, 1000 },
    { "small",  "prefill_attention_projection", 64,   64,   64 },
    { "small",  "prefill_ffn_up",               64,   64,  256 },
    { "small",  "training_ffn_down",            64,  256,   64 },
    { "medium", "decode_attention_projection",   1,  256,  256 },
    { "medium", "decode_ffn_up",                 1,  256, 1024 },
    { "medium", "decode_output_head",            1,  256, 4000 },
    { "medium", "prefill_attention_projection",128,  256,  256 },
    { "medium", "prefill_ffn_up",              128,  256, 1024 },
    { "medium", "training_ffn_down",           128, 1024,  256 },
};
static const size_t shape_count = sizeof(shapes) / sizeof(shapes[0]);

static double now_sec(void) { return bench_now_sec(); }

/* ------------------------------------------------- persistent worker pool ---
 *
 * A measurement prototype, not a component. It is the cheapest dispatch this
 * project could plausibly build - workers spin on an atomic generation counter
 * and never sleep, so a dispatch is one release store plus one acquire poll,
 * with no runtime, no scheduling policy, and no dynamic work distribution.
 * That makes it a lower bound on what any persistent worker strategy could
 * cost, which is exactly what is needed to decide whether to build a real one.
 */
#ifdef _OPENMP

#define POOL_MAX_WORKERS 64

typedef void (*pool_task_fn)(size_t begin, size_t end);

static _Atomic unsigned long pool_generation;
static _Atomic int pool_outstanding;
static _Atomic int pool_stopping;
static pool_task_fn pool_task;
static size_t pool_count;
static int pool_workers = 1;
static pthread_t pool_threads[POOL_MAX_WORKERS];

static void pool_slice(int id, size_t count, size_t *begin, size_t *end) {
    size_t per = (count + (size_t)pool_workers - 1) / (size_t)pool_workers;
    size_t b = per * (size_t)id;
    if (b > count) b = count;
    size_t e = b + per;
    if (e > count) e = count;
    *begin = b;
    *end = e;
}

static void *pool_worker(void *arg) {
    int id = (int)(intptr_t)arg;
    unsigned long seen = 0;
    for (;;) {
        while (atomic_load_explicit(&pool_generation, memory_order_acquire) == seen) {
            if (atomic_load_explicit(&pool_stopping, memory_order_relaxed)) return NULL;
        }
        seen = atomic_load_explicit(&pool_generation, memory_order_acquire);
        if (atomic_load_explicit(&pool_stopping, memory_order_relaxed)) return NULL;
        size_t begin, end;
        pool_slice(id, pool_count, &begin, &end);
        pool_task(begin, end);
        atomic_fetch_sub_explicit(&pool_outstanding, 1, memory_order_release);
    }
}

static void pool_start(int threads) {
    if (threads > POOL_MAX_WORKERS) threads = POOL_MAX_WORKERS;
    pool_workers = threads;
    for (int i = 1; i < threads; i++) {
        pthread_create(&pool_threads[i], NULL, pool_worker, (void *)(intptr_t)i);
    }
}

static void pool_stop(void) {
    atomic_store(&pool_stopping, 1);
    atomic_fetch_add(&pool_generation, 1);
    for (int i = 1; i < pool_workers; i++) pthread_join(pool_threads[i], NULL);
}

static void pool_dispatch(pool_task_fn fn, size_t count) {
    pool_task = fn;
    pool_count = count;
    atomic_store_explicit(&pool_outstanding, pool_workers - 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool_generation, 1, memory_order_release);
    size_t begin, end;
    pool_slice(0, count, &begin, &end);
    fn(begin, end);
    while (atomic_load_explicit(&pool_outstanding, memory_order_acquire) != 0) {
        /* spin */
    }
}

static void touch_range(size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) touch[i] += 1.0;
}

#endif /* _OPENMP */

/* ------------------------------------------------------------ entry cost ---
 *
 * Every variant does the same negligible work over the same eight elements, so
 * the difference between them is the dispatch mechanism and nothing else.
 */
static double best_of(double (*measure)(size_t), size_t reps, int rounds) {
    double best = 1e18;
    measure(reps); /* warm up: first entry creates the team */
    for (int r = 0; r < rounds; r++) {
        double us = measure(reps);
        if (us < best) best = us;
    }
    sink = touch[0];
    return best;
}

static double time_no_pragma(size_t reps) {
    double start = now_sec();
    for (size_t r = 0; r < reps; r++) {
        for (size_t i = 0; i < 8; i++) touch[i] += 1.0;
    }
    return (now_sec() - start) * 1e6 / (double)reps;
}

#ifdef _OPENMP
static double time_region(size_t reps) {
    double start = now_sec();
    for (size_t r = 0; r < reps; r++) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < 8; i++) touch[i] += 1.0;
    }
    return (now_sec() - start) * 1e6 / (double)reps;
}

static double time_region_if_false(size_t reps) {
    double start = now_sec();
    for (size_t r = 0; r < reps; r++) {
        #pragma omp parallel for schedule(static) if(0)
        for (size_t i = 0; i < 8; i++) touch[i] += 1.0;
    }
    return (now_sec() - start) * 1e6 / (double)reps;
}

static double time_pool(size_t reps) {
    double start = now_sec();
    for (size_t r = 0; r < reps; r++) pool_dispatch(touch_range, 8);
    return (now_sec() - start) * 1e6 / (double)reps;
}
#endif

/* ------------------------------------------------------- crossover sweep ---
 *
 * One shape, timed twice in the same process on the same buffers: once with
 * the cutoff set so every region is entered, once with it set so none is.
 * Everything else - kernel, tile, data, cache state - is held constant, so the
 * ratio is the value of forking at that shape and nothing else.
 */
static double time_shape(const shape_t *shape, const float *a, const float *b,
                         float *c, size_t min_work, int rounds) {
    parallel_set_min_work(min_work);

    size_t iterations = 1;
    double elapsed = 0.0;
    for (;;) {
        double start = now_sec();
        for (size_t i = 0; i < iterations; i++) {
            matmul_run(MATMUL_KERNEL_AUTO, a, b, c, shape->m, shape->k, shape->n);
        }
        elapsed = now_sec() - start;
        if (elapsed >= 0.05 || iterations >= (1u << 22)) break;
        iterations *= 4;
    }

    double best = elapsed * 1e6 / (double)iterations;
    for (int r = 1; r < rounds; r++) {
        double start = now_sec();
        for (size_t i = 0; i < iterations; i++) {
            matmul_run(MATMUL_KERNEL_AUTO, a, b, c, shape->m, shape->k, shape->n);
        }
        double us = (now_sec() - start) * 1e6 / (double)iterations;
        if (us < best) best = us;
    }
    sink = c[0];
    return best;
}

static void usage(const char *program) {
    printf("Usage: %s [--rounds N] [--csv-path FILE]\n\n"
           "  --rounds N     timing rounds per measurement (default 5, best wins)\n"
           "  --csv-path F   results file to append to (default %s)\n",
           program, DEFAULT_CSV_PATH);
}

int main(int argc, char **argv) {
    int rounds = 5;
    const char *csv_path = DEFAULT_CSV_PATH;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
            rounds = atoi(argv[++i]);
            if (rounds < 1) {
                fprintf(stderr, "Error: --rounds must be at least 1\n");
                return 1;
            }
            continue;
        }
        if (strcmp(argv[i], "--csv-path") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
            continue;
        }
        fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
        usage(argv[0]);
        return 1;
    }

    cpu_features_detect();
    bench_metadata_t metadata;
    bench_collect_metadata(&metadata);
    bench_print_metadata(&metadata);

#ifndef _OPENMP
    printf("Built without OpenMP, so there are no parallel regions to measure.\n"
           "Rebuild with `make bench-parallel OMP=1 CC=gcc` (clang needs libomp-dev).\n");
    (void)csv_path;
    (void)rounds;
    (void)shapes;
    (void)shape_count;
    (void)time_no_pragma;
    (void)best_of;
    (void)time_shape;
    return 0;
#else
    const int threads = omp_get_max_threads();
    printf("CPU matmul kernel: %s (tile %zu), cutoff %zu\n\n",
           matmul_kernel_name(matmul_select(1, 1, 1)), matmul_tile_size(),
           parallel_min_work());

    pool_start(threads);

    const size_t entry_reps = 20000;
    double plain = best_of(time_no_pragma, entry_reps, rounds);
    double region = best_of(time_region, entry_reps, rounds);
    double region_if = best_of(time_region_if_false, entry_reps, rounds);
    double pooled = best_of(time_pool, entry_reps, rounds);

    printf("Dispatch cost for a loop that does no work (%d threads, best of %d):\n",
           threads, rounds);
    printf("  no pragma at all                    %8.4f us\n", plain);
    printf("  omp parallel for                    %8.4f us   %6.1fx a plain loop\n",
           region, region / plain);
    printf("  omp parallel for if(0)              %8.4f us   %6.0f%% of a full region\n",
           region_if, 100.0 * region_if / region);
    printf("  persistent spinning worker pool     %8.4f us   %6.2fx cheaper than a region\n",
           pooled, region / pooled);
    printf("\n  The `if(0)` row is why core/parallel.h branches instead of using the\n"
           "  clause: libgomp still builds a team and calls the outlined function.\n\n");

    FILE *csv = fopen(csv_path, "a");
    if (csv) {
        if (ftell(csv) == 0) {
            fprintf(csv, "section,tier,shape,m,k,n,work,chunks,forked_us,serial_us,"
                         "speedup,threads," BENCH_METADATA_CSV_HEADER "\n");
        }
        fprintf(csv, "\"dispatch\",\"\",\"omp_parallel_for\",0,0,0,0,0,%.6f,%.6f,%.4f,%d",
                region, plain, plain / region, threads);
        bench_csv_metadata(csv, &metadata);
        fprintf(csv, "\"dispatch\",\"\",\"omp_parallel_for_if_false\",0,0,0,0,0,%.6f,%.6f,%.4f,%d",
                region_if, plain, plain / region_if, threads);
        bench_csv_metadata(csv, &metadata);
        fprintf(csv, "\"dispatch\",\"\",\"persistent_pool\",0,0,0,0,0,%.6f,%.6f,%.4f,%d",
                pooled, plain, plain / pooled, threads);
        bench_csv_metadata(csv, &metadata);
    }

    /* Stop the prototype before measuring anything else. Its workers spin, so
     * leaving them alive would hold one core for the whole sweep below and
     * quietly penalise the forked side of every comparison - which is exactly
     * what happened the first time this tool was run. */
    pool_stop();

    printf("Forking versus not, same kernel and buffers, best of %d:\n\n", rounds);
    printf("%-8s %-30s %12s %7s %10s %10s %8s  %s\n",
           "tier", "shape", "work", "chunks", "forked", "serial", "speedup",
           "verdict");

    const size_t saved_min_work = parallel_min_work();
    /* The chunks < 2 rows run identical code on both sides, so whatever ratio
     * they report is this machine's noise and nothing else. Collecting their
     * worst excursion turns the control into a stated resolution limit. */
    double noise_low = 1.0, noise_high = 1.0;
    for (size_t s = 0; s < shape_count; s++) {
        const shape_t *shape = &shapes[s];
        size_t m = shape->m, k = shape->k, n = shape->n;
        float *a = malloc(m * k * sizeof(float));
        float *b = malloc(k * n * sizeof(float));
        float *c = malloc(m * n * sizeof(float));
        if (!a || !b || !c) {
            fprintf(stderr, "Error: out of memory at %zux%zux%zu\n", m, k, n);
            free(a); free(b); free(c);
            if (csv) fclose(csv);
            return 1;
        }
        for (size_t i = 0; i < m * k; i++) a[i] = (float)((i % 13) - 6) / 7.0f;
        for (size_t i = 0; i < k * n; i++) b[i] = (float)((i % 11) - 5) / 7.0f;

        /* ABBA, not one side then the other, so a machine that gets busier
         * partway through the measurement cannot favour whichever side ran
         * first. */
        double forked = time_shape(shape, a, b, c, 1, rounds);
        double serial = time_shape(shape, a, b, c, (size_t)-1, rounds);
        double serial2 = time_shape(shape, a, b, c, (size_t)-1, rounds);
        double forked2 = time_shape(shape, a, b, c, 1, rounds);
        if (forked2 < forked) forked = forked2;
        if (serial2 < serial) serial = serial2;

        size_t work = m * k * n;
        double speedup = serial / forked;

        /* What the blocked kernels give OpenMP to distribute. Below two there
         * is nothing to distribute at any amount of work, so those rows say so
         * instead of reporting a speedup that is only ever noise. */
        const size_t tile = matmul_tile_size();
        const size_t chunks = ((m + tile - 1) / tile) * ((n + tile - 1) / tile);
        const char *verdict = (chunks < 2)      ? "cannot fork"
                            : (speedup > 1.05)  ? "fork"
                            : (speedup < 0.95)  ? "serial" : "tie";

        if (chunks < 2) {
            if (speedup < noise_low) noise_low = speedup;
            if (speedup > noise_high) noise_high = speedup;
        }

        printf("%-8s %-30s %12zu %7zu %9.3fus %9.3fus %7.2fx  %s\n",
               shape->tier, shape->name, work, chunks, forked, serial, speedup,
               verdict);

        if (csv) {
            fprintf(csv, "\"crossover\",");
            bench_csv_field(csv, shape->tier);
            fprintf(csv, ",");
            bench_csv_field(csv, shape->name);
            fprintf(csv, ",%zu,%zu,%zu,%zu,%zu,%.6f,%.6f,%.4f,%d",
                    m, k, n, work, chunks, forked, serial, speedup, threads);
            bench_csv_metadata(csv, &metadata);
        }

        free(a); free(b); free(c);
    }
    parallel_set_min_work(saved_min_work);

    printf("\n\"fork\" and \"serial\" mark a difference of more than 5%%; anything closer\n"
           "is a tie, which is what the shapes near the threshold are supposed to be.\n"
           "\"cannot fork\" means the blocked kernel produced fewer than two blocks at\n"
           "the current tile, so both columns ran the same serial code and the ratio\n"
           "measures only this machine's noise - useful as a noise floor, and not\n"
           "evidence about the threshold either way.\n");
    printf("\nNoise floor from those control rows this session: %.2fx to %.2fx.\n"
           "A forkable row inside that band has not measured anything. Only rows\n"
           "outside it are evidence, and a session whose band is wide should be\n"
           "repeated on a quieter machine rather than interpreted.\n",
           noise_low, noise_high);
    if (csv) {
        printf("Results appended to %s\n", csv_path);
        fclose(csv);
    }

    return 0;
#endif
}
