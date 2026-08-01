/*
 * Standalone benchmark tool: memory footprint and throughput (inference +
 * training) across a few representative model sizes, on CPU and - when a
 * usable CUDA GPU is present - also with model->use_gpu = 1 (see
 * transformer.c's dispatch_matmul()), so the two are directly comparable.
 * Single-threaded by default (OMP=1 builds parallelize the matmul/
 * attention internals, but the driver loop itself makes no threading
 * decisions) so the CPU numbers reflect what a single low-end core can do.
 *
 * Full-model runs append to bench_results_v2.csv. `--matmul-only` instead
 * compares the scalar and default portable CPU kernels on isolated shapes
 * and appends to matmul_results_v1.csv. Both files are gitignored so local
 * histories can accumulate without presenting cross-machine numbers as
 * directly comparable.
 *
 * Deliberately its own small file/binary (bench.out) rather than a mode
 * bolted onto main.c/cli.c - it links directly against the model modules
 * and has nothing to do with training-run orchestration or the tokenizer.
 *
 * Build:  make bench      (or: make bench OMP=1 CC=gcc)
 * Run:    ./bench.out
 */

#include "core/model.h"
#include "core/tensor_ops.h"
#include "backends/gpu/gpu_matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef DRANZER_BUILD_COMMAND
#define DRANZER_BUILD_COMMAND "unknown (built outside the project Makefile)"
#endif

typedef struct {
    const char *key;
    const char *name;
    size_t vocab_size, embedding_dim, num_heads, num_layers, max_seq_len;
} bench_config_t;

typedef struct {
    double inference_ms_per_token;
    double prefill_ms_per_token;
    double growing_decode_ms_per_token;
    double sliding_decode_ms_per_token;
    double training_ms_per_step;
} bench_result_t;

typedef struct {
    char compiler[192];
    char os[256];
    char cpu[256];
    const char *build_command;
    long online_cpus;
    long openmp_version;
    int max_threads;
} bench_metadata_t;

typedef struct {
    const char *name;
    size_t m, k, n;
} matmul_case_t;

typedef struct {
    size_t scalar_iterations;
    size_t dispatch_iterations;
    double scalar_us;
    double dispatch_us;
    double scalar_gflops;
    double dispatch_gflops;
    double speedup;
    float max_abs_error;
    float max_relative_error;
} matmul_result_t;

/* Read after every timed region so an optimizing compiler cannot consider
 * the matrix results dead. The kernels live in another translation unit,
 * but keeping an explicit observable consumer also makes that contract
 * survive a future link-time-optimization build. */
static volatile float matmul_sink;

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

static void collect_metadata(bench_metadata_t *metadata) {
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

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long peak_rss_kb(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; /* KB on Linux */
}

/* Runs `iters` forward passes at seq_len == cfg->max_seq_len (the
 * worst-case, steady-state cost once a generated sequence has filled its
 * context window - there's no KV cache here, so every generation step
 * reprocesses the whole context from scratch) and reports latency/throughput. */
static double bench_inference(neural_model_t *model, const bench_config_t *cfg, int iters) {
    uint32_t *tokens = malloc(cfg->max_seq_len * sizeof(uint32_t));
    float *logits = malloc(cfg->vocab_size * sizeof(float));
    for (size_t i = 0; i < cfg->max_seq_len; i++) tokens[i] = (uint32_t)((i * 31 + 7) % cfg->vocab_size);

    model->is_training = 0;
    model_forward(model, tokens, cfg->max_seq_len, logits); /* warm up */

    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        model_forward(model, tokens, cfg->max_seq_len, logits);
    }
    double elapsed = now_sec() - t0;
    double per_token_ms = 1000.0 * elapsed / iters;

    printf("  inference (seq_len=%-3zu, full context each step, no KV cache):\n", cfg->max_seq_len);
    printf("    %.3f ms/token   (%.1f tokens/sec)\n", per_token_ms, 1000.0 / per_token_ms);

    free(tokens);
    free(logits);
    return per_token_ms;
}

/* Measure prompt prefill, growing-cache decode, and full-window ring decode
 * independently. The full-prefix comparison covers positions before the
 * first eviction, where both paths have identical semantics. */
static int bench_kv_decode(neural_model_t *model, const bench_config_t *cfg,
                           bench_result_t *result) {
    size_t decode_steps = cfg->max_seq_len < 8 ? cfg->max_seq_len : 8;
    size_t prefill_len = cfg->max_seq_len - decode_steps;
    if (prefill_len == 0) {
        prefill_len = 1;
        decode_steps = cfg->max_seq_len - 1;
    }
    if (decode_steps == 0 || !result) return -1;

    size_t token_count = cfg->max_seq_len + decode_steps;
    uint32_t *tokens = malloc(token_count * sizeof(uint32_t));
    float *logits = malloc(cfg->vocab_size * sizeof(float));
    if (!tokens || !logits) {
        free(tokens);
        free(logits);
        printf("  KV-cache decode benchmark: allocation failed\n");
        return -1;
    }
    for (size_t i = 0; i < token_count; i++) {
        tokens[i] = (uint32_t)((i * 43 + 11) % cfg->vocab_size);
    }

    model->is_training = 0;
    model_forward(model, tokens, cfg->max_seq_len, logits); /* warm up */

    double full_t0 = now_sec();
    for (size_t pos = prefill_len; pos < cfg->max_seq_len; pos++) {
        model_forward(model, tokens, pos + 1, logits);
    }
    double full_ms = 1000.0 * (now_sec() - full_t0) / (double)decode_steps;

    model_kv_cache_t cache = {0};
    if (model_kv_cache_init(&cache, model) != MODEL_SUCCESS) {
        free(tokens);
        free(logits);
        printf("  KV-cache decode benchmark: cache allocation failed\n");
        return -1;
    }
    double prefill_t0 = now_sec();
    for (size_t i = 0; i < prefill_len; i++) {
        model_forward_token(model, &cache, tokens[i], logits);
    }
    double prefill_total_ms = 1000.0 * (now_sec() - prefill_t0);

    double cached_t0 = now_sec();
    for (size_t pos = prefill_len; pos < cfg->max_seq_len; pos++) {
        model_forward_token(model, &cache, tokens[pos], logits);
    }
    double cached_ms = 1000.0 * (now_sec() - cached_t0) / (double)decode_steps;

    double sliding_t0 = now_sec();
    for (size_t pos = cfg->max_seq_len;
         pos < cfg->max_seq_len + decode_steps; pos++) {
        model_forward_token(model, &cache, tokens[pos], logits);
    }
    double sliding_ms = 1000.0 * (now_sec() - sliding_t0) /
                        (double)decode_steps;
    result->prefill_ms_per_token = prefill_total_ms / (double)prefill_len;
    result->growing_decode_ms_per_token = cached_ms;
    result->sliding_decode_ms_per_token = sliding_ms;

    printf("  prompt prefill (%zu tokens, cache allocation excluded):\n", prefill_len);
    printf("    %.3f ms total   %.3f ms/token\n",
           prefill_total_ms, prefill_total_ms / (double)prefill_len);
    printf("  autoregressive decode (last %zu growing-cache positions):\n", decode_steps);
    printf("    full-prefix %.3f ms/token   KV-cache %.3f ms/token   speedup %.2fx\n",
           full_ms, cached_ms, full_ms / cached_ms);
    printf("  sliding decode (full %zu-token ring, %zu evictions):\n",
           cfg->max_seq_len, decode_steps);
    printf("    %.3f ms/token   (%.1f tokens/sec)\n",
           sliding_ms, 1000.0 / sliding_ms);

    model_kv_cache_free(&cache);
    free(tokens);
    free(logits);
    return 0;
}

/* GPU forward dispatch only touches model_forward's matmuls (see
 * transformer.c) - the backward pass and optimizer step inside
 * model_train_step stay CPU-only regardless of model->use_gpu, so this
 * measures "how much does GPU-accelerating just the forward half of a
 * training step help", not a fully GPU-resident training step. */
static double bench_training(neural_model_t *model, const bench_config_t *cfg, int iters) {
    uint32_t *tokens = malloc(cfg->max_seq_len * sizeof(uint32_t));
    for (size_t i = 0; i < cfg->max_seq_len; i++) tokens[i] = (uint32_t)((i * 17 + 3) % cfg->vocab_size);
    uint32_t target = (uint32_t)(cfg->vocab_size / 2);

    model_train_step(model, tokens, target, cfg->max_seq_len); /* warm up */

    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        model_train_step(model, tokens, target, cfg->max_seq_len);
    }
    double elapsed = now_sec() - t0;
    double per_step_ms = 1000.0 * elapsed / iters;

    printf("  training  (seq_len=%-3zu, forward+backward+optimizer step):\n", cfg->max_seq_len);
    printf("    %.3f ms/step    (%.1f steps/sec, %.1f context-tokens/sec)\n",
           per_step_ms, 1000.0 / per_step_ms, cfg->max_seq_len * 1000.0 / per_step_ms);

    free(tokens);
    return per_step_ms;
}

static void csv_field(FILE *csv, const char *value) {
    fputc('"', csv);
    for (const char *cursor = value ? value : ""; *cursor; cursor++) {
        if (*cursor == '"') fputc('"', csv);
        fputc(*cursor, csv);
    }
    fputc('"', csv);
}

static double measure_matmul(int use_scalar, float *a, float *b, float *c,
                             size_t m, size_t k, size_t n,
                             double target_seconds, size_t *iterations_out) {
    const size_t max_iterations = 1U << 20;
    size_t iterations = 1;
    double elapsed = 0.0;

    if (use_scalar) {
        matrix_multiply_scalar(a, b, c, m, k, n);
    } else {
        matrix_multiply(a, b, c, m, k, n);
    }

    for (;;) {
        double started = now_sec();
        for (size_t i = 0; i < iterations; i++) {
            if (use_scalar) {
                matrix_multiply_scalar(a, b, c, m, k, n);
            } else {
                matrix_multiply(a, b, c, m, k, n);
            }
        }
        elapsed = now_sec() - started;
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

static int compare_matmul_case(const matmul_case_t *test_case, int quick,
                               matmul_result_t *result) {
    if (!test_case || !result || test_case->m == 0 || test_case->k == 0 ||
        test_case->n == 0 || test_case->m > SIZE_MAX / test_case->k ||
        test_case->k > SIZE_MAX / test_case->n ||
        test_case->m > SIZE_MAX / test_case->n) {
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
    float *dispatch = malloc(c_count * sizeof(float));
    if (!a || !b || !reference || !dispatch) {
        free(a);
        free(b);
        free(reference);
        free(dispatch);
        return -1;
    }

    for (size_t i = 0; i < a_count; i++) {
        a[i] = (float)((int)(i % 101U) - 50) / 51.0f;
    }
    for (size_t i = 0; i < b_count; i++) {
        b[i] = (float)((int)((i * 17U + 13U) % 103U) - 51) / 52.0f;
    }

    matrix_multiply_scalar(a, b, reference, test_case->m, test_case->k,
                           test_case->n);
    matrix_multiply(a, b, dispatch, test_case->m, test_case->k,
                    test_case->n);

    float max_reference = 0.0f;
    result->max_abs_error = 0.0f;
    result->max_relative_error = 0.0f;
    for (size_t i = 0; i < c_count; i++) {
        float abs_error = fabsf(reference[i] - dispatch[i]);
        float reference_abs = fabsf(reference[i]);
        float relative_error = abs_error / fmaxf(reference_abs, 1e-6f);
        if (!isfinite(reference[i]) || !isfinite(dispatch[i])) {
            result->max_abs_error = INFINITY;
            result->max_relative_error = INFINITY;
            break;
        }
        if (reference_abs > max_reference) max_reference = reference_abs;
        if (abs_error > result->max_abs_error) result->max_abs_error = abs_error;
        if (relative_error > result->max_relative_error) {
            result->max_relative_error = relative_error;
        }
    }

    double target_seconds = quick ? 0.01 : 0.15;
    double scalar_seconds = measure_matmul(1, a, b, reference,
                                           test_case->m, test_case->k,
                                           test_case->n, target_seconds,
                                           &result->scalar_iterations);
    double dispatch_seconds = measure_matmul(0, a, b, dispatch,
                                             test_case->m, test_case->k,
                                             test_case->n, target_seconds,
                                             &result->dispatch_iterations);
    double operations = 2.0 * (double)test_case->m * (double)test_case->k *
                        (double)test_case->n;
    result->scalar_us = scalar_seconds * 1e6;
    result->dispatch_us = dispatch_seconds * 1e6;
    result->scalar_gflops = operations / scalar_seconds / 1e9;
    result->dispatch_gflops = operations / dispatch_seconds / 1e9;
    result->speedup = scalar_seconds / dispatch_seconds;

    int correct = result->max_abs_error <= 1e-4f * (1.0f + max_reference);
    free(a);
    free(b);
    free(reference);
    free(dispatch);
    return correct ? 0 : -1;
}

static void matmul_csv_log(FILE *csv, const char *timestamp,
                           const bench_config_t *cfg,
                           const matmul_case_t *test_case,
                           const matmul_result_t *result,
                           const bench_metadata_t *metadata) {
    if (!csv) return;
    csv_field(csv, timestamp);
    fputc(',', csv);
    csv_field(csv, cfg->key);
    fputc(',', csv);
    csv_field(csv, test_case->name);
    fprintf(csv, ",%zu,%zu,%zu,%zu,%zu,%.4f,%.4f,%.6f,%.6f,%.4f,%.8g,%.8g,",
            test_case->m, test_case->k, test_case->n,
            result->scalar_iterations, result->dispatch_iterations,
            result->scalar_us, result->dispatch_us,
            result->scalar_gflops, result->dispatch_gflops, result->speedup,
            result->max_abs_error, result->max_relative_error);
    csv_field(csv, metadata->build_command);
    fputc(',', csv);
    csv_field(csv, metadata->compiler);
    fputc(',', csv);
    csv_field(csv, metadata->os);
    fputc(',', csv);
    csv_field(csv, metadata->cpu);
    fprintf(csv, ",%ld,%ld,%d\n", metadata->online_cpus,
            metadata->openmp_version, metadata->max_threads);
    fflush(csv);
}

static int run_matmul_suite(const bench_config_t *configs, size_t num_configs,
                            const char *selected_tier, int quick,
                            const char *timestamp,
                            const bench_metadata_t *metadata) {
    const char *csv_path = "matmul_results_v1.csv";
    int csv_existed = (access(csv_path, F_OK) == 0);
    FILE *csv = fopen(csv_path, "a");
    if (!csv) {
        fprintf(stderr, "Error: cannot open %s\n", csv_path);
        return 1;
    }
    if (!csv_existed) {
        fprintf(csv, "timestamp,tier,case,m,k,n,scalar_iterations,dispatch_iterations,"
                     "scalar_us,dispatch_us,scalar_gflops,dispatch_gflops,dispatch_speedup,"
                     "max_abs_error,max_relative_error,build_command,compiler,os,cpu,"
                     "online_cpus,openmp_version,max_threads\n");
    }

    int failed = 0;
    for (size_t i = 0; i < num_configs; i++) {
        const bench_config_t *cfg = &configs[i];
        if (selected_tier && strcmp(selected_tier, cfg->key) != 0) continue;
        size_t ffn_dim = cfg->embedding_dim * 4U;
        matmul_case_t cases[] = {
            { "decode_attention_projection", 1, cfg->embedding_dim,
              cfg->embedding_dim },
            { "decode_ffn_up", 1, cfg->embedding_dim, ffn_dim },
            { "prefill_attention_projection", cfg->max_seq_len,
              cfg->embedding_dim, cfg->embedding_dim },
            { "prefill_ffn_up", cfg->max_seq_len, cfg->embedding_dim,
              ffn_dim },
            { "training_ffn_down", cfg->max_seq_len, ffn_dim,
              cfg->embedding_dim },
        };
        size_t case_count = sizeof(cases) / sizeof(cases[0]);
        printf("--- isolated matmul: %s ---\n", cfg->name);
        for (size_t j = 0; j < case_count; j++) {
            matmul_result_t result = {0};
            int rc = compare_matmul_case(&cases[j], quick, &result);
            printf("  %-31s %zux%zux%zu  scalar %8.3f us  dispatch %8.3f us  "
                   "%6.2fx  error %.3g%s\n",
                   cases[j].name, cases[j].m, cases[j].k, cases[j].n,
                   result.scalar_us, result.dispatch_us, result.speedup,
                   result.max_abs_error, rc == 0 ? "" : "  FAIL");
            matmul_csv_log(csv, timestamp, cfg, &cases[j], &result, metadata);
            if (rc != 0) failed = 1;
        }
        printf("\n");
    }

    printf("Matmul results appended to %s\n", csv_path);
    fclose(csv);
    return failed;
}

static void csv_log(FILE *csv, const char *timestamp, const bench_config_t *cfg,
                    const char *optimizer_name, const char *mode,
                    long param_floats, const bench_result_t *r,
                    long rss_delta_kb, const bench_metadata_t *metadata) {
    if (!csv) return;
    csv_field(csv, timestamp);
    fputc(',', csv);
    csv_field(csv, cfg->name);
    fprintf(csv, ",%zu,%zu,%zu,%zu,%zu,%s,%s,%ld,"
                 "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,",
            cfg->vocab_size, cfg->embedding_dim, cfg->num_heads,
            cfg->num_layers, cfg->max_seq_len, optimizer_name, mode, param_floats,
            r->inference_ms_per_token, 1000.0 / r->inference_ms_per_token,
            r->prefill_ms_per_token, r->growing_decode_ms_per_token,
            r->sliding_decode_ms_per_token,
            r->training_ms_per_step, 1000.0 / r->training_ms_per_step,
            rss_delta_kb / 1024.0);
    csv_field(csv, metadata->build_command);
    fputc(',', csv);
    csv_field(csv, metadata->compiler);
    fputc(',', csv);
    csv_field(csv, metadata->os);
    fputc(',', csv);
    csv_field(csv, metadata->cpu);
    fprintf(csv, ",%ld,%ld,%d\n", metadata->online_cpus,
            metadata->openmp_version, metadata->max_threads);
    fflush(csv);
}

static void run_config(const bench_config_t *cfg, optimizer_type_t optimizer,
                       int use_gpu, int use_scalar, int infer_iters,
                       int train_iters, FILE *csv, const char *timestamp,
                       const bench_metadata_t *metadata) {
    if (use_gpu && !gpu_matmul_available()) {
        printf("--- %s: GPU requested but not available - skipping GPU run ---\n\n", cfg->name);
        return;
    }

    neural_model_t model = {0};
    long rss_before = peak_rss_kb();

    if (model_new(&model, cfg->vocab_size, cfg->embedding_dim, cfg->num_heads,
                  cfg->num_layers, cfg->max_seq_len) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed for config %s\n", cfg->name);
        return;
    }
    model.optimizer_type = optimizer;
    model.use_gpu = use_gpu;
    model.use_scalar_matmul = use_scalar;

    const char *optimizer_name = (optimizer == OPTIMIZER_ADAM) ? "adam" : "sgd";
    printf("--- %s (vocab=%zu emb=%zu heads=%zu layers=%zu max_seq=%zu) optimizer=%s mode=%s ---\n",
           cfg->name, cfg->vocab_size, cfg->embedding_dim, cfg->num_heads, cfg->num_layers,
           cfg->max_seq_len, optimizer_name,
           use_gpu ? "GPU" : use_scalar ? "CPU scalar reference" : "CPU dispatch");

    size_t param_floats = model.total_param_count;
    printf("  parameters: %zu (%.2f MB weights only)\n",
           param_floats, param_floats * sizeof(float) / (1024.0 * 1024.0));

    bench_result_t r = {0};
    r.inference_ms_per_token = bench_inference(&model, cfg, infer_iters);
    (void)bench_kv_decode(&model, cfg, &r);
    r.training_ms_per_step = bench_training(&model, cfg, train_iters);

    long rss_after = peak_rss_kb();
    printf("  process peak RSS after this config: %.2f MB (delta from process start: %.2f MB)\n",
           rss_after / 1024.0, (rss_after - rss_before) / 1024.0);
    printf("\n");

    const char *mode = use_gpu ? "gpu" : use_scalar ? "cpu-scalar" : "cpu-dispatch";
    csv_log(csv, timestamp, cfg, optimizer_name, mode, (long)param_floats,
            &r, rss_after - rss_before, metadata);

    model_free(&model);
}

static void print_usage(const char *program) {
    printf("Usage: %s [--tier tiny|small|medium] [--scalar] [--quick] [--matmul-only]\n",
           program);
}

int main(int argc, char **argv) {
    const char *selected_tier = NULL;
    int use_scalar = 0;
    int quick = 0;
    int matmul_only = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tier") == 0 && i + 1 < argc) {
            selected_tier = argv[++i];
        } else if (strcmp(argv[i], "--scalar") == 0) {
            use_scalar = 1;
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        } else if (strcmp(argv[i], "--matmul-only") == 0) {
            matmul_only = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: unknown or incomplete option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (selected_tier && strcmp(selected_tier, "tiny") != 0 &&
        strcmp(selected_tier, "small") != 0 &&
        strcmp(selected_tier, "medium") != 0) {
        fprintf(stderr, "Error: unknown benchmark tier '%s'\n", selected_tier);
        return 2;
    }
    if (matmul_only && use_scalar) {
        fprintf(stderr, "Error: --matmul-only already compares scalar and dispatch paths\n");
        return 2;
    }

    bench_metadata_t metadata;
    collect_metadata(&metadata);
    printf("Build: %s\nCompiler: %s\nSystem: %s\nCPU: %s\n",
           metadata.build_command, metadata.compiler, metadata.os, metadata.cpu);
    printf("Threads: online=%ld OpenMP=%ld max=%d\n\n",
           metadata.online_cpus, metadata.openmp_version, metadata.max_threads);
#ifdef _OPENMP
    printf("Built with OpenMP (OMP=1) - matmul/attention parallelize across cores internally.\n\n");
#else
    printf("Built WITHOUT OpenMP - single-threaded throughout (rebuild with `make bench OMP=1 CC=gcc` to compare).\n\n");
#endif

    int gpu_available = gpu_matmul_available();
    printf(gpu_available
           ? "CUDA GPU detected - each config runs on CPU and GPU (GPU dispatch covers model_forward's\n"
             "matmuls only - see transformer.c's dispatch_matmul() - backward stays CPU-only regardless).\n\n"
           : "No CUDA GPU detected - CPU-only run (see gpu_probe.out for why).\n\n");

    const char *csv_path = "bench_results_v2.csv";
    int csv_existed = (access(csv_path, F_OK) == 0);
    FILE *csv = fopen(csv_path, "a");
    if (csv && !csv_existed) {
        fprintf(csv, "timestamp,config,vocab_size,embedding_dim,num_heads,num_layers,max_seq_len,"
                      "optimizer,mode,param_count,inference_ms_per_token,inference_tokens_per_sec,"
                      "prefill_ms_per_token,growing_decode_ms_per_token,sliding_decode_ms_per_token,"
                      "training_ms_per_step,training_steps_per_sec,rss_delta_mb,build_command,"
                      "compiler,os,cpu,online_cpus,openmp_version,max_threads\n");
    }
    char timestamp[32];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", localtime(&now));

    bench_config_t configs[] = {
        /* Matches this project's current main.c defaults. */
        { "tiny", "tiny (project default)", 260, 16, 2, 2, 32 },
        /* A step up: still CPU/RAM-friendly on something like a Raspberry Pi. */
        { "small", "small",                 1000, 64, 4, 4, 64 },
        /* Rough "GPT-2 small"-scale embedding/layer count, kept at a modest
         * vocab so it stays feasible to actually run in this benchmark. */
        { "medium", "medium",                4000, 256, 8, 6, 128 },
    };
    size_t num_configs = sizeof(configs) / sizeof(configs[0]);

    if (matmul_only) {
        return run_matmul_suite(configs, num_configs, selected_tier, quick,
                                timestamp, &metadata);
    }

    for (size_t i = 0; i < num_configs; i++) {
        if (selected_tier && strcmp(selected_tier, configs[i].key) != 0) continue;
        int infer_iters = quick ? 5 : 50;
        int train_iters = quick ? 2 : 20;
        run_config(&configs[i], OPTIMIZER_ADAM, 0, use_scalar,
                   infer_iters, train_iters, csv, timestamp, &metadata);
        /* GPU runs use fewer iterations: gpu_matmul() currently does a
         * fresh alloc/upload/launch/download/free on every single call
         * (see gpu_matmul.c) rather than reusing persistent device
         * buffers, so per-call overhead is real and this keeps total
         * benchmark time bounded while still measuring that overhead
         * honestly rather than hiding it. */
        if (gpu_available && !use_scalar) {
            run_config(&configs[i], OPTIMIZER_ADAM, 1, 0,
                       quick ? 2 : 10, quick ? 1 : 5,
                       csv, timestamp, &metadata);
        }
    }

    if (csv) {
        printf("Results appended to %s\n", csv_path);
        fclose(csv);
    }

    return 0;
}
