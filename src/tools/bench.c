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
 * hands off to tools/bench_matmul.c, which compares the interchangeable CPU
 * kernels on isolated shapes and appends to matmul_results_v2.csv. Both
 * files are gitignored so local histories can accumulate without presenting
 * cross-machine numbers as directly comparable.
 *
 * Deliberately its own small file/binary (bench.out) rather than a mode
 * bolted onto main.c/cli.c - it links directly against the model modules
 * and has nothing to do with training-run orchestration or the tokenizer.
 * Provenance/timing helpers live in tools/bench_support.c and the isolated
 * kernel comparison in tools/bench_matmul.c, so this file stays about
 * whole-model throughput and memory.
 *
 * Build:  make bench      (or: make bench OMP=1 CC=gcc)
 * Run:    ./bench.out
 */

#include "core/model.h"
#include "core/matmul.h"
#include "core/tensor_ops.h"
#include "backends/gpu/gpu_matmul.h"
#include "tools/bench_matmul.h"
#include "tools/bench_support.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

    double t0 = bench_now_sec();
    for (int it = 0; it < iters; it++) {
        model_forward(model, tokens, cfg->max_seq_len, logits);
    }
    double elapsed = bench_now_sec() - t0;
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

    double full_t0 = bench_now_sec();
    for (size_t pos = prefill_len; pos < cfg->max_seq_len; pos++) {
        model_forward(model, tokens, pos + 1, logits);
    }
    double full_ms = 1000.0 * (bench_now_sec() - full_t0) / (double)decode_steps;

    model_kv_cache_t cache = {0};
    if (model_kv_cache_init(&cache, model) != MODEL_SUCCESS) {
        free(tokens);
        free(logits);
        printf("  KV-cache decode benchmark: cache allocation failed\n");
        return -1;
    }
    double prefill_t0 = bench_now_sec();
    for (size_t i = 0; i < prefill_len; i++) {
        model_forward_token(model, &cache, tokens[i], logits);
    }
    double prefill_total_ms = 1000.0 * (bench_now_sec() - prefill_t0);

    double cached_t0 = bench_now_sec();
    for (size_t pos = prefill_len; pos < cfg->max_seq_len; pos++) {
        model_forward_token(model, &cache, tokens[pos], logits);
    }
    double cached_ms = 1000.0 * (bench_now_sec() - cached_t0) / (double)decode_steps;

    double sliding_t0 = bench_now_sec();
    for (size_t pos = cfg->max_seq_len;
         pos < cfg->max_seq_len + decode_steps; pos++) {
        model_forward_token(model, &cache, tokens[pos], logits);
    }
    double sliding_ms = 1000.0 * (bench_now_sec() - sliding_t0) /
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

    double t0 = bench_now_sec();
    for (int it = 0; it < iters; it++) {
        model_train_step(model, tokens, target, cfg->max_seq_len);
    }
    double elapsed = bench_now_sec() - t0;
    double per_step_ms = 1000.0 * elapsed / iters;

    printf("  training  (seq_len=%-3zu, forward+backward+optimizer step):\n", cfg->max_seq_len);
    printf("    %.3f ms/step    (%.1f steps/sec, %.1f context-tokens/sec)\n",
           per_step_ms, 1000.0 / per_step_ms, cfg->max_seq_len * 1000.0 / per_step_ms);

    free(tokens);
    return per_step_ms;
}

static void csv_log(FILE *csv, const char *timestamp, const bench_config_t *cfg,
                    const char *optimizer_name, const char *mode,
                    long param_floats, const bench_result_t *r,
                    long rss_delta_kb, const bench_metadata_t *metadata) {
    if (!csv) return;
    bench_csv_field(csv, timestamp);
    fputc(',', csv);
    bench_csv_field(csv, cfg->name);
    fprintf(csv, ",%zu,%zu,%zu,%zu,%zu,%s,%s,%ld,"
                 "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,",
            cfg->vocab_size, cfg->embedding_dim, cfg->num_heads,
            cfg->num_layers, cfg->max_seq_len, optimizer_name, mode, param_floats,
            r->inference_ms_per_token, 1000.0 / r->inference_ms_per_token,
            r->prefill_ms_per_token, r->growing_decode_ms_per_token,
            r->sliding_decode_ms_per_token,
            r->training_ms_per_step, 1000.0 / r->training_ms_per_step,
            rss_delta_kb / 1024.0);
    bench_csv_metadata(csv, metadata);
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
    long rss_before = bench_peak_rss_kb();

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
           use_gpu ? "GPU" : use_scalar ? "CPU scalar reference"
                                        : matmul_kernel_name(matmul_get_kernel()));

    size_t param_floats = model.total_param_count;
    printf("  parameters: %zu (%.2f MB weights only)\n",
           param_floats, param_floats * sizeof(float) / (1024.0 * 1024.0));

    bench_result_t r = {0};
    r.inference_ms_per_token = bench_inference(&model, cfg, infer_iters);
    (void)bench_kv_decode(&model, cfg, &r);
    r.training_ms_per_step = bench_training(&model, cfg, train_iters);

    long rss_after = bench_peak_rss_kb();
    printf("  process peak RSS after this config: %.2f MB (delta from process start: %.2f MB)\n",
           rss_after / 1024.0, (rss_after - rss_before) / 1024.0);
    printf("\n");

    /* Encode the CPU kernel and tile into the existing `mode` column rather
     * than adding columns: a v2 results file written by an older build stays
     * readable, and a row still says exactly which kernel produced it. */
    char mode[64];
    if (use_gpu) {
        snprintf(mode, sizeof(mode), "gpu");
    } else if (use_scalar) {
        snprintf(mode, sizeof(mode), "cpu-scalar");
    } else {
        snprintf(mode, sizeof(mode), "cpu-%s-tile%zu",
                 matmul_kernel_name(matmul_get_kernel()), matmul_tile_size());
    }
    csv_log(csv, timestamp, cfg, optimizer_name, mode, (long)param_floats,
            &r, rss_after - rss_before, metadata);

    model_free(&model);
}

static void print_usage(const char *program) {
    printf("Usage: %s [--tier tiny|small|medium] [--scalar] [--quick]\n"
           "          [--kernel auto|scalar|rowwise|tiled|tiled_mr4] [--tile N]\n"
           "          [--matmul-only [--sweep] [--repeats N] [--csv-path FILE]]\n\n"
           "  --kernel/--tile   override the CPU matmul kernel and tile for the run\n"
           "  --matmul-only     compare kernels on isolated shapes instead of whole models\n"
           "  --sweep           with --matmul-only: measure every kernel and tile candidate\n"
           "  --repeats N       with --matmul-only: timing rounds per candidate (median wins)\n"
           "  --csv-path FILE   with --matmul-only: results file to append to\n",
           program);
}

/* Strict positive-integer parsing, matching the CLI's own contract: no
 * silent truncation, no partially numeric arguments, no zero. */
static int parse_positive(const char *text, size_t *value_out) {
    if (!text || *text == '\0') return -1;
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed == 0 ||
        parsed > (unsigned long long)SIZE_MAX) {
        return -1;
    }
    *value_out = (size_t)parsed;
    return 0;
}

int main(int argc, char **argv) {
    const char *selected_tier = NULL;
    int use_scalar = 0;
    int quick = 0;
    int matmul_only = 0;
    int sweep = 0;
    size_t repeats = 3;
    size_t tile = 0;
    const char *matmul_csv_path = NULL;
    matmul_kernel_t kernel = MATMUL_KERNEL_AUTO;
    int kernel_overridden = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tier") == 0 && i + 1 < argc) {
            selected_tier = argv[++i];
        } else if (strcmp(argv[i], "--scalar") == 0) {
            use_scalar = 1;
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        } else if (strcmp(argv[i], "--matmul-only") == 0) {
            matmul_only = 1;
        } else if (strcmp(argv[i], "--sweep") == 0) {
            sweep = 1;
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            if (matmul_kernel_from_name(argv[++i], &kernel) != 0) {
                fprintf(stderr, "Error: unknown matmul kernel '%s'\n", argv[i]);
                return 2;
            }
            kernel_overridden = 1;
        } else if (strcmp(argv[i], "--tile") == 0 && i + 1 < argc) {
            if (parse_positive(argv[++i], &tile) != 0) {
                fprintf(stderr, "Error: --tile needs a positive integer, got '%s'\n",
                        argv[i]);
                return 2;
            }
        } else if (strcmp(argv[i], "--csv-path") == 0 && i + 1 < argc) {
            matmul_csv_path = argv[++i];
            if (*matmul_csv_path == '\0') {
                fprintf(stderr, "Error: --csv-path needs a file name\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            if (parse_positive(argv[++i], &repeats) != 0) {
                fprintf(stderr, "Error: --repeats needs a positive integer, got '%s'\n",
                        argv[i]);
                return 2;
            }
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
    if (sweep && !matmul_only) {
        fprintf(stderr, "Error: --sweep only applies to --matmul-only runs\n");
        return 2;
    }
    if (matmul_csv_path && !matmul_only) {
        fprintf(stderr, "Error: --csv-path only applies to --matmul-only runs\n");
        return 2;
    }
    if (matmul_only && kernel_overridden) {
        fprintf(stderr, "Error: --matmul-only measures every kernel itself; drop --kernel\n");
        return 2;
    }
    if (tile != 0 && matmul_set_tile_size(tile) != 0) {
        fprintf(stderr, "Error: --tile must be positive\n");
        return 2;
    }
    if (kernel_overridden) matmul_set_kernel(kernel);

    bench_metadata_t metadata;
    bench_collect_metadata(&metadata);
    bench_print_metadata(&metadata);
    printf("CPU matmul kernel: %s (tile %zu)\n\n",
           matmul_kernel_name(matmul_get_kernel()), matmul_tile_size());
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
        bench_matmul_tier_t tiers[sizeof(configs) / sizeof(configs[0])];
        for (size_t i = 0; i < num_configs; i++) {
            tiers[i].key = configs[i].key;
            tiers[i].name = configs[i].name;
            tiers[i].vocab_size = configs[i].vocab_size;
            tiers[i].embedding_dim = configs[i].embedding_dim;
            tiers[i].max_seq_len = configs[i].max_seq_len;
        }
        bench_matmul_options_t options;
        bench_matmul_default_options(&options);
        options.sweep = sweep;
        options.quick = quick;
        options.repeats = repeats;
        if (matmul_csv_path) options.csv_path = matmul_csv_path;
        return bench_matmul_run(tiers, num_configs, selected_tier, &options,
                                timestamp, &metadata);
    }

    /* Opened only for full-model runs, so a matmul-only session never leaves
     * a header-only bench_results_v2.csv behind. */
    const char *csv_path = "bench_results_v2.csv";
    int csv_existed = (access(csv_path, F_OK) == 0);
    FILE *csv = fopen(csv_path, "a");
    if (csv && !csv_existed) {
        fprintf(csv, "timestamp,config,vocab_size,embedding_dim,num_heads,num_layers,max_seq_len,"
                      "optimizer,mode,param_count,inference_ms_per_token,inference_tokens_per_sec,"
                      "prefill_ms_per_token,growing_decode_ms_per_token,sliding_decode_ms_per_token,"
                      "training_ms_per_step,training_steps_per_sec,rss_delta_mb,"
                      BENCH_METADATA_CSV_HEADER "\n");
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
