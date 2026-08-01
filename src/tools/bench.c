/*
 * Standalone benchmark tool: memory footprint and throughput (inference +
 * training) across a few representative model sizes, on CPU and - when a
 * usable CUDA GPU is present - also with model->use_gpu = 1 (see
 * transformer.c's dispatch_matmul()), so the two are directly comparable.
 * Single-threaded by default (OMP=1 builds parallelize the matmul/
 * attention internals, but the driver loop itself makes no threading
 * decisions) so the CPU numbers reflect what a single low-end core can do.
 *
 * Every run appends its results to bench_results.csv (gitignored, like
 * gpu_capability_cache/) so historical numbers accumulate for comparison
 * across code changes or machines, instead of only ever living in
 * whatever terminal happened to be open when the benchmark ran.
 *
 * Deliberately its own small file/binary (bench.out) rather than a mode
 * bolted onto main.c/cli.c - it links directly against the model modules
 * and has nothing to do with training-run orchestration or the tokenizer.
 *
 * Build:  make bench      (or: make bench OMP=1 CC=gcc)
 * Run:    ./bench.out
 */

#include "core/model.h"
#include "backends/gpu/gpu_matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    const char *name;
    size_t vocab_size, embedding_dim, num_heads, num_layers, max_seq_len;
} bench_config_t;

typedef struct {
    double inference_ms_per_token;
    double training_ms_per_step;
} bench_result_t;

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

/* Compare the old full-prefix decode path with incremental decoding over
 * the last few positions of a full context. Prompt prefill is excluded
 * from both timings: this isolates steady-state per-token decode cost,
 * which is exactly what the KV cache is intended to reduce. */
static void bench_kv_decode(neural_model_t *model, const bench_config_t *cfg) {
    size_t decode_steps = cfg->max_seq_len < 8 ? cfg->max_seq_len : 8;
    size_t prefill_len = cfg->max_seq_len - decode_steps;
    if (prefill_len == 0) {
        prefill_len = 1;
        decode_steps = cfg->max_seq_len - 1;
    }
    if (decode_steps == 0) return;

    uint32_t *tokens = malloc(cfg->max_seq_len * sizeof(uint32_t));
    float *logits = malloc(cfg->vocab_size * sizeof(float));
    if (!tokens || !logits) {
        free(tokens);
        free(logits);
        printf("  KV-cache decode benchmark: allocation failed\n");
        return;
    }
    for (size_t i = 0; i < cfg->max_seq_len; i++) {
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
        return;
    }
    for (size_t i = 0; i < prefill_len; i++) {
        model_forward_token(model, &cache, tokens[i], logits);
    }

    double cached_t0 = now_sec();
    for (size_t pos = prefill_len; pos < cfg->max_seq_len; pos++) {
        model_forward_token(model, &cache, tokens[pos], logits);
    }
    double cached_ms = 1000.0 * (now_sec() - cached_t0) / (double)decode_steps;

    printf("  autoregressive decode (last %zu positions, prefill excluded):\n", decode_steps);
    printf("    full-prefix %.3f ms/token   KV-cache %.3f ms/token   speedup %.2fx\n",
           full_ms, cached_ms, full_ms / cached_ms);

    model_kv_cache_free(&cache);
    free(tokens);
    free(logits);
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

static void csv_log(FILE *csv, const char *timestamp, const bench_config_t *cfg,
                     const char *optimizer_name, const char *mode, long param_floats,
                     const bench_result_t *r, long rss_delta_kb) {
    if (!csv) return;
    fprintf(csv, "%s,%s,%zu,%zu,%zu,%zu,%zu,%s,%s,%ld,%.4f,%.4f,%.4f,%.4f,%.2f\n",
            timestamp, cfg->name, cfg->vocab_size, cfg->embedding_dim, cfg->num_heads,
            cfg->num_layers, cfg->max_seq_len, optimizer_name, mode, param_floats,
            r->inference_ms_per_token, 1000.0 / r->inference_ms_per_token,
            r->training_ms_per_step, 1000.0 / r->training_ms_per_step,
            rss_delta_kb / 1024.0);
    fflush(csv);
}

static void run_config(const bench_config_t *cfg, optimizer_type_t optimizer, int use_gpu,
                        int infer_iters, int train_iters, FILE *csv, const char *timestamp) {
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

    const char *optimizer_name = (optimizer == OPTIMIZER_ADAM) ? "adam" : "sgd";
    printf("--- %s (vocab=%zu emb=%zu heads=%zu layers=%zu max_seq=%zu) optimizer=%s mode=%s ---\n",
           cfg->name, cfg->vocab_size, cfg->embedding_dim, cfg->num_heads, cfg->num_layers,
           cfg->max_seq_len, optimizer_name, use_gpu ? "GPU" : "CPU");

    size_t param_floats = model.total_param_count;
    printf("  parameters: %zu (%.2f MB weights only)\n",
           param_floats, param_floats * sizeof(float) / (1024.0 * 1024.0));

    bench_result_t r;
    r.inference_ms_per_token = bench_inference(&model, cfg, infer_iters);
    bench_kv_decode(&model, cfg);
    r.training_ms_per_step = bench_training(&model, cfg, train_iters);

    long rss_after = peak_rss_kb();
    printf("  process peak RSS after this config: %.2f MB (delta from process start: %.2f MB)\n",
           rss_after / 1024.0, (rss_after - rss_before) / 1024.0);
    printf("\n");

    csv_log(csv, timestamp, cfg, optimizer_name, use_gpu ? "gpu" : "cpu",
            (long)param_floats, &r, rss_after - rss_before);

    model_free(&model);
}

int main(void) {
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

    const char *csv_path = "bench_results.csv";
    int csv_existed = (access(csv_path, F_OK) == 0);
    FILE *csv = fopen(csv_path, "a");
    if (csv && !csv_existed) {
        fprintf(csv, "timestamp,config,vocab_size,embedding_dim,num_heads,num_layers,max_seq_len,"
                      "optimizer,mode,param_count,inference_ms_per_token,inference_tokens_per_sec,"
                      "training_ms_per_step,training_steps_per_sec,rss_delta_mb\n");
    }
    char timestamp[32];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", localtime(&now));

    bench_config_t configs[] = {
        /* Matches this project's current main.c defaults. */
        { "tiny (project default)", 257, 16, 2, 2, 32 },
        /* A step up: still CPU/RAM-friendly on something like a Raspberry Pi. */
        { "small",                 1000, 64, 4, 4, 64 },
        /* Rough "GPT-2 small"-scale embedding/layer count, kept at a modest
         * vocab so it stays feasible to actually run in this benchmark. */
        { "medium",                4000, 256, 8, 6, 128 },
    };
    size_t num_configs = sizeof(configs) / sizeof(configs[0]);

    for (size_t i = 0; i < num_configs; i++) {
        run_config(&configs[i], OPTIMIZER_ADAM, 0, 50, 20, csv, timestamp);
        /* GPU runs use fewer iterations: gpu_matmul() currently does a
         * fresh alloc/upload/launch/download/free on every single call
         * (see gpu_matmul.c) rather than reusing persistent device
         * buffers, so per-call overhead is real and this keeps total
         * benchmark time bounded while still measuring that overhead
         * honestly rather than hiding it. */
        if (gpu_available) {
            run_config(&configs[i], OPTIMIZER_ADAM, 1, 10, 5, csv, timestamp);
        }
    }

    if (csv) {
        printf("Results appended to %s\n", csv_path);
        fclose(csv);
    }

    return 0;
}
