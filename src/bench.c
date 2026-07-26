/*
 * Standalone benchmark tool: memory footprint and throughput (inference +
 * training) across a few representative model sizes, single-threaded by
 * default (OMP=1 builds will parallelize the matmul/attention internals,
 * but the driver loop itself makes no threading decisions) so the numbers
 * reflect what a single low-end CPU core can do.
 *
 * Deliberately its own small file/binary (bench.out) rather than a mode
 * bolted onto main.c/cli.c - it links directly against the model modules
 * and has nothing to do with training-run orchestration or the tokenizer.
 *
 * Build:  make bench      (or: make bench OMP=1 CC=gcc)
 * Run:    ./bench.out
 */

#include "include/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

typedef struct {
    const char *name;
    size_t vocab_size, embedding_dim, num_heads, num_layers, max_seq_len;
} bench_config_t;

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
static void bench_inference(neural_model_t *model, const bench_config_t *cfg, int iters) {
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
}

static void bench_training(neural_model_t *model, const bench_config_t *cfg, int iters) {
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
}

static void run_config(const bench_config_t *cfg, optimizer_type_t optimizer, int infer_iters, int train_iters) {
    neural_model_t model = {0};
    long rss_before = peak_rss_kb();

    if (model_new(&model, cfg->vocab_size, cfg->embedding_dim, cfg->num_heads,
                  cfg->num_layers, cfg->max_seq_len) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed for config %s\n", cfg->name);
        return;
    }
    model.optimizer_type = optimizer;

    printf("--- %s (vocab=%zu emb=%zu heads=%zu layers=%zu max_seq=%zu) optimizer=%s ---\n",
           cfg->name, cfg->vocab_size, cfg->embedding_dim, cfg->num_heads, cfg->num_layers,
           cfg->max_seq_len, optimizer == OPTIMIZER_ADAM ? "adam" : "sgd");

    size_t param_floats = model.total_param_count;
    printf("  parameters: %zu (%.2f MB weights only)\n",
           param_floats, param_floats * sizeof(float) / (1024.0 * 1024.0));

    bench_inference(&model, cfg, infer_iters);
    bench_training(&model, cfg, train_iters);

    long rss_after = peak_rss_kb();
    printf("  process peak RSS after this config: %.2f MB (delta from process start: %.2f MB)\n",
           rss_after / 1024.0, (rss_after - rss_before) / 1024.0);
    printf("\n");

    model_free(&model);
}

int main(void) {
#ifdef _OPENMP
    printf("Built with OpenMP (OMP=1) - matmul/attention parallelize across cores internally.\n\n");
#else
    printf("Built WITHOUT OpenMP - single-threaded throughout (rebuild with `make bench OMP=1 CC=gcc` to compare).\n\n");
#endif

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
        run_config(&configs[i], OPTIMIZER_SGD, 50, 20);
        run_config(&configs[i], OPTIMIZER_ADAM, 50, 20);
    }

    return 0;
}
