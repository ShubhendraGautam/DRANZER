/*
 * How much of the forward and backward pass is attention, and how much of that is
 * the part with no SIMD kernel.
 *
 * Why this exists, and why it runs before any kernel work
 * ------------------------------------------------------
 * core/transformer.c computes attention scores, the causal mask, the softmax, and
 * the value-weighted sum in scalar triple loops, while the projections on either
 * side of them go through tuned AVX-512/AVX2 kernels. At sequence length T the
 * scalar part is O(T^2 * d) and the projections are O(T * d^2), so the
 * unvectorized share grows with context length while the optimized share does
 * not.
 *
 * Every kernel result this project has published was measured against a model
 * that spends part of its time in that scalar code. Amdahl's law then sets a hard
 * ceiling on what further matmul work can buy, and nobody had measured where the
 * ceiling is. This tool measures it, so the ceiling on the next optimization is
 * known in advance rather than discovered after the work.
 *
 * The decomposition
 * -----------------
 * No instrumentation is added to the hot path - an added timer would change what
 * it measures. Instead three quantities are timed independently, each on the real
 * model through its public entry points:
 *
 *   forward total      model_forward_hidden()      the whole layer stack
 *   attention total    multihead_attention_forward() summed over layers
 *   attention matmuls  model_dispatch_matmul() at the same four shapes
 *                      attention uses (Q, K, V, and the output projection)
 *
 * and the scalar core is attention total minus attention matmuls. That
 * subtraction is the one soft step here: it assumes the projections cost the same
 * when called directly as they do inside attention, which is true up to cache
 * state. The residual is reported as a share rather than to three digits, and
 * the shape of its growth with T - which is what the conclusion rests on - does
 * not depend on the subtraction being exact.
 *
 * Ratios only, medians over replicates, same process. See
 * include/tools/timing_spread.h for why a single reading of a ratio is not
 * evidence.
 */

/* First, deliberately: it requests _POSIX_C_SOURCE for clock_gettime(), which
 * has to happen before any libc header is pulled in by the headers below. */
#include "tools/timing_spread.h"
#include "core/model.h"
#include "core/transformer.h"
#include "core/matmul_dispatch.h"
#include "core/cpu_features.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPLICATES 7

typedef struct {
    const char *name;
    size_t vocab, embedding_dim, heads, layers, seq_len;
} config_t;

/* Shapes chosen to span the question rather than to flatter it: the same width
 * at four context lengths isolates T, and the two wider rows show that the share
 * falls as d grows (the projections are O(d^2) and the scalar core is O(d)). */
static const config_t configs[] = {
    { "tiny, T=32",       260,  16, 2, 2,  32 },
    { "tiny, T=128",      260,  16, 2, 2, 128 },
    { "tiny, T=256",      260,  16, 2, 2, 256 },
    { "small, T=64",     1000,  64, 4, 4,  64 },
    { "small, T=256",    1000,  64, 4, 4, 256 },
    { "small, T=512",    1000,  64, 4, 4, 512 },
    { "medium, T=256",   2000, 128, 8, 6, 256 },
    { "medium, T=512",   2000, 128, 8, 6, 512 },
};

static neural_model_t model;
static uint32_t *tokens;
static size_t current_seq_len;
static size_t current_layers;
static float *d_attn_raw, *d_hidden;

static void op_forward_total(void) {
    model_forward_hidden(&model, tokens, current_seq_len);
}

static void op_attention_forward(void) {
    for (size_t l = 0; l < current_layers; l++) {
        multihead_attention_forward(&model, l, current_seq_len);
    }
}

/* The four matmuls attention performs per layer, at exactly the shapes it uses:
 * three (T x d) x (d x d) projections for Q, K, V and one for the output. */
static void op_attention_matmuls(void) {
    for (size_t l = 0; l < current_layers; l++) {
        transformer_layer_t *layer = &model.layers[l];
        size_t d = model.embedding_dim;
        model_dispatch_matmul(&model, model.cache_hidden[l], layer->W_q,
                              model.cache_Q[l], current_seq_len, d, d);
        model_dispatch_matmul(&model, model.cache_hidden[l], layer->W_k,
                              model.cache_K[l], current_seq_len, d, d);
        model_dispatch_matmul(&model, model.cache_hidden[l], layer->W_v,
                              model.cache_V[l], current_seq_len, d, d);
        model_dispatch_matmul(&model, model.cache_attn_concat[l], layer->W_o,
                              model.ws_fwd_attn_raw, current_seq_len, d, d);
    }
}

static void op_attention_backward(void) {
    for (size_t l = 0; l < current_layers; l++) {
        multihead_attention_backward(&model, l, current_seq_len, d_attn_raw,
                                     d_hidden);
    }
}

static void op_forward_and_backward(void) {
    static uint32_t targets[8192];
    for (size_t i = 0; i < current_seq_len && i < 8192; i++) {
        targets[i] = tokens[(i + 1) % current_seq_len];
    }
    model_zero_gradients(&model);
    model_accumulate_gradients_all(&model, tokens, targets, current_seq_len,
                                   NULL, NULL);
}

/* All four quantities timed inside ONE replicate, then medians taken across
 * replicates.
 *
 * Timing them in separate phases instead - measure the forward seven times, then
 * attention seven times - is wrong, and wrong in a way that is easy to miss: on a
 * machine that drifts between the phases, the two medians are not comparable, and
 * the first version of this tool reported attention as 141% of the forward pass
 * it is a part of. An impossible number is a lucky failure; the same error at 85%
 * would have been published. The order also alternates, so a machine warming up
 * across a replicate biases the four measurements in turn rather than always the
 * same one. */
typedef struct {
    double forward;
    double attention;
    double matmuls;
    double step;              /* forward + head + full backward */
    double attention_backward;
    /* The shares, formed per replicate and then summarized, rather than as a
     * ratio of two medians. Ratios of separately-summarized quantities have no
     * error bar and can land outside [0, 1] - this decomposition reads slightly
     * over 100% at shapes where attention is essentially the whole forward pass,
     * because the two measurements see different cache states. Reporting the
     * range makes that precision visible instead of implying three digits the
     * method does not have. */
    double attention_share_median, attention_share_low, attention_share_high;
    double scalar_share_median;
    double backward_share_median, backward_share_low, backward_share_high;
} timings_t;

static double median_of(double *values, size_t count) {
    qsort(values, count, sizeof(values[0]), timing_compare_double);
    return values[count / 2];
}

static timings_t measure_interleaved(void) {
    double forward[REPLICATES], attention[REPLICATES], matmuls[REPLICATES];
    double step[REPLICATES], attention_backward[REPLICATES];

    /* Warm-up of each, discarded. */
    op_forward_total();
    op_attention_forward();
    op_attention_matmuls();
    op_forward_and_backward();
    op_forward_total();          /* restore the cache the backward expects */
    op_attention_backward();

    for (size_t r = 0; r < REPLICATES; r++) {
        if (r % 2 == 0) {
            forward[r] = timing_once(op_forward_total);
            attention[r] = timing_once(op_attention_forward);
            matmuls[r] = timing_once(op_attention_matmuls);
            step[r] = timing_once(op_forward_and_backward);
            op_forward_total();
            attention_backward[r] = timing_once(op_attention_backward);
        } else {
            op_forward_total();
            attention_backward[r] = timing_once(op_attention_backward);
            step[r] = timing_once(op_forward_and_backward);
            matmuls[r] = timing_once(op_attention_matmuls);
            attention[r] = timing_once(op_attention_forward);
            forward[r] = timing_once(op_forward_total);
        }
    }

    /* Per-replicate shares, before anything is summarized. */
    double attention_share[REPLICATES], scalar_share[REPLICATES];
    double backward_share[REPLICATES];
    for (size_t r = 0; r < REPLICATES; r++) {
        attention_share[r] = forward[r] > 0.0 ? attention[r] / forward[r] : 0.0;
        double scalar = attention[r] - matmuls[r];
        if (scalar < 0.0) scalar = 0.0;
        scalar_share[r] = forward[r] > 0.0 ? scalar / forward[r] : 0.0;
        double backward_only = step[r] - forward[r];
        if (backward_only <= 0.0) backward_only = step[r];
        backward_share[r] = attention_backward[r] / backward_only;
    }

    timings_t out;
    out.forward = median_of(forward, REPLICATES);
    out.attention = median_of(attention, REPLICATES);
    out.matmuls = median_of(matmuls, REPLICATES);
    out.step = median_of(step, REPLICATES);
    out.attention_backward = median_of(attention_backward, REPLICATES);

    /* median_of() sorts in place, so read the extremes after sorting. */
    out.attention_share_median = median_of(attention_share, REPLICATES);
    out.attention_share_low = attention_share[0];
    out.attention_share_high = attention_share[REPLICATES - 1];
    out.scalar_share_median = median_of(scalar_share, REPLICATES);
    out.backward_share_median = median_of(backward_share, REPLICATES);
    out.backward_share_low = backward_share[0];
    out.backward_share_high = backward_share[REPLICATES - 1];
    return out;
}

int main(void) {
    printf("cpu: %s\n", cpu_features_summary());
    printf("attention's share of the forward pass, and how much of it has no "
           "SIMD kernel\n");
    printf("median of %d replicates per measurement, ratios only\n\n", REPLICATES);

    printf("%-14s %4s %5s %8s  %-15s %5s  %9s  %-15s\n", "config", "d", "T",
           "fwd (ms)", "attn share", "scalar", "bwd (ms)", "attn share");

    for (size_t c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const config_t *cfg = &configs[c];
        memset(&model, 0, sizeof(model));
        if (model_new_seeded(&model, cfg->vocab, cfg->embedding_dim, cfg->heads,
                             cfg->layers, cfg->seq_len, 7) != MODEL_SUCCESS) {
            fprintf(stderr, "model_new_seeded failed for %s\n", cfg->name);
            return 1;
        }
        model.dropout_rate = 0.0f;
        current_seq_len = cfg->seq_len;
        current_layers = cfg->layers;

        tokens = malloc(cfg->seq_len * sizeof(uint32_t));
        d_attn_raw = calloc(cfg->seq_len * cfg->embedding_dim, sizeof(float));
        d_hidden = calloc(cfg->seq_len * cfg->embedding_dim, sizeof(float));
        if (!tokens || !d_attn_raw || !d_hidden) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }
        for (size_t i = 0; i < cfg->seq_len; i++) {
            tokens[i] = (uint32_t)(i % cfg->vocab);
        }

        /* One forward first: the attention and backward measurements read the
         * activation cache this populates. */
        model_forward_hidden(&model, tokens, cfg->seq_len);

        timings_t t = measure_interleaved();
        double backward = t.step - t.forward;
        if (backward <= 0.0) backward = t.step;

        printf("%-14s %4zu %5zu %8.2f  %3.0f%% [%3.0f-%3.0f]  %3.0f%%  %9.2f  "
               "%3.0f%% [%3.0f-%3.0f]\n",
               cfg->name, cfg->embedding_dim, cfg->seq_len, t.forward * 1e3,
               100.0 * t.attention_share_median,
               100.0 * t.attention_share_low, 100.0 * t.attention_share_high,
               100.0 * t.scalar_share_median,
               backward * 1e3,
               100.0 * t.backward_share_median,
               100.0 * t.backward_share_low, 100.0 * t.backward_share_high);

        free(tokens);
        free(d_attn_raw);
        free(d_hidden);
        model_free(&model);
    }

    printf("\nShares are medians over replicates with the observed range beside\n");
    printf("them. A range touching 100%% means attention is essentially the whole\n");
    printf("pass at that shape, within this decomposition's precision - not that\n");
    printf("a part is larger than its whole.\n\n");
    printf("Read the scalar column against T at fixed d. That is the ceiling: no\n");
    printf("amount of matmul tuning touches it, and it is what a vectorized\n");
    printf("attention kernel would be competing for.\n");
    return 0;
}
