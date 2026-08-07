/*
 * Model lifecycle: allocation and teardown. Every trainable parameter is
 * carved out of one flat params buffer (and a matching flat grads buffer,
 * plus adam_m/adam_v moment buffers) - the named fields on neural_model_t
 * and on each transformer_layer_t are just views into these, assigned via
 * a running cursor below. This is what lets model_zero_gradients, the
 * optimizer step, and weight serialization each be a single loop/fwrite
 * over total_param_count instead of one line per named field (see
 * optimizer.c and serialization.c).
 *
 * The actual math (forward/backward, training step) lives in
 * tensor_ops.c, transformer.c, training.c, optimizer.c - see model.h for
 * the full module map.
 */

#include "core/model.h"
#include "backends/gpu/gpu_matmul.h"
#include "common/debug.h"
#include <stdlib.h>
#include <string.h>

static int checked_multiply(size_t left, size_t right, size_t *out) {
    if (left != 0 && right > SIZE_MAX / left) return 0;
    *out = left * right;
    return 1;
}

static int checked_add(size_t left, size_t right, size_t *out) {
    if (right > SIZE_MAX - left) return 0;
    *out = left + right;
    return 1;
}

/* Carves one transformer_layer_t's worth of views out of the params/grads
 * cursors. model_new validates the complete count before calling this. */
static void layout_layer(transformer_layer_t *layer, float **pc, float **gc,
                          size_t embedding_dim) {
    size_t ffn_dim = embedding_dim * 4;
    size_t emb2 = embedding_dim * embedding_dim;

    layer->W_q = *pc; layer->W_q_grad = *gc; *pc += emb2; *gc += emb2;
    layer->W_k = *pc; layer->W_k_grad = *gc; *pc += emb2; *gc += emb2;
    layer->W_v = *pc; layer->W_v_grad = *gc; *pc += emb2; *gc += emb2;
    layer->W_o = *pc; layer->W_o_grad = *gc; *pc += emb2; *gc += emb2;

    layer->ln_gamma_attn = *pc; layer->ln_gamma_attn_grad = *gc; *pc += embedding_dim; *gc += embedding_dim;
    layer->ln_beta_attn = *pc; layer->ln_beta_attn_grad = *gc; *pc += embedding_dim; *gc += embedding_dim;

    layer->W_ff1 = *pc; layer->W_ff1_grad = *gc; *pc += embedding_dim * ffn_dim; *gc += embedding_dim * ffn_dim;
    layer->b_ff1 = *pc; layer->b_ff1_grad = *gc; *pc += ffn_dim; *gc += ffn_dim;
    layer->W_ff2 = *pc; layer->W_ff2_grad = *gc; *pc += ffn_dim * embedding_dim; *gc += ffn_dim * embedding_dim;
    layer->b_ff2 = *pc; layer->b_ff2_grad = *gc; *pc += embedding_dim; *gc += embedding_dim;

    layer->ln_gamma_ffn = *pc; layer->ln_gamma_ffn_grad = *gc; *pc += embedding_dim; *gc += embedding_dim;
    layer->ln_beta_ffn = *pc; layer->ln_beta_ffn_grad = *gc; *pc += embedding_dim; *gc += embedding_dim;
}

/* Initialize a new model: allocate every weight/gradient/cache/scratch
 * buffer up front (all checked for failure, with model_free()-based
 * rollback), so nothing in the forward/train/predict hot path ever
 * mallocs. `model` is memset to zero first so model_free() is always safe
 * to call as cleanup, no matter how far allocation got before failing. */
model_errors_t model_new(neural_model_t *model,
                         size_t vocab_size,
                         size_t embedding_dim,
                         size_t num_heads,
                         size_t num_layers,
                         size_t max_seq_len) {
    if (model == NULL || vocab_size == 0 || embedding_dim == 0 ||
        num_heads == 0 || embedding_dim % num_heads != 0 ||
        num_layers == 0 || max_seq_len == 0) {
        return MODEL_INVALID_INPUT;
    }

    size_t ffn_dim, emb2, layer_params, layer_square_term, layer_linear_term;
    size_t vocab_emb, global_count, doubled_vocab_emb, layered_params;
    size_t total_param_count, seq_emb, ffn_cache, max_seq_squared, probs_cache;
    size_t hidden_pointer_count;
    if (!checked_multiply(embedding_dim, 4, &ffn_dim) ||
        !checked_multiply(embedding_dim, embedding_dim, &emb2) ||
        !checked_multiply(emb2, 12, &layer_square_term) ||
        !checked_multiply(embedding_dim, 9, &layer_linear_term) ||
        !checked_add(layer_square_term, layer_linear_term, &layer_params) ||
        !checked_multiply(vocab_size, embedding_dim, &vocab_emb) ||
        !checked_multiply(vocab_emb, 2, &doubled_vocab_emb) ||
        !checked_add(doubled_vocab_emb, vocab_size, &global_count) ||
        !checked_multiply(num_layers, layer_params, &layered_params) ||
        !checked_add(global_count, layered_params, &total_param_count) ||
        !checked_multiply(max_seq_len, embedding_dim, &seq_emb) ||
        !checked_multiply(max_seq_len, ffn_dim, &ffn_cache) ||
        !checked_multiply(max_seq_len, max_seq_len, &max_seq_squared) ||
        !checked_multiply(num_heads, max_seq_squared, &probs_cache) ||
        !checked_add(num_layers, 1, &hidden_pointer_count) ||
        total_param_count > SIZE_MAX / sizeof(float) ||
        seq_emb > SIZE_MAX / sizeof(float) ||
        ffn_cache > SIZE_MAX / sizeof(float) ||
        probs_cache > SIZE_MAX / sizeof(float) ||
        vocab_size > SIZE_MAX / sizeof(float) ||
        max_seq_len > SIZE_MAX / sizeof(float) ||
        num_layers > SIZE_MAX / sizeof(float *) ||
        hidden_pointer_count > SIZE_MAX / sizeof(float *) ||
        num_layers > SIZE_MAX / sizeof(transformer_layer_t)) {
        return MODEL_INVALID_INPUT;
    }

    memset(model, 0, sizeof(neural_model_t));

    DEBUG_PRINT("Initializing neural model: vocab_size=%zu, embedding_dim=%zu, num_heads=%zu, num_layers=%zu\n",
                vocab_size, embedding_dim, num_heads, num_layers);

    model->vocab_size = vocab_size;
    model->embedding_dim = embedding_dim;
    model->num_heads = num_heads;
    model->num_layers = num_layers;
    model->max_seq_len = max_seq_len;
    model->learning_rate = 0.001f;

    /* --- Optimizer/schedule/dropout defaults. Override any of these on
     * the model directly before training starts, same as learning_rate. --- */
    model->optimizer_type = OPTIMIZER_ADAM;
    model->adam_beta1 = 0.9f;
    model->adam_beta2 = 0.999f;
    model->adam_eps = 1e-8f;
    model->weight_decay = 0.01f;
    model->grad_clip_norm = 1.0f;
    model->adam_t = 0;
    model->warmup_steps = 0;
    model->total_steps = 0;   /* 0 = schedule disabled until the caller configures it */
    model->base_lr = model->learning_rate;
    model->min_lr = 0.0f;
    model->dropout_rate = 0.0f;
    model->is_training = 0;
    model->rng_state = UINT64_C(0x9e3779b97f4a7c15);
    model->use_gpu = 0;
    model->use_scalar_matmul = 0;

    /* --- Flat parameter storage: one buffer for every trainable weight,
     * one for every gradient. Every named pointer below (token_embeddings,
     * output_*, every layer's fields) is a view into these, assigned by
     * layout_layer()/the cursor arithmetic below - none of them are
     * individually freed.
     *
     * adam_m/adam_v (Adam's moment estimates) are deliberately NOT
     * allocated here - on memory-constrained targets, a caller using plain
     * SGD (model->optimizer_type = OPTIMIZER_SGD) shouldn't pay for two
     * more full-size float buffers it'll never touch. optimizer.c
     * allocates them lazily, the first time an Adam step actually runs. */
    model->total_param_count = total_param_count;

    model->params = malloc(model->total_param_count * sizeof(float));
    model->grads = calloc(model->total_param_count, sizeof(float));
    model->position_embeddings = malloc(max_seq_len * embedding_dim * sizeof(float));

    if (!model->params || !model->grads || !model->position_embeddings) {
        model_free(model);
        return MODEL_ALLOCATION_FAILURE;
    }

    model->layers = malloc(num_layers * sizeof(transformer_layer_t));
    if (!model->layers) {
        model_free(model);
        return MODEL_ALLOCATION_FAILURE;
    }
    memset(model->layers, 0, num_layers * sizeof(transformer_layer_t));

    float *pc = model->params;
    float *gc = model->grads;

    model->token_embeddings = pc; model->token_embeddings_grad = gc;
    pc += vocab_size * embedding_dim; gc += vocab_size * embedding_dim;
    model->output_projection = pc; model->output_projection_grad = gc;
    pc += embedding_dim * vocab_size; gc += embedding_dim * vocab_size;
    model->output_bias = pc; model->output_bias_grad = gc;
    pc += vocab_size; gc += vocab_size;

    for (size_t l = 0; l < num_layers; l++) {
        layout_layer(&model->layers[l], &pc, &gc, embedding_dim);
    }

    for (size_t l = 0; l < num_layers; l++) {
        transformer_layer_t *layer = &model->layers[l];
        xavier_init(layer->W_q, embedding_dim * embedding_dim, embedding_dim, embedding_dim);
        xavier_init(layer->W_k, embedding_dim * embedding_dim, embedding_dim, embedding_dim);
        xavier_init(layer->W_v, embedding_dim * embedding_dim, embedding_dim, embedding_dim);
        xavier_init(layer->W_o, embedding_dim * embedding_dim, embedding_dim, embedding_dim);
        xavier_init(layer->W_ff1, embedding_dim * ffn_dim, embedding_dim, ffn_dim);
        xavier_init(layer->W_ff2, ffn_dim * embedding_dim, ffn_dim, embedding_dim);

        memset(layer->b_ff1, 0, ffn_dim * sizeof(float));
        memset(layer->b_ff2, 0, embedding_dim * sizeof(float));
        for (size_t d = 0; d < embedding_dim; d++) {
            layer->ln_gamma_attn[d] = 1.0f;
            layer->ln_beta_attn[d] = 0.0f;
            layer->ln_gamma_ffn[d] = 1.0f;
            layer->ln_beta_ffn[d] = 0.0f;
        }
    }

    xavier_init(model->token_embeddings, vocab_size * embedding_dim, 1, embedding_dim);
    xavier_init(model->output_projection, embedding_dim * vocab_size, embedding_dim, vocab_size);
    memset(model->output_bias, 0, vocab_size * sizeof(float));

    if (compute_positional_encoding(model->position_embeddings, max_seq_len, embedding_dim) != 0) {
        model_free(model);
        return MODEL_ALLOCATION_FAILURE;
    }

    /* --- Forward-pass activation cache (for backward). Sized for
     * max_seq_len - the worst case - and reused for every training step
     * regardless of that step's actual seq_len. --- */
    model->cache_hidden = malloc((num_layers + 1) * sizeof(float *));
    model->cache_Q = malloc(num_layers * sizeof(float *));
    model->cache_K = malloc(num_layers * sizeof(float *));
    model->cache_V = malloc(num_layers * sizeof(float *));
    model->cache_probs = malloc(num_layers * sizeof(float *));
    model->cache_attn_concat = malloc(num_layers * sizeof(float *));
    model->cache_attn_ln_out = malloc(num_layers * sizeof(float *));
    model->cache_attn_xhat = malloc(num_layers * sizeof(float *));
    model->cache_attn_std = malloc(num_layers * sizeof(float *));
    model->cache_ff_hidden = malloc(num_layers * sizeof(float *));
    model->cache_ffn_xhat = malloc(num_layers * sizeof(float *));
    model->cache_ffn_std = malloc(num_layers * sizeof(float *));
    model->cache_attn_dropout_mask = malloc(num_layers * sizeof(float *));
    model->cache_ffn_dropout_mask = malloc(num_layers * sizeof(float *));

    if (!model->cache_hidden || !model->cache_Q || !model->cache_K || !model->cache_V ||
        !model->cache_probs || !model->cache_attn_concat || !model->cache_attn_ln_out ||
        !model->cache_attn_xhat || !model->cache_attn_std || !model->cache_ff_hidden ||
        !model->cache_ffn_xhat || !model->cache_ffn_std ||
        !model->cache_attn_dropout_mask || !model->cache_ffn_dropout_mask) {
        model_free(model);
        return MODEL_ALLOCATION_FAILURE;
    }
    /* Zero every pointer-array up front so that if a later entry in the
     * fill-in loops below fails to allocate, model_free()'s cleanup can
     * safely free(NULL) every not-yet-reached entry. */
    memset(model->cache_hidden, 0, (num_layers + 1) * sizeof(float *));
    memset(model->cache_Q, 0, num_layers * sizeof(float *));
    memset(model->cache_K, 0, num_layers * sizeof(float *));
    memset(model->cache_V, 0, num_layers * sizeof(float *));
    memset(model->cache_probs, 0, num_layers * sizeof(float *));
    memset(model->cache_attn_concat, 0, num_layers * sizeof(float *));
    memset(model->cache_attn_ln_out, 0, num_layers * sizeof(float *));
    memset(model->cache_attn_xhat, 0, num_layers * sizeof(float *));
    memset(model->cache_attn_std, 0, num_layers * sizeof(float *));
    memset(model->cache_ff_hidden, 0, num_layers * sizeof(float *));
    memset(model->cache_ffn_xhat, 0, num_layers * sizeof(float *));
    memset(model->cache_ffn_std, 0, num_layers * sizeof(float *));
    memset(model->cache_attn_dropout_mask, 0, num_layers * sizeof(float *));
    memset(model->cache_ffn_dropout_mask, 0, num_layers * sizeof(float *));

    for (size_t l = 0; l <= num_layers; l++) {
        model->cache_hidden[l] = malloc(seq_emb * sizeof(float));
        if (!model->cache_hidden[l]) {
            model_free(model);
            return MODEL_ALLOCATION_FAILURE;
        }
    }
    for (size_t l = 0; l < num_layers; l++) {
        model->cache_Q[l] = malloc(seq_emb * sizeof(float));
        model->cache_K[l] = malloc(seq_emb * sizeof(float));
        model->cache_V[l] = malloc(seq_emb * sizeof(float));
        model->cache_probs[l] = malloc(probs_cache * sizeof(float));
        model->cache_attn_concat[l] = malloc(seq_emb * sizeof(float));
        model->cache_attn_ln_out[l] = malloc(seq_emb * sizeof(float));
        model->cache_attn_xhat[l] = malloc(seq_emb * sizeof(float));
        model->cache_attn_std[l] = malloc(max_seq_len * sizeof(float));
        model->cache_ff_hidden[l] = malloc(ffn_cache * sizeof(float));
        model->cache_ffn_xhat[l] = malloc(seq_emb * sizeof(float));
        model->cache_ffn_std[l] = malloc(max_seq_len * sizeof(float));
        model->cache_attn_dropout_mask[l] = malloc(seq_emb * sizeof(float));
        model->cache_ffn_dropout_mask[l] = malloc(seq_emb * sizeof(float));

        if (!model->cache_Q[l] || !model->cache_K[l] || !model->cache_V[l] || !model->cache_probs[l] ||
            !model->cache_attn_concat[l] || !model->cache_attn_ln_out[l] || !model->cache_attn_xhat[l] ||
            !model->cache_attn_std[l] || !model->cache_ff_hidden[l] || !model->cache_ffn_xhat[l] ||
            !model->cache_ffn_std[l] || !model->cache_attn_dropout_mask[l] || !model->cache_ffn_dropout_mask[l]) {
            model_free(model);
            return MODEL_ALLOCATION_FAILURE;
        }
    }

    /* --- Reusable forward/backward scratch (single instance each) --- */
    model->ws_fwd_attn_raw = malloc(seq_emb * sizeof(float));
    model->ws_fwd_ff_raw = malloc(seq_emb * sizeof(float));
    model->ws_dhidden_in = malloc(seq_emb * sizeof(float));
    model->ws_dhidden_out = malloc(seq_emb * sizeof(float));
    model->ws_d_s2 = malloc(seq_emb * sizeof(float));
    model->ws_d_x1_total = malloc(seq_emb * sizeof(float));
    model->ws_d_s1 = malloc(seq_emb * sizeof(float));
    model->ws_d_ff_hidden = malloc(ffn_cache * sizeof(float));
    model->ws_d_attn_concat = malloc(seq_emb * sizeof(float));
    model->ws_d_scores = malloc(num_heads * max_seq_len * max_seq_len * sizeof(float));
    model->ws_d_Q = malloc(seq_emb * sizeof(float));
    model->ws_d_K = malloc(seq_emb * sizeof(float));
    model->ws_d_V = malloc(seq_emb * sizeof(float));
    model->ws_d_attn_dropout = malloc(seq_emb * sizeof(float));
    model->ws_d_ffn_dropout = malloc(seq_emb * sizeof(float));
    model->ws_logits = malloc(vocab_size * sizeof(float));
    model->ws_grad_logits = malloc(vocab_size * sizeof(float));
    /* All-position head (core/lm_head.c). max_seq_len rows of vocab_size,
     * so this is checked for overflow rather than trusting the product:
     * both factors are caller-supplied and their product is the largest
     * allocation the model makes at a wide vocabulary. */
    if (max_seq_len > SIZE_MAX / vocab_size ||
        max_seq_len * vocab_size > SIZE_MAX / sizeof(float)) {
        model_free(model);
        return MODEL_INVALID_INPUT;
    }
    size_t logits_all = max_seq_len * vocab_size;
    model->ws_logits_all = malloc(logits_all * sizeof(float));
    model->ws_grad_logits_all = malloc(logits_all * sizeof(float));

    if (!model->ws_fwd_attn_raw || !model->ws_fwd_ff_raw || !model->ws_dhidden_in || !model->ws_dhidden_out ||
        !model->ws_d_s2 || !model->ws_d_x1_total || !model->ws_d_s1 || !model->ws_d_ff_hidden ||
        !model->ws_d_attn_concat || !model->ws_d_scores || !model->ws_d_Q || !model->ws_d_K || !model->ws_d_V ||
        !model->ws_d_attn_dropout || !model->ws_d_ffn_dropout || !model->ws_logits || !model->ws_grad_logits ||
        !model->ws_logits_all || !model->ws_grad_logits_all) {
        model_free(model);
        return MODEL_ALLOCATION_FAILURE;
    }

    /* --- Learning metrics --- */
    model->metrics.history_capacity = 1000;
    model->metrics.loss_history = malloc(1000 * sizeof(float));
    if (!model->metrics.loss_history) {
        model_free(model);
        return MODEL_ALLOCATION_FAILURE;
    }
    model->metrics.history_size = 0;
    model->metrics.best_loss = 1e9f;
    model->metrics.worst_loss = 0.0f;
    model->metrics.avg_loss = 0.0f;
    model->metrics.learning_rate = model->learning_rate;
    model->metrics.initial_learning_rate = model->learning_rate;
    model->metrics.steps_without_improvement = 0;

    DEBUG_PRINT("Neural model initialized successfully\n");

    /* A freed model's params buffer can be reused by malloc for a
     * *different* model at the same address (see model_free below) -
     * invalidating here too means gpu_matmul.c's weight cache never
     * mistakes this fresh model's weights for stale cached bytes left
     * over from whatever previously occupied this memory. */
    gpu_matmul_invalidate_weights();

    return MODEL_SUCCESS;
}

void model_seed_rng(neural_model_t *model, uint64_t seed) {
    if (!model) return;
    model->rng_state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
}

/* Free all model resources */
void model_free(neural_model_t *model) {
    if (!model) return;

    /* See model_new: this pointer's params buffer is about to be freed
     * and may be handed back out by malloc for a different model later -
     * bump the generation so gpu_matmul.c's weight cache re-uploads
     * rather than trusting a coincidentally-matching pointer. */
    gpu_matmul_invalidate_weights();

    free(model->params);
    free(model->grads);
    free(model->adam_m);
    free(model->adam_v);
    free(model->position_embeddings);
    free(model->layers); /* just the struct array - its float* fields are views into params/grads, not owners */

    if (model->cache_hidden) {
        for (size_t l = 0; l <= model->num_layers; l++) free(model->cache_hidden[l]);
        free(model->cache_hidden);
    }
#define FREE_LAYER_CACHE(arr) \
    if (model->arr) { \
        for (size_t l = 0; l < model->num_layers; l++) free(model->arr[l]); \
        free(model->arr); \
    }
    FREE_LAYER_CACHE(cache_Q)
    FREE_LAYER_CACHE(cache_K)
    FREE_LAYER_CACHE(cache_V)
    FREE_LAYER_CACHE(cache_probs)
    FREE_LAYER_CACHE(cache_attn_concat)
    FREE_LAYER_CACHE(cache_attn_ln_out)
    FREE_LAYER_CACHE(cache_attn_xhat)
    FREE_LAYER_CACHE(cache_attn_std)
    FREE_LAYER_CACHE(cache_ff_hidden)
    FREE_LAYER_CACHE(cache_ffn_xhat)
    FREE_LAYER_CACHE(cache_ffn_std)
    FREE_LAYER_CACHE(cache_attn_dropout_mask)
    FREE_LAYER_CACHE(cache_ffn_dropout_mask)
#undef FREE_LAYER_CACHE

    free(model->ws_fwd_attn_raw);
    free(model->ws_fwd_ff_raw);
    free(model->ws_dhidden_in);
    free(model->ws_dhidden_out);
    free(model->ws_d_s2);
    free(model->ws_d_x1_total);
    free(model->ws_d_s1);
    free(model->ws_d_ff_hidden);
    free(model->ws_d_attn_concat);
    free(model->ws_d_scores);
    free(model->ws_d_Q);
    free(model->ws_d_K);
    free(model->ws_d_V);
    free(model->ws_d_attn_dropout);
    free(model->ws_d_ffn_dropout);
    free(model->ws_logits);
    free(model->ws_grad_logits);
    free(model->ws_logits_all);
    free(model->ws_grad_logits_all);

    free(model->metrics.loss_history);

    memset(model, 0, sizeof(neural_model_t));
}
