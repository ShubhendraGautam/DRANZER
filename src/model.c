/*
 * Neural model implementation: stacked multi-head attention transformer
 * blocks with full backpropagation (attention, FFN, layer norm, token
 * embeddings) and plain SGD training. position_embeddings are fixed
 * (sinusoidal) and never trained.
 */

#include "include/model.h"
#include "include/debug.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Xavier initialization for weight matrices */
static void xavier_init(float *weights, size_t size, size_t fan_in, size_t fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    for (size_t i = 0; i < size; i++) {
        weights[i] = (rand() / (float)RAND_MAX) * 2.0f * limit - limit;
    }
}

/* Softmax implementation - OPTIMIZED with better numerical stability */
static void softmax(float *values, size_t size) {
    if (size == 0) return;

    float max_val = values[0];
    for (size_t i = 1; i < size; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float exp_val = expf(values[i] - max_val);
        values[i] = exp_val;
        sum += exp_val;
    }

    if (sum > 0) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < size; i++) {
            values[i] *= inv_sum;
        }
    }
}

/* Backprop through one row of softmax. `probs` is the cached forward
 * output (read-only). `dL_dprobs` is overwritten in place with dL/dscores:
 * safe because the row's dot product is computed from the original values
 * before anything in the row is overwritten. */
static void softmax_backward(float *probs, float *dL_dprobs, size_t size) {
    float dot = 0.0f;
    for (size_t j = 0; j < size; j++) {
        dot += dL_dprobs[j] * probs[j];
    }
    for (size_t j = 0; j < size; j++) {
        dL_dprobs[j] = probs[j] * (dL_dprobs[j] - dot);
    }
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* Derivative of ReLU. Safe to call with either the pre- or post-activation
 * value: post > 0 iff pre > 0, so the cached post-ReLU activations (which
 * is all we store) give the same result. */
static inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

/* Matrix multiplication with cache blocking for improved locality.
 * C (m x n) = A (m x k) @ B (k x n). Zeroes C first. */
static void matrix_multiply(float *A, float *B, float *C,
                           size_t m, size_t k, size_t n) {
    const size_t BLOCK_SIZE = 64;

    memset(C, 0, m * n * sizeof(float));

    for (size_t ii = 0; ii < m; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (size_t ll = 0; ll < k; ll += BLOCK_SIZE) {
                size_t i_limit = (ii + BLOCK_SIZE > m) ? m : ii + BLOCK_SIZE;
                size_t j_limit = (jj + BLOCK_SIZE > n) ? n : jj + BLOCK_SIZE;
                size_t l_limit = (ll + BLOCK_SIZE > k) ? k : ll + BLOCK_SIZE;

                for (size_t i = ii; i < i_limit; i++) {
                    for (size_t j = jj; j < j_limit; j++) {
                        float sum = C[i * n + j];
                        for (size_t l = ll; l < l_limit; l++) {
                            sum += A[i * k + l] * B[l * n + j];
                        }
                        C[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

/* Backprop through C = A @ B (A: m x k, B: k x n, C: m x n).
 * Accumulates (+=) into dA; caller must zero dA first for a fresh gradient. */
static void matmul_backward_input(float *dC, float *B, float *dA, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t l = 0; l < k; l++) {
            float sum = 0.0f;
            for (size_t j = 0; j < n; j++) {
                sum += dC[i * n + j] * B[l * n + j];
            }
            dA[i * k + l] += sum;
        }
    }
}

/* Backprop through C = A @ B (A: m x k, B: k x n, C: m x n).
 * Accumulates (+=) into dB; caller must zero dB first for a fresh gradient. */
static void matmul_backward_weight(float *A, float *dC, float *dB, size_t m, size_t k, size_t n) {
    for (size_t l = 0; l < k; l++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < m; i++) {
                sum += A[i * k + l] * dC[i * n + j];
            }
            dB[l * n + j] += sum;
        }
    }
}

/* Layer normalization - normalize + scale + shift (used by the public
 * layer_normalize() wrapper only; the training path uses the *_cached
 * variant below since it needs to keep xhat/std around for backward). */
static void layer_norm_internal(float *input, float *gamma, float *beta,
                                size_t size, float epsilon) {
    float mean = 0.0f;
    for (size_t i = 0; i < size; i++) mean += input[i];
    mean /= size;

    float variance = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;

    float std_dev = sqrtf(variance + epsilon);
    for (size_t i = 0; i < size; i++) {
        input[i] = gamma[i] * ((input[i] - mean) / std_dev) + beta[i];
    }
}

/* Forward layer norm over `seq_len` rows of `size` elements, caching xhat
 * and std_dev per row (needed by layer_norm_backward). */
static void layer_norm_forward_cached(float *pre_ln_sum, float *xhat_out, float *std_out,
                                       float *ln_out, float *gamma, float *beta,
                                       size_t seq_len, size_t size, float epsilon) {
    for (size_t i = 0; i < seq_len; i++) {
        float *row_in = &pre_ln_sum[i * size];
        float *row_xhat = &xhat_out[i * size];
        float *row_out = &ln_out[i * size];

        float mean = 0.0f;
        for (size_t d = 0; d < size; d++) mean += row_in[d];
        mean /= (float)size;

        float variance = 0.0f;
        for (size_t d = 0; d < size; d++) {
            float diff = row_in[d] - mean;
            variance += diff * diff;
        }
        variance /= (float)size;

        float std_dev = sqrtf(variance + epsilon);
        std_out[i] = std_dev;

        for (size_t d = 0; d < size; d++) {
            float xhat = (row_in[d] - mean) / std_dev;
            row_xhat[d] = xhat;
            row_out[d] = gamma[d] * xhat + beta[d];
        }
    }
}

/* Backprop through y = gamma*xhat+beta (LayerNorm), one row of `size`
 * elements at a time, seq_len rows. gamma/beta are shared across rows, so
 * their gradients accumulate (+=) across all seq_len positions - caller
 * must zero gamma_grad/beta_grad first for a fresh gradient. dL_dout and
 * dL_dinput may safely be the same buffer (each row is fully read before
 * it is written). */
static void layer_norm_backward(float *dL_dout, float *xhat, float *std,
                                 float *gamma, float *gamma_grad, float *beta_grad,
                                 float *dL_dinput, size_t seq_len, size_t size) {
    for (size_t i = 0; i < seq_len; i++) {
        float *row_dout = &dL_dout[i * size];
        float *row_xhat = &xhat[i * size];
        float *row_din = &dL_dinput[i * size];
        float std_dev = std[i];

        float dxhat[size]; /* VLA: `size` is embedding_dim, always small here */
        float mean_dxhat = 0.0f;
        float mean_dxhat_xhat = 0.0f;

        for (size_t d = 0; d < size; d++) {
            gamma_grad[d] += row_dout[d] * row_xhat[d];
            beta_grad[d] += row_dout[d];

            dxhat[d] = row_dout[d] * gamma[d];
            mean_dxhat += dxhat[d];
            mean_dxhat_xhat += dxhat[d] * row_xhat[d];
        }
        mean_dxhat /= (float)size;
        mean_dxhat_xhat /= (float)size;

        for (size_t d = 0; d < size; d++) {
            row_din[d] = (dxhat[d] - mean_dxhat - row_xhat[d] * mean_dxhat_xhat) / std_dev;
        }
    }
}

/* Positional encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d))
 * Returns 0 on success, -1 on allocation failure (caller must treat
 * pos_embed as uninitialized in that case - it is not touched). */
static int compute_positional_encoding(float *pos_embed, size_t seq_len, size_t embedding_dim) {
    float *dim_scales = malloc(embedding_dim * sizeof(float));
    if (dim_scales == NULL) return -1;

    for (size_t i = 0; i < embedding_dim; i++) {
        dim_scales[i] = 1.0f / powf(10000.0f, (2.0f * i) / embedding_dim);
    }

    for (size_t pos = 0; pos < seq_len; pos++) {
        for (size_t i = 0; i < embedding_dim; i++) {
            float angle = pos * dim_scales[i];
            if (i % 2 == 0) {
                pos_embed[pos * embedding_dim + i] = sinf(angle);
            } else {
                pos_embed[pos * embedding_dim + i] = cosf(angle);
            }
        }
    }

    free(dim_scales);
    return 0;
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
    if (model == NULL || num_heads == 0 || embedding_dim % num_heads != 0 ||
        num_layers == 0 || max_seq_len == 0) {
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

    size_t ffn_dim = embedding_dim * 4;

    /* --- Global (non-per-layer) parameters --- */
    model->token_embeddings = malloc(vocab_size * embedding_dim * sizeof(float));
    model->token_embeddings_grad = malloc(vocab_size * embedding_dim * sizeof(float));
    model->position_embeddings = malloc(max_seq_len * embedding_dim * sizeof(float));
    model->output_projection = malloc(embedding_dim * vocab_size * sizeof(float));
    model->output_projection_grad = malloc(embedding_dim * vocab_size * sizeof(float));
    model->output_bias = malloc(vocab_size * sizeof(float));
    model->output_bias_grad = malloc(vocab_size * sizeof(float));

    if (!model->token_embeddings || !model->token_embeddings_grad || !model->position_embeddings ||
        !model->output_projection || !model->output_projection_grad ||
        !model->output_bias || !model->output_bias_grad) {
        model_free(model);
        return MODEL_ALLOCATION_FAILURE;
    }

    /* --- Per-layer parameters --- */
    model->layers = malloc(num_layers * sizeof(transformer_layer_t));
    if (!model->layers) {
        model_free(model);
        return MODEL_ALLOCATION_FAILURE;
    }
    memset(model->layers, 0, num_layers * sizeof(transformer_layer_t));

    for (size_t l = 0; l < num_layers; l++) {
        transformer_layer_t *layer = &model->layers[l];
        layer->W_q = malloc(embedding_dim * embedding_dim * sizeof(float));
        layer->W_k = malloc(embedding_dim * embedding_dim * sizeof(float));
        layer->W_v = malloc(embedding_dim * embedding_dim * sizeof(float));
        layer->W_o = malloc(embedding_dim * embedding_dim * sizeof(float));
        layer->W_q_grad = malloc(embedding_dim * embedding_dim * sizeof(float));
        layer->W_k_grad = malloc(embedding_dim * embedding_dim * sizeof(float));
        layer->W_v_grad = malloc(embedding_dim * embedding_dim * sizeof(float));
        layer->W_o_grad = malloc(embedding_dim * embedding_dim * sizeof(float));
        layer->ln_gamma_attn = malloc(embedding_dim * sizeof(float));
        layer->ln_beta_attn = malloc(embedding_dim * sizeof(float));
        layer->ln_gamma_attn_grad = malloc(embedding_dim * sizeof(float));
        layer->ln_beta_attn_grad = malloc(embedding_dim * sizeof(float));
        layer->W_ff1 = malloc(embedding_dim * ffn_dim * sizeof(float));
        layer->b_ff1 = malloc(ffn_dim * sizeof(float));
        layer->W_ff2 = malloc(ffn_dim * embedding_dim * sizeof(float));
        layer->b_ff2 = malloc(embedding_dim * sizeof(float));
        layer->W_ff1_grad = malloc(embedding_dim * ffn_dim * sizeof(float));
        layer->b_ff1_grad = malloc(ffn_dim * sizeof(float));
        layer->W_ff2_grad = malloc(ffn_dim * embedding_dim * sizeof(float));
        layer->b_ff2_grad = malloc(embedding_dim * sizeof(float));
        layer->ln_gamma_ffn = malloc(embedding_dim * sizeof(float));
        layer->ln_beta_ffn = malloc(embedding_dim * sizeof(float));
        layer->ln_gamma_ffn_grad = malloc(embedding_dim * sizeof(float));
        layer->ln_beta_ffn_grad = malloc(embedding_dim * sizeof(float));

        if (!layer->W_q || !layer->W_k || !layer->W_v || !layer->W_o ||
            !layer->W_q_grad || !layer->W_k_grad || !layer->W_v_grad || !layer->W_o_grad ||
            !layer->ln_gamma_attn || !layer->ln_beta_attn ||
            !layer->ln_gamma_attn_grad || !layer->ln_beta_attn_grad ||
            !layer->W_ff1 || !layer->b_ff1 || !layer->W_ff2 || !layer->b_ff2 ||
            !layer->W_ff1_grad || !layer->b_ff1_grad || !layer->W_ff2_grad || !layer->b_ff2_grad ||
            !layer->ln_gamma_ffn || !layer->ln_beta_ffn ||
            !layer->ln_gamma_ffn_grad || !layer->ln_beta_ffn_grad) {
            model_free(model);
            return MODEL_ALLOCATION_FAILURE;
        }

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
    size_t seq_emb = max_seq_len * embedding_dim;
    size_t ffn_cache = max_seq_len * ffn_dim;
    size_t probs_cache = num_heads * max_seq_len * max_seq_len;

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

    if (!model->cache_hidden || !model->cache_Q || !model->cache_K || !model->cache_V ||
        !model->cache_probs || !model->cache_attn_concat || !model->cache_attn_ln_out ||
        !model->cache_attn_xhat || !model->cache_attn_std || !model->cache_ff_hidden ||
        !model->cache_ffn_xhat || !model->cache_ffn_std) {
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

        if (!model->cache_Q[l] || !model->cache_K[l] || !model->cache_V[l] || !model->cache_probs[l] ||
            !model->cache_attn_concat[l] || !model->cache_attn_ln_out[l] || !model->cache_attn_xhat[l] ||
            !model->cache_attn_std[l] || !model->cache_ff_hidden[l] || !model->cache_ffn_xhat[l] ||
            !model->cache_ffn_std[l]) {
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
    model->ws_d_scores = malloc(max_seq_len * max_seq_len * sizeof(float));
    model->ws_d_Q = malloc(seq_emb * sizeof(float));
    model->ws_d_K = malloc(seq_emb * sizeof(float));
    model->ws_d_V = malloc(seq_emb * sizeof(float));
    model->ws_logits = malloc(vocab_size * sizeof(float));
    model->ws_grad_logits = malloc(vocab_size * sizeof(float));

    if (!model->ws_fwd_attn_raw || !model->ws_fwd_ff_raw || !model->ws_dhidden_in || !model->ws_dhidden_out ||
        !model->ws_d_s2 || !model->ws_d_x1_total || !model->ws_d_s1 || !model->ws_d_ff_hidden ||
        !model->ws_d_attn_concat || !model->ws_d_scores || !model->ws_d_Q || !model->ws_d_K || !model->ws_d_V ||
        !model->ws_logits || !model->ws_grad_logits) {
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

    return MODEL_SUCCESS;
}

/* Multi-head self-attention forward pass for layer l. Reads
 * model->cache_hidden[l] as input, writes Q/K/V/probs/concat into that
 * layer's cache entries (needed by backward), and writes the final
 * (post-W_o, pre-residual) result into model->ws_fwd_attn_raw. */
static void multihead_attention_forward(neural_model_t *model, size_t l, size_t seq_len) {
    transformer_layer_t *layer = &model->layers[l];
    size_t embedding_dim = model->embedding_dim;
    size_t num_heads = model->num_heads;
    size_t head_dim = embedding_dim / num_heads;

    float *sequence = model->cache_hidden[l];
    float *Q = model->cache_Q[l];
    float *K = model->cache_K[l];
    float *V = model->cache_V[l];
    float *probs = model->cache_probs[l];
    float *concat = model->cache_attn_concat[l];

    matrix_multiply(sequence, layer->W_q, Q, seq_len, embedding_dim, embedding_dim);
    matrix_multiply(sequence, layer->W_k, K, seq_len, embedding_dim, embedding_dim);
    matrix_multiply(sequence, layer->W_v, V, seq_len, embedding_dim, embedding_dim);

    memset(concat, 0, seq_len * embedding_dim * sizeof(float));

    for (size_t head = 0; head < num_heads; head++) {
        float *head_probs = probs + head * seq_len * seq_len;

        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    size_t q_idx = i * embedding_dim + head * head_dim + d;
                    size_t k_idx = j * embedding_dim + head * head_dim + d;
                    score += Q[q_idx] * K[k_idx];
                }
                head_probs[i * seq_len + j] = score / sqrtf((float)head_dim);
            }
        }

        for (size_t i = 0; i < seq_len; i++) {
            softmax(&head_probs[i * seq_len], seq_len);
        }

        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    sum += head_probs[i * seq_len + j] * V[j * embedding_dim + head * head_dim + d];
                }
                concat[i * embedding_dim + head * head_dim + d] = sum;
            }
        }
    }

    matrix_multiply(concat, layer->W_o, model->ws_fwd_attn_raw, seq_len, embedding_dim, embedding_dim);
}

/* Backprop through multihead_attention_forward for layer l.
 * dL_dattn_raw: incoming gradient w.r.t. the attention block's raw output
 *   (post-W_o, pre-residual) - i.e. dL/d(model->ws_fwd_attn_raw).
 * dL_dhidden_accum: caller-owned buffer that this function ACCUMULATES
 *   into (+=) - the caller must have already seeded it with the
 *   residual-branch contribution to dL/dhidden[l] before calling.
 * Accumulates into layer->W_q_grad/W_k_grad/W_v_grad/W_o_grad - caller
 * must have zeroed them for a fresh gradient (model_zero_gradients does). */
static void multihead_attention_backward(neural_model_t *model, size_t l, size_t seq_len,
                                          float *dL_dattn_raw, float *dL_dhidden_accum) {
    transformer_layer_t *layer = &model->layers[l];
    size_t embedding_dim = model->embedding_dim;
    size_t num_heads = model->num_heads;
    size_t head_dim = embedding_dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    float *sequence = model->cache_hidden[l];
    float *Q = model->cache_Q[l];
    float *K = model->cache_K[l];
    float *V = model->cache_V[l];
    float *probs = model->cache_probs[l];
    float *concat = model->cache_attn_concat[l];

    matmul_backward_weight(concat, dL_dattn_raw, layer->W_o_grad, seq_len, embedding_dim, embedding_dim);

    float *d_concat = model->ws_d_attn_concat;
    memset(d_concat, 0, seq_len * embedding_dim * sizeof(float));
    matmul_backward_input(dL_dattn_raw, layer->W_o, d_concat, seq_len, embedding_dim, embedding_dim);

    float *dQ = model->ws_d_Q;
    float *dK = model->ws_d_K;
    float *dV = model->ws_d_V;
    memset(dQ, 0, seq_len * embedding_dim * sizeof(float));
    memset(dK, 0, seq_len * embedding_dim * sizeof(float));
    memset(dV, 0, seq_len * embedding_dim * sizeof(float));

    float *d_scores = model->ws_d_scores; /* seq_len x seq_len, reused per head */

    for (size_t head = 0; head < num_heads; head++) {
        float *head_probs = probs + head * seq_len * seq_len;

        /* dL/dprobs[i][j] from context[i] = sum_j probs[i][j]*V[j]; also
         * accumulate dL/dV while we're indexed by (i,j) anyway. */
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    dot += d_concat[i * embedding_dim + head * head_dim + d] *
                           V[j * embedding_dim + head * head_dim + d];
                }
                d_scores[i * seq_len + j] = dot; /* holds dL/dprobs[i][j] for now */
            }
        }
        for (size_t j = 0; j < seq_len; j++) {
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (size_t i = 0; i < seq_len; i++) {
                    sum += head_probs[i * seq_len + j] * d_concat[i * embedding_dim + head * head_dim + d];
                }
                dV[j * embedding_dim + head * head_dim + d] += sum;
            }
        }

        /* dL/dprobs -> dL/dscores via softmax backward, one row at a time */
        for (size_t i = 0; i < seq_len; i++) {
            softmax_backward(&head_probs[i * seq_len], &d_scores[i * seq_len], seq_len);
        }

        /* dL/dQ, dL/dK from scores = (Q . K) * scale */
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    sum += d_scores[i * seq_len + j] * K[j * embedding_dim + head * head_dim + d];
                }
                dQ[i * embedding_dim + head * head_dim + d] += sum * scale;
            }
        }
        for (size_t j = 0; j < seq_len; j++) {
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (size_t i = 0; i < seq_len; i++) {
                    sum += d_scores[i * seq_len + j] * Q[i * embedding_dim + head * head_dim + d];
                }
                dK[j * embedding_dim + head * head_dim + d] += sum * scale;
            }
        }
    }

    matmul_backward_weight(sequence, dQ, layer->W_q_grad, seq_len, embedding_dim, embedding_dim);
    matmul_backward_weight(sequence, dK, layer->W_k_grad, seq_len, embedding_dim, embedding_dim);
    matmul_backward_weight(sequence, dV, layer->W_v_grad, seq_len, embedding_dim, embedding_dim);

    matmul_backward_input(dQ, layer->W_q, dL_dhidden_accum, seq_len, embedding_dim, embedding_dim);
    matmul_backward_input(dK, layer->W_k, dL_dhidden_accum, seq_len, embedding_dim, embedding_dim);
    matmul_backward_input(dV, layer->W_v, dL_dhidden_accum, seq_len, embedding_dim, embedding_dim);
}

/* Forward pass through the model. Populates the activation cache as a side
 * effect (needed by model_train_step's backward pass); harmless for
 * standalone inference use. */
model_errors_t model_forward(neural_model_t *model,
                             uint32_t *token_ids,
                             size_t seq_len,
                             float *output_logits) {

    if (!model || !token_ids || !output_logits) {
        return MODEL_INVALID_INPUT;
    }

    if (seq_len == 0 || seq_len > model->max_seq_len) {
        return MODEL_INVALID_INPUT;
    }

    size_t embedding_dim = model->embedding_dim;
    size_t ffn_dim = embedding_dim * 4;
    const float epsilon = 1e-6f;

    DEBUG_PRINT("Model forward pass: seq_len=%zu, embedding_dim=%zu, num_layers=%zu\n",
                seq_len, embedding_dim, model->num_layers);

    /* 1. Embed tokens + positional encoding -> cache_hidden[0] */
    float *h0 = model->cache_hidden[0];
    for (size_t i = 0; i < seq_len; i++) {
        uint32_t token_id = token_ids[i];
        if (token_id >= model->vocab_size) token_id = 0; // OOV handling

        for (size_t d = 0; d < embedding_dim; d++) {
            h0[i * embedding_dim + d] =
                model->token_embeddings[token_id * embedding_dim + d] +
                model->position_embeddings[i * embedding_dim + d];
        }
    }

    /* 2. Stacked transformer blocks */
    for (size_t l = 0; l < model->num_layers; l++) {
        transformer_layer_t *layer = &model->layers[l];
        float *x_in = model->cache_hidden[l];

        /* Attention sub-block: attn -> residual -> LN */
        multihead_attention_forward(model, l, seq_len);
        for (size_t i = 0; i < seq_len * embedding_dim; i++) {
            model->ws_fwd_attn_raw[i] += x_in[i];
        }
        layer_norm_forward_cached(model->ws_fwd_attn_raw, model->cache_attn_xhat[l], model->cache_attn_std[l],
                                   model->cache_attn_ln_out[l], layer->ln_gamma_attn, layer->ln_beta_attn,
                                   seq_len, embedding_dim, epsilon);

        /* FFN sub-block: matmul -> bias -> ReLU -> matmul -> bias -> residual -> LN */
        float *x1 = model->cache_attn_ln_out[l];
        matrix_multiply(x1, layer->W_ff1, model->cache_ff_hidden[l], seq_len, embedding_dim, ffn_dim);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < ffn_dim; d++) {
                float *v = &model->cache_ff_hidden[l][i * ffn_dim + d];
                *v = relu(*v + layer->b_ff1[d]);
            }
        }
        matrix_multiply(model->cache_ff_hidden[l], layer->W_ff2, model->ws_fwd_ff_raw,
                       seq_len, ffn_dim, embedding_dim);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < embedding_dim; d++) {
                model->ws_fwd_ff_raw[i * embedding_dim + d] += layer->b_ff2[d] + x1[i * embedding_dim + d];
            }
        }
        layer_norm_forward_cached(model->ws_fwd_ff_raw, model->cache_ffn_xhat[l], model->cache_ffn_std[l],
                                   model->cache_hidden[l + 1], layer->ln_gamma_ffn, layer->ln_beta_ffn,
                                   seq_len, embedding_dim, epsilon);
    }

    /* 3. Output projection to vocabulary, from the last layer's last position */
    float *last_hidden = &model->cache_hidden[model->num_layers][(seq_len - 1) * embedding_dim];
    matrix_multiply(last_hidden, model->output_projection, output_logits, 1, embedding_dim, model->vocab_size);

    for (size_t i = 0; i < model->vocab_size; i++) {
        output_logits[i] += model->output_bias[i];
    }

    return MODEL_SUCCESS;
}

static void sgd_update(float *param, float *grad, size_t size, float lr) {
    for (size_t i = 0; i < size; i++) {
        param[i] -= lr * grad[i];
    }
}

/* Zero every gradient buffer before a fresh backward pass. token_embeddings
 * and output_projection are zeroed in full even though only a handful of
 * rows actually receive a nonzero gradient each step - fine at this
 * project's scale (vocab_size in the low hundreds); would need a sparse
 * update if vocab_size grew much larger. */
static void model_zero_gradients(neural_model_t *model) {
    size_t embedding_dim = model->embedding_dim;
    size_t ffn_dim = embedding_dim * 4;
    size_t vocab_size = model->vocab_size;

    memset(model->token_embeddings_grad, 0, vocab_size * embedding_dim * sizeof(float));
    memset(model->output_projection_grad, 0, embedding_dim * vocab_size * sizeof(float));
    memset(model->output_bias_grad, 0, vocab_size * sizeof(float));

    for (size_t l = 0; l < model->num_layers; l++) {
        transformer_layer_t *layer = &model->layers[l];
        memset(layer->W_q_grad, 0, embedding_dim * embedding_dim * sizeof(float));
        memset(layer->W_k_grad, 0, embedding_dim * embedding_dim * sizeof(float));
        memset(layer->W_v_grad, 0, embedding_dim * embedding_dim * sizeof(float));
        memset(layer->W_o_grad, 0, embedding_dim * embedding_dim * sizeof(float));
        memset(layer->ln_gamma_attn_grad, 0, embedding_dim * sizeof(float));
        memset(layer->ln_beta_attn_grad, 0, embedding_dim * sizeof(float));
        memset(layer->W_ff1_grad, 0, embedding_dim * ffn_dim * sizeof(float));
        memset(layer->b_ff1_grad, 0, ffn_dim * sizeof(float));
        memset(layer->W_ff2_grad, 0, ffn_dim * embedding_dim * sizeof(float));
        memset(layer->b_ff2_grad, 0, embedding_dim * sizeof(float));
        memset(layer->ln_gamma_ffn_grad, 0, embedding_dim * sizeof(float));
        memset(layer->ln_beta_ffn_grad, 0, embedding_dim * sizeof(float));
    }
}

/* Training step: forward pass (caching activations), cross-entropy loss,
 * full backpropagation through the output head, every transformer layer
 * (in reverse), and the token embeddings, then a plain SGD update of every
 * parameter that received a gradient. position_embeddings are fixed and
 * are never touched. */
model_errors_t model_train_step(neural_model_t *model,
                                uint32_t *token_ids,
                                uint32_t target_id,
                                size_t seq_len) {

    if (!model || !token_ids || target_id >= model->vocab_size) {
        return MODEL_INVALID_INPUT;
    }
    if (seq_len == 0 || seq_len > model->max_seq_len) {
        return MODEL_INVALID_INPUT;
    }

    size_t embedding_dim = model->embedding_dim;
    size_t ffn_dim = embedding_dim * 4;

    /* ---- Forward pass (populates the activation cache) ---- */
    float *logits = model->ws_logits;
    model_forward(model, token_ids, seq_len, logits);

    softmax(logits, model->vocab_size);
    float loss = -logf(fmaxf(logits[target_id], 1e-7f));
    model->current_loss = loss;

    DEBUG_PRINT("Training step: loss=%.4f, target_id=%u\n", loss, target_id);

    float *grad_logits = model->ws_grad_logits;
    memcpy(grad_logits, logits, model->vocab_size * sizeof(float));
    grad_logits[target_id] -= 1.0f; // Gradient for cross-entropy

    /* ---- Backward pass ---- */
    model_zero_gradients(model);

    float *last_hidden = &model->cache_hidden[model->num_layers][(seq_len - 1) * embedding_dim];

    for (size_t i = 0; i < model->vocab_size; i++) {
        model->output_bias_grad[i] += grad_logits[i];
    }
    matmul_backward_weight(last_hidden, grad_logits, model->output_projection_grad,
                            1, embedding_dim, model->vocab_size);

    /* Only the LAST sequence position feeds the output head, so dL/dhidden
     * is zero everywhere else - zero the whole buffer, then fill just the
     * last row. */
    memset(model->ws_dhidden_in, 0, seq_len * embedding_dim * sizeof(float));
    matmul_backward_input(grad_logits, model->output_projection,
                           &model->ws_dhidden_in[(seq_len - 1) * embedding_dim],
                           1, embedding_dim, model->vocab_size);

    for (size_t li = 0; li < model->num_layers; li++) {
        size_t l = model->num_layers - 1 - li;
        transformer_layer_t *layer = &model->layers[l];

        /* FFN-LN backward: dL/dx2 (ws_dhidden_in) -> dL/d(pre-ffn-ln-sum) (ws_d_s2) */
        layer_norm_backward(model->ws_dhidden_in, model->cache_ffn_xhat[l], model->cache_ffn_std[l],
                             layer->ln_gamma_ffn, layer->ln_gamma_ffn_grad, layer->ln_beta_ffn_grad,
                             model->ws_d_s2, seq_len, embedding_dim);

        /* ws_d_s2 is now dL/dffn_out (== dL/d(pre-b_ff2 matmul output),
         * since adding a bias just passes the gradient through) AND the
         * residual branch's contribution to dL/dx1. */
        for (size_t d = 0; d < embedding_dim; d++) {
            float sum = 0.0f;
            for (size_t i = 0; i < seq_len; i++) sum += model->ws_d_s2[i * embedding_dim + d];
            layer->b_ff2_grad[d] += sum;
        }
        matmul_backward_weight(model->cache_ff_hidden[l], model->ws_d_s2, layer->W_ff2_grad,
                                seq_len, ffn_dim, embedding_dim);

        memset(model->ws_d_ff_hidden, 0, seq_len * ffn_dim * sizeof(float));
        matmul_backward_input(model->ws_d_s2, layer->W_ff2, model->ws_d_ff_hidden,
                               seq_len, ffn_dim, embedding_dim);

        /* ReLU backward, in place, using the cached post-ReLU activations as the indicator */
        for (size_t idx = 0; idx < seq_len * ffn_dim; idx++) {
            model->ws_d_ff_hidden[idx] *= relu_derivative(model->cache_ff_hidden[l][idx]);
        }

        for (size_t d = 0; d < ffn_dim; d++) {
            float sum = 0.0f;
            for (size_t i = 0; i < seq_len; i++) sum += model->ws_d_ff_hidden[i * ffn_dim + d];
            layer->b_ff1_grad[d] += sum;
        }
        matmul_backward_weight(model->cache_attn_ln_out[l], model->ws_d_ff_hidden, layer->W_ff1_grad,
                                seq_len, embedding_dim, ffn_dim);

        memset(model->ws_d_x1_total, 0, seq_len * embedding_dim * sizeof(float));
        matmul_backward_input(model->ws_d_ff_hidden, layer->W_ff1, model->ws_d_x1_total,
                               seq_len, embedding_dim, ffn_dim);
        for (size_t i = 0; i < seq_len * embedding_dim; i++) {
            model->ws_d_x1_total[i] += model->ws_d_s2[i]; /* + residual branch */
        }

        /* Attn-LN backward: dL/dx1 (ws_d_x1_total) -> dL/d(pre-attn-ln-sum) (ws_d_s1) */
        layer_norm_backward(model->ws_d_x1_total, model->cache_attn_xhat[l], model->cache_attn_std[l],
                             layer->ln_gamma_attn, layer->ln_gamma_attn_grad, layer->ln_beta_attn_grad,
                             model->ws_d_s1, seq_len, embedding_dim);

        /* ws_d_s1 is dL/dattn_raw AND the residual branch's contribution to dL/dhidden[l] */
        memcpy(model->ws_dhidden_out, model->ws_d_s1, seq_len * embedding_dim * sizeof(float));
        multihead_attention_backward(model, l, seq_len, model->ws_d_s1, model->ws_dhidden_out);

        if (l == 0) {
            /* Scatter into the token embedding gradients (position
             * embeddings are fixed, so no gradient needed for them). */
            for (size_t i = 0; i < seq_len; i++) {
                uint32_t token_id = token_ids[i];
                if (token_id >= model->vocab_size) token_id = 0;
                for (size_t d = 0; d < embedding_dim; d++) {
                    model->token_embeddings_grad[token_id * embedding_dim + d] +=
                        model->ws_dhidden_out[i * embedding_dim + d];
                }
            }
        } else {
            memcpy(model->ws_dhidden_in, model->ws_dhidden_out, seq_len * embedding_dim * sizeof(float));
        }
    }

    /* ---- SGD update ---- */
    float lr = model->learning_rate;
    sgd_update(model->token_embeddings, model->token_embeddings_grad, model->vocab_size * embedding_dim, lr);
    sgd_update(model->output_projection, model->output_projection_grad, embedding_dim * model->vocab_size, lr);
    sgd_update(model->output_bias, model->output_bias_grad, model->vocab_size, lr);

    for (size_t l = 0; l < model->num_layers; l++) {
        transformer_layer_t *layer = &model->layers[l];
        sgd_update(layer->W_q, layer->W_q_grad, embedding_dim * embedding_dim, lr);
        sgd_update(layer->W_k, layer->W_k_grad, embedding_dim * embedding_dim, lr);
        sgd_update(layer->W_v, layer->W_v_grad, embedding_dim * embedding_dim, lr);
        sgd_update(layer->W_o, layer->W_o_grad, embedding_dim * embedding_dim, lr);
        sgd_update(layer->ln_gamma_attn, layer->ln_gamma_attn_grad, embedding_dim, lr);
        sgd_update(layer->ln_beta_attn, layer->ln_beta_attn_grad, embedding_dim, lr);
        sgd_update(layer->W_ff1, layer->W_ff1_grad, embedding_dim * ffn_dim, lr);
        sgd_update(layer->b_ff1, layer->b_ff1_grad, ffn_dim, lr);
        sgd_update(layer->W_ff2, layer->W_ff2_grad, ffn_dim * embedding_dim, lr);
        sgd_update(layer->b_ff2, layer->b_ff2_grad, embedding_dim, lr);
        sgd_update(layer->ln_gamma_ffn, layer->ln_gamma_ffn_grad, embedding_dim, lr);
        sgd_update(layer->ln_beta_ffn, layer->ln_beta_ffn_grad, embedding_dim, lr);
    }

    model->training_steps++;

    /* Phase 2: Update learning metrics */
    if (model->metrics.history_size < model->metrics.history_capacity) {
        model->metrics.loss_history[model->metrics.history_size] = loss;
        model->metrics.history_size++;
    }

    if (loss < model->metrics.best_loss) {
        model->metrics.best_loss = loss;
        model->metrics.steps_without_improvement = 0;
    } else {
        model->metrics.steps_without_improvement++;
    }

    if (loss > model->metrics.worst_loss) {
        model->metrics.worst_loss = loss;
    }

    model->metrics.avg_loss = (model->metrics.avg_loss * (model->training_steps - 1) + loss) / model->training_steps;

    if (model->metrics.steps_without_improvement > 10) {
        model->metrics.learning_rate *= 0.99f;
        model->learning_rate = model->metrics.learning_rate;
        model->metrics.steps_without_improvement = 0;
    }

    return MODEL_SUCCESS;
}

/* Predict next token */
uint32_t model_predict_next_token(neural_model_t *model,
                                  uint32_t *token_ids,
                                  size_t seq_len) {

    float *logits = model->ws_logits;
    model_forward(model, token_ids, seq_len, logits);

    uint32_t next_token = 0;
    float max_logit = logits[0];

    for (uint32_t i = 1; i < model->vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            next_token = i;
        }
    }

    return next_token;
}

/* Write model dimensions + all weights to an already-open stream. Shared by
 * model_save and checkpoint_save so the on-disk weight format has one
 * source of truth (previously they were independently duplicated and had
 * already drifted: model_save didn't persist layer-norm params at all). */
model_errors_t model_write_state(const neural_model_t *model, FILE *f) {
    if (!model || !f) {
        return MODEL_INVALID_INPUT;
    }

    fwrite(&model->vocab_size, sizeof(size_t), 1, f);
    fwrite(&model->embedding_dim, sizeof(size_t), 1, f);
    fwrite(&model->num_heads, sizeof(size_t), 1, f);
    fwrite(&model->num_layers, sizeof(size_t), 1, f);
    fwrite(&model->max_seq_len, sizeof(size_t), 1, f);
    fwrite(&model->training_steps, sizeof(uint32_t), 1, f);
    fwrite(&model->current_loss, sizeof(float), 1, f);

    size_t embedding_dim = model->embedding_dim;
    size_t ffn_dim = embedding_dim * 4;
    size_t vocab_size = model->vocab_size;

    fwrite(model->token_embeddings, sizeof(float), vocab_size * embedding_dim, f);
    fwrite(model->output_projection, sizeof(float), embedding_dim * vocab_size, f);
    fwrite(model->output_bias, sizeof(float), vocab_size, f);

    for (size_t l = 0; l < model->num_layers; l++) {
        transformer_layer_t *layer = &model->layers[l];
        fwrite(layer->W_q, sizeof(float), embedding_dim * embedding_dim, f);
        fwrite(layer->W_k, sizeof(float), embedding_dim * embedding_dim, f);
        fwrite(layer->W_v, sizeof(float), embedding_dim * embedding_dim, f);
        fwrite(layer->W_o, sizeof(float), embedding_dim * embedding_dim, f);
        fwrite(layer->ln_gamma_attn, sizeof(float), embedding_dim, f);
        fwrite(layer->ln_beta_attn, sizeof(float), embedding_dim, f);
        fwrite(layer->W_ff1, sizeof(float), embedding_dim * ffn_dim, f);
        fwrite(layer->b_ff1, sizeof(float), ffn_dim, f);
        fwrite(layer->W_ff2, sizeof(float), ffn_dim * embedding_dim, f);
        fwrite(layer->b_ff2, sizeof(float), embedding_dim, f);
        fwrite(layer->ln_gamma_ffn, sizeof(float), embedding_dim, f);
        fwrite(layer->ln_beta_ffn, sizeof(float), embedding_dim, f);
    }

    return MODEL_SUCCESS;
}

/* Read model dimensions + all weights from an already-open stream,
 * reinitializing *model if its current dimensions don't match. Shared by
 * model_load and checkpoint_load. */
model_errors_t model_read_state(neural_model_t *model, FILE *f) {
    if (!model || !f) {
        return MODEL_INVALID_INPUT;
    }

    size_t vocab_size, embedding_dim, num_heads, num_layers, max_seq_len;
    uint32_t training_steps;
    float current_loss;

    if (fread(&vocab_size, sizeof(size_t), 1, f) != 1 ||
        fread(&embedding_dim, sizeof(size_t), 1, f) != 1 ||
        fread(&num_heads, sizeof(size_t), 1, f) != 1 ||
        fread(&num_layers, sizeof(size_t), 1, f) != 1 ||
        fread(&max_seq_len, sizeof(size_t), 1, f) != 1 ||
        fread(&training_steps, sizeof(uint32_t), 1, f) != 1 ||
        fread(&current_loss, sizeof(float), 1, f) != 1) {
        return MODEL_IO_ERROR;
    }

    if (model->vocab_size != vocab_size || model->embedding_dim != embedding_dim ||
        model->num_heads != num_heads || model->num_layers != num_layers ||
        model->max_seq_len != max_seq_len) {
        model_free(model);
        model_errors_t rc = model_new(model, vocab_size, embedding_dim, num_heads, num_layers, max_seq_len);
        if (rc != MODEL_SUCCESS) {
            return rc;
        }
    }

    model->training_steps = training_steps;
    model->current_loss = current_loss;

    size_t ffn_dim = embedding_dim * 4;

    fread(model->token_embeddings, sizeof(float), vocab_size * embedding_dim, f);
    fread(model->output_projection, sizeof(float), embedding_dim * vocab_size, f);
    fread(model->output_bias, sizeof(float), vocab_size, f);

    for (size_t l = 0; l < model->num_layers; l++) {
        transformer_layer_t *layer = &model->layers[l];
        fread(layer->W_q, sizeof(float), embedding_dim * embedding_dim, f);
        fread(layer->W_k, sizeof(float), embedding_dim * embedding_dim, f);
        fread(layer->W_v, sizeof(float), embedding_dim * embedding_dim, f);
        fread(layer->W_o, sizeof(float), embedding_dim * embedding_dim, f);
        fread(layer->ln_gamma_attn, sizeof(float), embedding_dim, f);
        fread(layer->ln_beta_attn, sizeof(float), embedding_dim, f);
        fread(layer->W_ff1, sizeof(float), embedding_dim * ffn_dim, f);
        fread(layer->b_ff1, sizeof(float), ffn_dim, f);
        fread(layer->W_ff2, sizeof(float), ffn_dim * embedding_dim, f);
        fread(layer->b_ff2, sizeof(float), embedding_dim, f);
        fread(layer->ln_gamma_ffn, sizeof(float), embedding_dim, f);
        fread(layer->ln_beta_ffn, sizeof(float), embedding_dim, f);
    }

    return MODEL_SUCCESS;
}

/* Save model to file */
model_errors_t model_save(neural_model_t *model, const char *filename) {
    if (!model || !filename) {
        return MODEL_INVALID_INPUT;
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        return MODEL_IO_ERROR;
    }

    DEBUG_PRINT("Saving model to %s\n", filename);
    model_errors_t rc = model_write_state(model, f);
    fclose(f);
    if (rc == MODEL_SUCCESS) {
        DEBUG_PRINT("Model saved successfully\n");
    }
    return rc;
}

/* Load model from file */
model_errors_t model_load(neural_model_t *model, const char *filename) {
    if (!model || !filename) {
        return MODEL_INVALID_INPUT;
    }

    FILE *f = fopen(filename, "rb");
    if (!f) {
        return MODEL_IO_ERROR;
    }

    DEBUG_PRINT("Loading model from %s\n", filename);
    model_errors_t rc = model_read_state(model, f);
    fclose(f);
    if (rc == MODEL_SUCCESS) {
        DEBUG_PRINT("Model loaded successfully (training_steps=%u, loss=%.4f)\n",
                    model->training_steps, model->current_loss);
    }
    return rc;
}

/* Free all model resources */
void model_free(neural_model_t *model) {
    if (!model) return;

    free(model->token_embeddings);
    free(model->token_embeddings_grad);
    free(model->position_embeddings);
    free(model->output_projection);
    free(model->output_projection_grad);
    free(model->output_bias);
    free(model->output_bias_grad);

    if (model->layers) {
        for (size_t l = 0; l < model->num_layers; l++) {
            transformer_layer_t *layer = &model->layers[l];
            free(layer->W_q); free(layer->W_k); free(layer->W_v); free(layer->W_o);
            free(layer->W_q_grad); free(layer->W_k_grad); free(layer->W_v_grad); free(layer->W_o_grad);
            free(layer->ln_gamma_attn); free(layer->ln_beta_attn);
            free(layer->ln_gamma_attn_grad); free(layer->ln_beta_attn_grad);
            free(layer->W_ff1); free(layer->b_ff1); free(layer->W_ff2); free(layer->b_ff2);
            free(layer->W_ff1_grad); free(layer->b_ff1_grad); free(layer->W_ff2_grad); free(layer->b_ff2_grad);
            free(layer->ln_gamma_ffn); free(layer->ln_beta_ffn);
            free(layer->ln_gamma_ffn_grad); free(layer->ln_beta_ffn_grad);
        }
        free(model->layers);
    }

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
    free(model->ws_logits);
    free(model->ws_grad_logits);

    free(model->metrics.loss_history);

    memset(model, 0, sizeof(neural_model_t));
}

/* Phase 2: Public layer normalization wrapper */
void layer_normalize(float *input, float *output, size_t size,
                     float *gamma, float *beta, float epsilon) {
    memcpy(output, input, size * sizeof(float));
    layer_norm_internal(output, gamma, beta, size, epsilon);
}

/* Phase 2: Update learning rate based on training progress */
void update_learning_rate(neural_model_t *model) {
    if (!model) return;

    if (model->metrics.steps_without_improvement > 20) {
        model->metrics.learning_rate *= 0.95f;
        model->learning_rate = model->metrics.learning_rate;
        model->metrics.steps_without_improvement = 0;

        DEBUG_PRINT("Learning rate reduced to %.6f\n", model->metrics.learning_rate);
    }
}

/* Phase 2: Get learning metrics */
void model_get_metrics(neural_model_t *model, learning_metrics_t *out_metrics) {
    if (!model || !out_metrics) return;
    *out_metrics = model->metrics;
}

/* Phase 2: Print training metrics and statistics */
void model_print_metrics(neural_model_t *model) {
    if (!model) return;

    printf("\n=== Phase 2: Learning Metrics ===\n");
    printf("Training steps: %u\n", model->training_steps);
    printf("Current loss: %.6f\n", model->current_loss);
    printf("Best loss: %.6f\n", model->metrics.best_loss);
    printf("Worst loss: %.6f\n", model->metrics.worst_loss);
    printf("Average loss: %.6f\n", model->metrics.avg_loss);
    printf("Current learning rate: %.8f\n", model->metrics.learning_rate);
    printf("Steps without improvement: %u\n", model->metrics.steps_without_improvement);

    if (model->metrics.history_size > 0) {
        printf("\nLast %zu loss values: ",
               model->metrics.history_size < 10 ? model->metrics.history_size : 10);
        size_t start = model->metrics.history_size < 10 ? 0 : model->metrics.history_size - 10;
        for (size_t i = start; i < model->metrics.history_size; i++) {
            printf("%.4f ", model->metrics.loss_history[i]);
        }
        printf("\n");
    }

    printf("===================================\n\n");
}
