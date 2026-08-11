/*
 * Multi-head self-attention and the stacked transformer forward pass.
 * Backward for the FFN/layer-norm/output-head parts of each block lives in
 * training.c (it's tightly interleaved with the training step's gradient
 * bookkeeping); this file owns attention's own forward+backward plus the
 * overall layer stack's forward.
 */

#include "core/transformer.h"
#include "core/parallel.h"
#include "core/tensor_ops.h"
#include "core/matmul_dispatch.h"
#include "common/debug.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward-pass matmul dispatch (GPU when the model opts in and one is
 * usable, CPU otherwise) moved to core/matmul_dispatch.c when core/lm_head.c
 * became a third caller needing the same policy; call sites below use
 * model_dispatch_matmul() directly. */

/* One head of the forward pass, lifted out of the loop below so the loop body
 * is a single call and DRANZER_PARALLEL_FOR can guard it without duplicating
 * fifty lines - see core/parallel.h. */
static void attention_head_forward(size_t head, size_t seq_len,
                                   size_t embedding_dim, size_t head_dim,
                                   const float *Q, const float *K, const float *V,
                                   const uint8_t *attention_allowed,
                                   float *probs, float *concat) {
    float *head_probs = probs + head * seq_len * seq_len;

    if (attention_allowed != NULL) {
        /* General sparse boolean mask. Scores exist only for allowed edges;
         * fully masked rows deliberately stay all-zero. This is the finite
         * equivalent of an all -infinity row, without the NaN softmax that
         * representation would produce (or violating -ffinite-math-only). */
        for (size_t i = 0; i < seq_len; i++) {
            float *row = &head_probs[i * seq_len];
            memset(row, 0, seq_len * sizeof(*row));
            size_t allowed_count = 0;
            float max_score = 0.0f;
            for (size_t j = 0; j < seq_len; j++) {
                if (!attention_allowed[i * seq_len + j]) continue;
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    size_t q_idx = i * embedding_dim + head * head_dim + d;
                    size_t k_idx = j * embedding_dim + head * head_dim + d;
                    score += Q[q_idx] * K[k_idx];
                }
                row[j] = score / sqrtf((float)head_dim);
                if (allowed_count == 0 || row[j] > max_score) {
                    max_score = row[j];
                }
                allowed_count++;
            }
            if (allowed_count == 0) continue;

            float sum = 0.0f;
            for (size_t j = 0; j < seq_len; j++) {
                if (!attention_allowed[i * seq_len + j]) continue;
                row[j] = expf(row[j] - max_score);
                sum += row[j];
            }
            if (sum > 0.0f) {
                float inverse = 1.0f / sum;
                for (size_t j = 0; j < seq_len; j++) {
                    if (attention_allowed[i * seq_len + j]) row[j] *= inverse;
                }
            }
        }

        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    if (!attention_allowed[i * seq_len + j]) continue;
                    sum += head_probs[i * seq_len + j] *
                           V[j * embedding_dim + head * head_dim + d];
                }
                concat[i * embedding_dim + head * head_dim + d] = sum;
            }
        }
        return;
    }

    /* Causal mask: position i may only attend to j <= i.
     *
     * The mask is structural, not a value. Future positions are written as
     * exact 0.0f - the probability they must end up with - and the softmax
     * normalizes only the j <= i prefix of the row, which is the only part
     * that ever holds a score. Nothing downstream can tell the difference:
     * a row's masked tail was already exactly zero after the old softmax,
     * and the prefix is normalized over the same set of scores by the same
     * arithmetic in the same order, so probs is bit-identical to what the
     * -INFINITY form produced.
     *
     * It used to write -INFINITY and let softmax underflow it away. That was
     * a correctness bet against this project's own build flags: src/Makefile
     * compiles with -ffast-math, which implies -ffinite-math-only, which
     * promises the compiler that no infinity will ever appear. The masking
     * worked only because no optimization had yet acted on the permission it
     * was given, and the same flag had already produced two wrong results
     * elsewhere in the tree (see include/common/fp_bits.h). The flag is worth
     * keeping - it is measured, not inherited (src/Makefile) - so the
     * dependence on it goes instead.
     *
     * Row i always has at least the j == i entry, so no prefix is ever empty
     * and no row is ever entirely masked.
     *
     * Both loops now stop at i rather than seq_len, which halves the scalar
     * work in the two O(seq_len^2 * head_dim) passes as a side effect: the
     * skipped terms were multiplications by a probability of exactly zero. */
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j <= i; j++) {
            float score = 0.0f;
            for (size_t d = 0; d < head_dim; d++) {
                size_t q_idx = i * embedding_dim + head * head_dim + d;
                size_t k_idx = j * embedding_dim + head * head_dim + d;
                score += Q[q_idx] * K[k_idx];
            }
            head_probs[i * seq_len + j] = score / sqrtf((float)head_dim);
        }
        for (size_t j = i + 1; j < seq_len; j++) {
            head_probs[i * seq_len + j] = 0.0f;
        }
    }

    for (size_t i = 0; i < seq_len; i++) {
        softmax(&head_probs[i * seq_len], i + 1);
    }

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (size_t j = 0; j <= i; j++) {
                sum += head_probs[i * seq_len + j] * V[j * embedding_dim + head * head_dim + d];
            }
            concat[i * embedding_dim + head * head_dim + d] = sum;
        }
    }
}

void multihead_attention_forward(neural_model_t *model, size_t l, size_t seq_len) {
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

    model_dispatch_matmul(model, sequence, layer->W_q, Q, seq_len, embedding_dim, embedding_dim);
    model_dispatch_matmul(model, sequence, layer->W_k, K, seq_len, embedding_dim, embedding_dim);
    model_dispatch_matmul(model, sequence, layer->W_v, V, seq_len, embedding_dim, embedding_dim);

    memset(concat, 0, seq_len * embedding_dim * sizeof(float));

    /* Parallel over heads: each head reads all of Q/K/V but only ever
     * writes its own head*head_dim slice of concat and its own
     * head*seq_len*seq_len slab of probs - disjoint per head, so this is
     * safe with no cross-thread reduction.
     *
     * Work is two seq_len x seq_len x head_dim passes per head - the scores
     * and the value-weighted sum - which over all heads is
     * 2 * seq_len^2 * embedding_dim, plus one softmax per row of every head.
     * See DRANZER_PARALLEL_SOFTMAX_WORK for why the softmax is counted. */
    DRANZER_PARALLEL_FOR(num_heads,
                         2 * seq_len * seq_len * embedding_dim +
                             num_heads * seq_len * seq_len * DRANZER_PARALLEL_SOFTMAX_WORK,
                         head,
        attention_head_forward(head, seq_len, embedding_dim, head_dim,
                               Q, K, V,
                               model->cache_attention_mask_active
                                   ? model->cache_attention_allowed : NULL,
                               probs, concat);
    );

    model_dispatch_matmul(model, concat, layer->W_o, model->ws_fwd_attn_raw, seq_len, embedding_dim, embedding_dim);
}

/* One head of the backward pass, lifted out for the same reason as
 * attention_head_forward() above. */
static void attention_head_backward(size_t head, size_t seq_len,
                                    size_t embedding_dim, size_t head_dim,
                                    float scale, const float *Q, const float *K,
                                    const float *V, const float *d_concat,
                                    float *probs, float *d_scores_base,
                                    float *dQ, float *dK, float *dV) {
    float *head_probs = probs + head * seq_len * seq_len;
    float *d_scores = d_scores_base + head * seq_len * seq_len;

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

void multihead_attention_backward(neural_model_t *model, size_t l, size_t seq_len,
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

    /* One seq_len x seq_len slab per head (ws_d_scores is sized
     * num_heads*max_seq_len*max_seq_len) so heads never share a scratch
     * buffer - required for the head loop below to be safely parallel. */
    float *d_scores_base = model->ws_d_scores;

    /* Parallel over heads: dQ/dK/dV are accumulated (+=) but each head only
     * ever touches its own head*head_dim slice of them, so writes are
     * disjoint across threads - no cross-thread reduction, no race.
     *
     * Four seq_len x seq_len x head_dim passes per head against the forward
     * pass's two, so twice its work estimate. No softmax term: this pass runs
     * softmax_backward(), which has no exponential in it. */
    DRANZER_PARALLEL_FOR(num_heads, 4 * seq_len * seq_len * embedding_dim, head,
        attention_head_backward(head, seq_len, embedding_dim, head_dim, scale,
                                Q, K, V, d_concat, probs, d_scores_base,
                                dQ, dK, dV);
    );

    matmul_backward_weight(sequence, dQ, layer->W_q_grad, seq_len, embedding_dim, embedding_dim);
    matmul_backward_weight(sequence, dK, layer->W_k_grad, seq_len, embedding_dim, embedding_dim);
    matmul_backward_weight(sequence, dV, layer->W_v_grad, seq_len, embedding_dim, embedding_dim);

    matmul_backward_input(dQ, layer->W_q, dL_dhidden_accum, seq_len, embedding_dim, embedding_dim);
    matmul_backward_input(dK, layer->W_k, dL_dhidden_accum, seq_len, embedding_dim, embedding_dim);
    matmul_backward_input(dV, layer->W_v, dL_dhidden_accum, seq_len, embedding_dim, embedding_dim);
}

static void zero_padded_rows(const neural_model_t *model, float *values,
                             size_t seq_len, size_t width) {
    if (!model->cache_padding_mask_active) return;
    for (size_t i = 0; i < seq_len; i++) {
        if (!model->cache_padding_mask[i]) {
            memset(&values[i * width], 0, width * sizeof(*values));
        }
    }
}

static void prepare_attention_mask(neural_model_t *model, size_t seq_len,
                                   const model_attention_mask_t *mask) {
    const uint8_t *padding = mask ? mask->padding_mask : NULL;
    const uint8_t *attention = mask ? mask->attention_mask : NULL;
    model->cache_padding_mask_active = padding != NULL;
    model->cache_attention_mask_active = padding != NULL || attention != NULL;

    if (padding != NULL) {
        for (size_t i = 0; i < seq_len; i++) {
            model->cache_padding_mask[i] = padding[i] != 0;
        }
    }
    if (!model->cache_attention_mask_active) return;

    for (size_t i = 0; i < seq_len; i++) {
        const int query_valid = padding == NULL || padding[i] != 0;
        for (size_t j = 0; j < seq_len; j++) {
            const int key_valid = padding == NULL || padding[j] != 0;
            const int custom_allowed =
                attention == NULL || attention[i * seq_len + j] != 0;
            model->cache_attention_allowed[i * seq_len + j] =
                (uint8_t)(j <= i && query_valid && key_valid && custom_allowed);
        }
    }
}

model_errors_t model_forward_hidden_masked(
    neural_model_t *model, uint32_t *token_ids, size_t seq_len,
    const model_attention_mask_t *mask) {

    if (!model || !token_ids) {
        return MODEL_INVALID_INPUT;
    }

    if (seq_len == 0 || seq_len > model->max_seq_len) {
        return MODEL_INVALID_INPUT;
    }

    prepare_attention_mask(model, seq_len, mask);

    size_t embedding_dim = model->embedding_dim;
    size_t ffn_dim = embedding_dim * 4;
    const float epsilon = 1e-6f;

    DEBUG_PRINT("Model forward pass: seq_len=%zu, embedding_dim=%zu, num_layers=%zu\n",
                seq_len, embedding_dim, model->num_layers);

    /* 1. Embed tokens + positional encoding -> cache_hidden[0] */
    float *h0 = model->cache_hidden[0];
    for (size_t i = 0; i < seq_len; i++) {
        if (model->cache_padding_mask_active &&
            !model->cache_padding_mask[i]) {
            memset(&h0[i * embedding_dim], 0,
                   embedding_dim * sizeof(*h0));
            continue;
        }
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

        /* Attention sub-block: attn -> dropout -> residual -> LN */
        multihead_attention_forward(model, l, seq_len);
        dropout_forward(model->ws_fwd_attn_raw, model->cache_attn_dropout_mask[l],
                             seq_len * embedding_dim, model->dropout_rate,
                             model->is_training, &model->rng_state);
        for (size_t i = 0; i < seq_len * embedding_dim; i++) {
            model->ws_fwd_attn_raw[i] += x_in[i];
        }
        layer_norm_forward_cached(model->ws_fwd_attn_raw, model->cache_attn_xhat[l], model->cache_attn_std[l],
                                   model->cache_attn_ln_out[l], layer->ln_gamma_attn, layer->ln_beta_attn,
                                   seq_len, embedding_dim, epsilon);
        zero_padded_rows(model, model->cache_attn_ln_out[l], seq_len,
                         embedding_dim);

        /* FFN sub-block: matmul -> bias -> ReLU -> matmul -> bias -> dropout -> residual -> LN */
        float *x1 = model->cache_attn_ln_out[l];
        model_dispatch_matmul(model, x1, layer->W_ff1, model->cache_ff_hidden[l], seq_len, embedding_dim, ffn_dim);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < ffn_dim; d++) {
                float *v = &model->cache_ff_hidden[l][i * ffn_dim + d];
                *v = relu(*v + layer->b_ff1[d]);
            }
        }
        model_dispatch_matmul(model, model->cache_ff_hidden[l], layer->W_ff2, model->ws_fwd_ff_raw,
                       seq_len, ffn_dim, embedding_dim);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < embedding_dim; d++) {
                model->ws_fwd_ff_raw[i * embedding_dim + d] += layer->b_ff2[d];
            }
        }
        dropout_forward(model->ws_fwd_ff_raw, model->cache_ffn_dropout_mask[l],
                             seq_len * embedding_dim, model->dropout_rate,
                             model->is_training, &model->rng_state);
        for (size_t i = 0; i < seq_len * embedding_dim; i++) {
            model->ws_fwd_ff_raw[i] += x1[i];
        }
        layer_norm_forward_cached(model->ws_fwd_ff_raw, model->cache_ffn_xhat[l], model->cache_ffn_std[l],
                                   model->cache_hidden[l + 1], layer->ln_gamma_ffn, layer->ln_beta_ffn,
                                   seq_len, embedding_dim, epsilon);
        zero_padded_rows(model, model->cache_hidden[l + 1], seq_len,
                         embedding_dim);
    }

    return MODEL_SUCCESS;
}

model_errors_t model_forward_hidden(neural_model_t *model,
                                    uint32_t *token_ids,
                                    size_t seq_len) {
    return model_forward_hidden_masked(model, token_ids, seq_len, NULL);
}

model_errors_t model_forward_masked(neural_model_t *model,
                                    uint32_t *token_ids,
                                    size_t seq_len,
                                    const model_attention_mask_t *mask,
                                    float *output_logits) {
    if (!model || !token_ids || !output_logits || seq_len == 0 ||
        seq_len > model->max_seq_len) {
        return MODEL_INVALID_INPUT;
    }

    size_t last_position = seq_len - 1;
    if (mask && mask->padding_mask) {
        size_t found = 0;
        for (size_t i = 0; i < seq_len; i++) {
            if (mask->padding_mask[i]) {
                last_position = i;
                found = 1;
            }
        }
        if (!found) return MODEL_INVALID_INPUT;
    }

    model_errors_t rc = model_forward_hidden_masked(model, token_ids, seq_len,
                                                    mask);
    if (rc != MODEL_SUCCESS) {
        return rc;
    }

    /* Output projection to vocabulary, from the last layer's last real
     * position only. In an unpadded call that is the physical last position.
     * That is all inference ever needs: the causal mask means every earlier
     * position's row predicts a token the caller already has.
     *
     * Training goes through lm_head.c instead, which projects every row -
     * the same arithmetic, seq_len times over, for seq_len times the
     * supervision out of this one forward pass. Keeping the m=1 path here
     * matters: routing prefill through the all-positions head would multiply
     * its output-projection cost by seq_len for logits nothing reads. */
    size_t embedding_dim = model->embedding_dim;
    float *last_hidden = &model->cache_hidden[model->num_layers]
                                      [last_position * embedding_dim];
    model_dispatch_matmul(model, last_hidden, model->output_projection, output_logits, 1, embedding_dim, model->vocab_size);

    for (size_t i = 0; i < model->vocab_size; i++) {
        output_logits[i] += model->output_bias[i];
    }

    return MODEL_SUCCESS;
}

model_errors_t model_forward(neural_model_t *model,
                             uint32_t *token_ids,
                             size_t seq_len,
                             float *output_logits) {
    return model_forward_masked(model, token_ids, seq_len, NULL,
                                output_logits);
}

model_errors_t model_kv_cache_init(model_kv_cache_t *cache, const neural_model_t *model) {
    return model_kv_cache_init_with_capacity(cache, model,
                                             model ? model->max_seq_len : 0);
}

model_errors_t model_kv_cache_init_with_capacity(model_kv_cache_t *cache,
                                                  const neural_model_t *model,
                                                  size_t capacity) {
    if (!cache || !model || model->num_layers == 0 || model->num_heads == 0 ||
        model->embedding_dim == 0 || capacity == 0 ||
        capacity > model->max_seq_len) {
        return MODEL_INVALID_INPUT;
    }

    memset(cache, 0, sizeof(*cache));
    cache->model = model;
    cache->capacity = capacity;
    cache->num_layers = model->num_layers;
    cache->num_heads = model->num_heads;
    cache->embedding_dim = model->embedding_dim;

    size_t embedding_dim = model->embedding_dim;
    size_t ffn_dim = embedding_dim * 4;
    size_t cache_floats = capacity * embedding_dim;

    cache->keys = calloc(model->num_layers, sizeof(float *));
    cache->values = calloc(model->num_layers, sizeof(float *));
    if (!cache->keys || !cache->values) {
        model_kv_cache_free(cache);
        return MODEL_ALLOCATION_FAILURE;
    }

    for (size_t l = 0; l < model->num_layers; l++) {
        cache->keys[l] = malloc(cache_floats * sizeof(float));
        cache->values[l] = malloc(cache_floats * sizeof(float));
        if (!cache->keys[l] || !cache->values[l]) {
            model_kv_cache_free(cache);
            return MODEL_ALLOCATION_FAILURE;
        }
    }

    cache->hidden = malloc(embedding_dim * sizeof(float));
    cache->attn_norm = malloc(embedding_dim * sizeof(float));
    cache->query = malloc(embedding_dim * sizeof(float));
    cache->attn_concat = malloc(embedding_dim * sizeof(float));
    cache->attn_raw = malloc(embedding_dim * sizeof(float));
    cache->ff_hidden = malloc(ffn_dim * sizeof(float));
    cache->ff_raw = malloc(embedding_dim * sizeof(float));
    cache->scores = malloc(model->num_heads * capacity * sizeof(float));
    cache->position_embedding = malloc(embedding_dim * sizeof(float));

    if (!cache->hidden || !cache->attn_norm || !cache->query || !cache->attn_concat ||
        !cache->attn_raw || !cache->ff_hidden || !cache->ff_raw || !cache->scores ||
        !cache->position_embedding) {
        model_kv_cache_free(cache);
        return MODEL_ALLOCATION_FAILURE;
    }

    return MODEL_SUCCESS;
}

void model_kv_cache_reset(model_kv_cache_t *cache) {
    if (cache) {
        cache->length = 0;
        cache->start = 0;
        cache->total_tokens = 0;
    }
}

void model_kv_cache_free(model_kv_cache_t *cache) {
    if (!cache) return;
    if (cache->keys) {
        for (size_t l = 0; l < cache->num_layers; l++) free(cache->keys[l]);
    }
    if (cache->values) {
        for (size_t l = 0; l < cache->num_layers; l++) free(cache->values[l]);
    }
    free(cache->keys);
    free(cache->values);
    free(cache->hidden);
    free(cache->attn_norm);
    free(cache->query);
    free(cache->attn_concat);
    free(cache->attn_raw);
    free(cache->ff_hidden);
    free(cache->ff_raw);
    free(cache->scores);
    free(cache->position_embedding);
    memset(cache, 0, sizeof(*cache));
}

/* One head of cached single-token decode. Lifted out like the two above, and
 * the most cutoff-sensitive of the three: this is the generation hot path, and
 * a head here does only context_len x head_dim work - well under a microsecond
 * at every tier this project benchmarks. */
static void attention_head_forward_token(size_t head, model_kv_cache_t *cache,
                                         size_t layer_index, size_t context_len,
                                         size_t embedding_dim, size_t head_dim) {
    float *scores = &cache->scores[head * cache->capacity];
    for (size_t j = 0; j < context_len; j++) {
        float score = 0.0f;
        size_t physical = (cache->start + j) % cache->capacity;
        const float *cached_key =
            &cache->keys[layer_index][physical * embedding_dim];
        for (size_t d = 0; d < head_dim; d++) {
            size_t idx = head * head_dim + d;
            score += cache->query[idx] * cached_key[idx];
        }
        scores[j] = score / sqrtf((float)head_dim);
    }
    softmax(scores, context_len);

    for (size_t d = 0; d < head_dim; d++) {
        size_t idx = head * head_dim + d;
        float sum = 0.0f;
        for (size_t j = 0; j < context_len; j++) {
            size_t physical = (cache->start + j) % cache->capacity;
            sum += scores[j] *
                   cache->values[layer_index][physical * embedding_dim + idx];
        }
        cache->attn_concat[idx] = sum;
    }
}

static void attention_forward_token(neural_model_t *model, model_kv_cache_t *cache,
                                    size_t layer_index, size_t slot,
                                    size_t context_len) {
    transformer_layer_t *layer = &model->layers[layer_index];
    size_t embedding_dim = model->embedding_dim;
    size_t num_heads = model->num_heads;
    size_t head_dim = embedding_dim / num_heads;

    float *key = &cache->keys[layer_index][slot * embedding_dim];
    float *value = &cache->values[layer_index][slot * embedding_dim];
    model_dispatch_matmul(model, cache->hidden, layer->W_q, cache->query, 1, embedding_dim, embedding_dim);
    model_dispatch_matmul(model, cache->hidden, layer->W_k, key, 1, embedding_dim, embedding_dim);
    model_dispatch_matmul(model, cache->hidden, layer->W_v, value, 1, embedding_dim, embedding_dim);

    memset(cache->attn_concat, 0, embedding_dim * sizeof(float));

    /* Two context_len x head_dim passes per head - 2 * context_len *
     * embedding_dim over all of them - plus one softmax of context_len
     * elements per head. The softmax term is 40% of this estimate and decides
     * the small tier's verdict, which is what makes it worth counting here
     * rather than treating attention as pure multiply-adds. */
    DRANZER_PARALLEL_FOR(num_heads,
                         2 * context_len * embedding_dim +
                             num_heads * context_len * DRANZER_PARALLEL_SOFTMAX_WORK,
                         head,
        attention_head_forward_token(head, cache, layer_index, context_len,
                                     embedding_dim, head_dim);
    );

    model_dispatch_matmul(model, cache->attn_concat, layer->W_o, cache->attn_raw,
                    1, embedding_dim, embedding_dim);
}

model_errors_t model_forward_token(neural_model_t *model, model_kv_cache_t *cache,
                                   uint32_t token_id, float *output_logits) {
    if (!model || !cache || !output_logits || cache->model != model ||
        cache->embedding_dim != model->embedding_dim ||
        cache->num_layers != model->num_layers ||
        cache->num_heads != model->num_heads || cache->capacity == 0 ||
        cache->length > cache->capacity || cache->total_tokens == SIZE_MAX) {
        return MODEL_INVALID_INPUT;
    }

    size_t absolute_position = cache->total_tokens;
    size_t slot = 0;
    size_t context_len = 0;
    if (cache->length < cache->capacity) {
        slot = (cache->start + cache->length) % cache->capacity;
        context_len = cache->length + 1;
    } else {
        slot = cache->start;
        cache->start = (cache->start + 1) % cache->capacity;
        context_len = cache->capacity;
    }
    size_t embedding_dim = model->embedding_dim;
    size_t ffn_dim = embedding_dim * 4;
    const float epsilon = 1e-6f;

    const float *position_embedding = NULL;
    if (absolute_position < model->max_seq_len) {
        position_embedding =
            &model->position_embeddings[absolute_position * embedding_dim];
    } else {
        if (compute_positional_encoding_at(cache->position_embedding,
                                           absolute_position,
                                           embedding_dim) != 0) {
            return MODEL_INVALID_INPUT;
        }
        position_embedding = cache->position_embedding;
    }

    if (token_id >= model->vocab_size) token_id = 0;
    for (size_t d = 0; d < embedding_dim; d++) {
        cache->hidden[d] = model->token_embeddings[token_id * embedding_dim + d] +
                           position_embedding[d];
    }

    for (size_t l = 0; l < model->num_layers; l++) {
        transformer_layer_t *layer = &model->layers[l];
        attention_forward_token(model, cache, l, slot, context_len);

        for (size_t d = 0; d < embedding_dim; d++) {
            cache->attn_raw[d] += cache->hidden[d];
        }
        layer_normalize(cache->attn_raw, cache->attn_norm, embedding_dim,
                        layer->ln_gamma_attn, layer->ln_beta_attn, epsilon);

        model_dispatch_matmul(model, cache->attn_norm, layer->W_ff1, cache->ff_hidden,
                        1, embedding_dim, ffn_dim);
        for (size_t d = 0; d < ffn_dim; d++) {
            cache->ff_hidden[d] = relu(cache->ff_hidden[d] + layer->b_ff1[d]);
        }
        model_dispatch_matmul(model, cache->ff_hidden, layer->W_ff2, cache->ff_raw,
                        1, ffn_dim, embedding_dim);
        for (size_t d = 0; d < embedding_dim; d++) {
            cache->ff_raw[d] += layer->b_ff2[d] + cache->attn_norm[d];
        }
        layer_normalize(cache->ff_raw, cache->hidden, embedding_dim,
                        layer->ln_gamma_ffn, layer->ln_beta_ffn, epsilon);
    }

    model_dispatch_matmul(model, cache->hidden, model->output_projection, output_logits,
                    1, embedding_dim, model->vocab_size);
    for (size_t i = 0; i < model->vocab_size; i++) {
        output_logits[i] += model->output_bias[i];
    }

    if (cache->length < cache->capacity) cache->length++;
    cache->total_tokens++;
    return MODEL_SUCCESS;
}
