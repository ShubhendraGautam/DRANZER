/*
 * Multi-head self-attention and the stacked transformer forward pass.
 * Backward for the FFN/layer-norm/output-head parts of each block lives in
 * training.c (it's tightly interleaved with the training step's gradient
 * bookkeeping); this file owns attention's own forward+backward plus the
 * overall layer stack's forward.
 */

#include "core/transformer.h"
#include "core/tensor_ops.h"
#include "backends/gpu/gpu_matmul.h"
#include "common/debug.h"
#include <string.h>
#include <math.h>

/* Forward-pass matmul dispatch: GPU (hand-written PTX, gpu_matmul.c) when
 * model->use_gpu is set and a CUDA GPU is actually usable, CPU otherwise.
 * gpu_matmul_available() is checked at every call (not cached once) so a
 * model can be constructed and used identically regardless of whether the
 * machine has an NVIDIA GPU - the flag is simply a no-op without one.
 * Backward-pass matmuls (matmul_backward_input/matmul_backward_weight,
 * called directly from training.c) are CPU-only for now; see gpu_matmul.h
 * for why the forward-only, opt-in scope was chosen first (host<->device
 * transfer overhead needs to be measured against actual compute savings
 * at this project's model sizes before it's worth extending). */
static void dispatch_matmul(neural_model_t *model, float *A, float *B, float *C,
                             size_t m, size_t k, size_t n) {
    if (model->use_gpu && gpu_matmul_available() && gpu_matmul(A, B, C, m, k, n) == 0) {
        return;
    }
    matrix_multiply(A, B, C, m, k, n);
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

    dispatch_matmul(model, sequence, layer->W_q, Q, seq_len, embedding_dim, embedding_dim);
    dispatch_matmul(model, sequence, layer->W_k, K, seq_len, embedding_dim, embedding_dim);
    dispatch_matmul(model, sequence, layer->W_v, V, seq_len, embedding_dim, embedding_dim);

    memset(concat, 0, seq_len * embedding_dim * sizeof(float));

    /* Parallel over heads: each head reads all of Q/K/V but only ever
     * writes its own head*head_dim slice of concat and its own
     * head*seq_len*seq_len slab of probs - disjoint per head, so this is
     * safe with no cross-thread reduction. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t head = 0; head < num_heads; head++) {
        float *head_probs = probs + head * seq_len * seq_len;

        for (size_t i = 0; i < seq_len; i++) {
            /* Causal mask: position i may only attend to j <= i. Future
             * positions get -INFINITY so softmax zeroes them out; row i
             * always has at least the j==i entry, so no row is ever
             * all -INFINITY. */
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
                head_probs[i * seq_len + j] = -INFINITY;
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

    dispatch_matmul(model, concat, layer->W_o, model->ws_fwd_attn_raw, seq_len, embedding_dim, embedding_dim);
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
     * disjoint across threads - no cross-thread reduction, no race. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t head = 0; head < num_heads; head++) {
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

    matmul_backward_weight(sequence, dQ, layer->W_q_grad, seq_len, embedding_dim, embedding_dim);
    matmul_backward_weight(sequence, dK, layer->W_k_grad, seq_len, embedding_dim, embedding_dim);
    matmul_backward_weight(sequence, dV, layer->W_v_grad, seq_len, embedding_dim, embedding_dim);

    matmul_backward_input(dQ, layer->W_q, dL_dhidden_accum, seq_len, embedding_dim, embedding_dim);
    matmul_backward_input(dK, layer->W_k, dL_dhidden_accum, seq_len, embedding_dim, embedding_dim);
    matmul_backward_input(dV, layer->W_v, dL_dhidden_accum, seq_len, embedding_dim, embedding_dim);
}

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

        /* Attention sub-block: attn -> dropout -> residual -> LN */
        multihead_attention_forward(model, l, seq_len);
        dropout_forward(model->ws_fwd_attn_raw, model->cache_attn_dropout_mask[l],
                         seq_len * embedding_dim, model->dropout_rate, model->is_training);
        for (size_t i = 0; i < seq_len * embedding_dim; i++) {
            model->ws_fwd_attn_raw[i] += x_in[i];
        }
        layer_norm_forward_cached(model->ws_fwd_attn_raw, model->cache_attn_xhat[l], model->cache_attn_std[l],
                                   model->cache_attn_ln_out[l], layer->ln_gamma_attn, layer->ln_beta_attn,
                                   seq_len, embedding_dim, epsilon);

        /* FFN sub-block: matmul -> bias -> ReLU -> matmul -> bias -> dropout -> residual -> LN */
        float *x1 = model->cache_attn_ln_out[l];
        dispatch_matmul(model, x1, layer->W_ff1, model->cache_ff_hidden[l], seq_len, embedding_dim, ffn_dim);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < ffn_dim; d++) {
                float *v = &model->cache_ff_hidden[l][i * ffn_dim + d];
                *v = relu(*v + layer->b_ff1[d]);
            }
        }
        dispatch_matmul(model, model->cache_ff_hidden[l], layer->W_ff2, model->ws_fwd_ff_raw,
                       seq_len, ffn_dim, embedding_dim);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < embedding_dim; d++) {
                model->ws_fwd_ff_raw[i * embedding_dim + d] += layer->b_ff2[d];
            }
        }
        dropout_forward(model->ws_fwd_ff_raw, model->cache_ffn_dropout_mask[l],
                         seq_len * embedding_dim, model->dropout_rate, model->is_training);
        for (size_t i = 0; i < seq_len * embedding_dim; i++) {
            model->ws_fwd_ff_raw[i] += x1[i];
        }
        layer_norm_forward_cached(model->ws_fwd_ff_raw, model->cache_ffn_xhat[l], model->cache_ffn_std[l],
                                   model->cache_hidden[l + 1], layer->ln_gamma_ffn, layer->ln_beta_ffn,
                                   seq_len, embedding_dim, epsilon);
    }

    /* 3. Output projection to vocabulary, from the last layer's last position */
    float *last_hidden = &model->cache_hidden[model->num_layers][(seq_len - 1) * embedding_dim];
    dispatch_matmul(model, last_hidden, model->output_projection, output_logits, 1, embedding_dim, model->vocab_size);

    for (size_t i = 0; i < model->vocab_size; i++) {
        output_logits[i] += model->output_bias[i];
    }

    return MODEL_SUCCESS;
}
