/*
 * Training step: forward pass (via transformer.c), cross-entropy loss,
 * full backpropagation through the output head, every transformer layer
 * (in reverse, including dropout and layer norm), and the token
 * embeddings, then an optimizer step (optimizer.c).
 */

#include "core/training.h"
#include "core/transformer.h"
#include "core/tensor_ops.h"
#include "core/optimizer.h"
#include "backends/gpu/gpu_matmul.h"
#include "core/matmul.h"
#include "common/debug.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Backward matmuls reach the GPU only above a measured size, unlike the
 * forward ones in transformer.c which go whenever `--gpu` is set.
 *
 * The asymmetry is structural, not a policy preference. A forward matmul
 * moves two buffers (activations up, result down) and its weight operand is
 * already device-resident. A backward matmul moves four: both inputs up, and
 * the destination both up and down, because these functions accumulate into
 * a gradient that earlier call sites in the same minibatch have already
 * written. Below a certain amount of arithmetic, those two extra transfers
 * cost more than the kernel saves, and the GPU path is a straight loss.
 *
 * The thresholds are the measured crossovers from `./gpu_latency.out` on this
 * project's MX450 test system (docs/gpu.md), rounded to a power of two. They
 * have now been re-derived twice, both times because the CPU got faster:
 *
 *   2^20 / 2^23   original, against a matmul_backward_weight() that strided
 *                 both operands in its innermost loop
 *   2^23 / 2^23   after that loop order was fixed (CPU 5.6-21.8x faster)
 *   2^25 / 2^26   after the backward kernels gained AVX-512 (CPU ~2-3x faster
 *                 again)
 *
 * The GPU never got worse. Each time, the baseline it is measured against got
 * better, and the range of shapes where a round trip pays shrank. That is the
 * standing caution for these constants and for every speedup ratio in this
 * project: a threshold measures two implementations, not one.
 *
 * Only the weight threshold is measured. backward_weight wins 1.62x and 1.32x
 * at the two largest shapes the benchmark issues (both 2^25 multiply-
 * accumulates) and loses at 2^23, so 2^25 sits on the far side of a crossover
 * that was actually observed. backward_input never won at any measured shape -
 * it reaches 0.95x at 2^25 and is still climbing - so its threshold is an
 * extrapolation of that trend, deliberately set beyond every shape this
 * project benchmarks. In practice that means backward_input runs on the CPU
 * for every model here, which is what the measurements support; the path stays
 * for larger models and faster cards, where it should be re-measured rather
 * than trusted.
 *
 * Below either threshold the CPU path is used, which is always correct and
 * never slower than not having the GPU at all. */
#define GPU_BACKWARD_WEIGHT_MIN_WORK (1u << 25)
#define GPU_BACKWARD_INPUT_MIN_WORK  (1u << 26)

/* True when this shape is worth the round trip. Guards the multiplication
 * against overflow rather than trusting model dimensions to stay small. */
static int gpu_backward_worthwhile(size_t m, size_t k, size_t n, size_t min_work) {
    if (m == 0 || k == 0 || n == 0) return 0;
    if (k > SIZE_MAX / m) return 1; /* astronomically large: certainly worth it */
    size_t mk = m * k;
    if (n > SIZE_MAX / mk) return 1;
    return mk * n >= min_work;
}

/* dA (m x k) += dC (m x n) @ B_transposed, on the GPU when that pays.
 * A failed GPU call is not an error: it falls through to the CPU, which is
 * what also happens when no GPU exists at all. */
static void dispatch_backward_input(neural_model_t *model, float *dC, float *B,
                                    float *dA, size_t m, size_t k, size_t n) {
    if (model->use_gpu &&
        gpu_backward_worthwhile(m, k, n, GPU_BACKWARD_INPUT_MIN_WORK) &&
        gpu_matmul_available() &&
        gpu_matmul_backward_input(dC, B, dA, m, k, n) == 0) {
        return;
    }
    matmul_backward_input(dC, B, dA, m, k, n);
}

/* dB (k x n) += A_transposed @ dC (m x n), same contract. */
static void dispatch_backward_weight(neural_model_t *model, float *A, float *dC,
                                     float *dB, size_t m, size_t k, size_t n) {
    if (model->use_gpu &&
        gpu_backward_worthwhile(m, k, n, GPU_BACKWARD_WEIGHT_MIN_WORK) &&
        gpu_matmul_available() &&
        gpu_matmul_backward_weight(A, dC, dB, m, k, n) == 0) {
        return;
    }
    matmul_backward_weight(A, dC, dB, m, k, n);
}

model_errors_t model_accumulate_gradients(neural_model_t *model,
                                          uint32_t *token_ids,
                                          uint32_t target_id,
                                          size_t seq_len,
                                          float *out_loss) {

    if (!model || !token_ids || target_id >= model->vocab_size) {
        return MODEL_INVALID_INPUT;
    }
    if (seq_len == 0 || seq_len > model->max_seq_len) {
        return MODEL_INVALID_INPUT;
    }

    size_t embedding_dim = model->embedding_dim;
    size_t ffn_dim = embedding_dim * 4;

    /* ---- Forward pass (populates the activation cache, including dropout masks) ---- */
    model->is_training = 1;
    float *logits = model->ws_logits;
    model_forward(model, token_ids, seq_len, logits);

    softmax(logits, model->vocab_size);
    float loss = -logf(fmaxf(logits[target_id], 1e-7f));
    model->current_loss = loss;

    DEBUG_PRINT("Training step: loss=%.4f, target_id=%u\n", loss, target_id);

    float *grad_logits = model->ws_grad_logits;
    memcpy(grad_logits, logits, model->vocab_size * sizeof(float));
    grad_logits[target_id] -= 1.0f; // Gradient for cross-entropy

    /* ---- Backward pass ----
     * Gradients are added to the existing flat buffer. The caller decides
     * where a minibatch/accumulation cycle begins by zeroing once, then
     * applies the averaged result with model_apply_accumulated_gradients(). */

    float *last_hidden = &model->cache_hidden[model->num_layers][(seq_len - 1) * embedding_dim];

    for (size_t i = 0; i < model->vocab_size; i++) {
        model->output_bias_grad[i] += grad_logits[i];
    }
    dispatch_backward_weight(model, last_hidden, grad_logits, model->output_projection_grad,
                            1, embedding_dim, model->vocab_size);

    /* Only the LAST sequence position feeds the output head, so dL/dhidden
     * is zero everywhere else - zero the whole buffer, then fill just the
     * last row. */
    memset(model->ws_dhidden_in, 0, seq_len * embedding_dim * sizeof(float));
    dispatch_backward_input(model, grad_logits, model->output_projection,
                           &model->ws_dhidden_in[(seq_len - 1) * embedding_dim],
                           1, embedding_dim, model->vocab_size);

    for (size_t li = 0; li < model->num_layers; li++) {
        size_t l = model->num_layers - 1 - li;
        transformer_layer_t *layer = &model->layers[l];

        /* FFN-LN backward: dL/dx2 (ws_dhidden_in) -> dL/d(post-dropout ffn raw) (ws_d_s2) */
        layer_norm_backward(model->ws_dhidden_in, model->cache_ffn_xhat[l], model->cache_ffn_std[l],
                             layer->ln_gamma_ffn, layer->ln_gamma_ffn_grad, layer->ln_beta_ffn_grad,
                             model->ws_d_s2, seq_len, embedding_dim);

        /* ws_d_s2 is dL/d(post-dropout ffn_raw) - the value that was added
         * to the residual branch directly (dropout only sits on the FFN
         * path, not the skip connection), so it is ALSO the residual
         * branch's contribution to dL/dx1 as-is. The FFN path itself needs
         * dropout backward applied first, on a separate copy, before
         * continuing through b_ff2/W_ff2/ReLU/W_ff1. */
        memcpy(model->ws_d_ffn_dropout, model->ws_d_s2, seq_len * embedding_dim * sizeof(float));
        dropout_backward(model->ws_d_ffn_dropout, model->cache_ffn_dropout_mask[l],
                          seq_len * embedding_dim, model->dropout_rate);

        for (size_t d = 0; d < embedding_dim; d++) {
            float sum = 0.0f;
            for (size_t i = 0; i < seq_len; i++) sum += model->ws_d_ffn_dropout[i * embedding_dim + d];
            layer->b_ff2_grad[d] += sum;
        }
        dispatch_backward_weight(model, model->cache_ff_hidden[l], model->ws_d_ffn_dropout, layer->W_ff2_grad,
                                seq_len, ffn_dim, embedding_dim);

        memset(model->ws_d_ff_hidden, 0, seq_len * ffn_dim * sizeof(float));
        dispatch_backward_input(model, model->ws_d_ffn_dropout, layer->W_ff2, model->ws_d_ff_hidden,
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
        dispatch_backward_weight(model, model->cache_attn_ln_out[l], model->ws_d_ff_hidden, layer->W_ff1_grad,
                                seq_len, embedding_dim, ffn_dim);

        memset(model->ws_d_x1_total, 0, seq_len * embedding_dim * sizeof(float));
        dispatch_backward_input(model, model->ws_d_ff_hidden, layer->W_ff1, model->ws_d_x1_total,
                               seq_len, embedding_dim, ffn_dim);
        for (size_t i = 0; i < seq_len * embedding_dim; i++) {
            model->ws_d_x1_total[i] += model->ws_d_s2[i]; /* + residual branch (bypasses dropout) */
        }

        /* Attn-LN backward: dL/dx1 (ws_d_x1_total) -> dL/d(post-dropout attn raw) (ws_d_s1) */
        layer_norm_backward(model->ws_d_x1_total, model->cache_attn_xhat[l], model->cache_attn_std[l],
                             layer->ln_gamma_attn, layer->ln_gamma_attn_grad, layer->ln_beta_attn_grad,
                             model->ws_d_s1, seq_len, embedding_dim);

        /* ws_d_s1 is dL/d(post-dropout attn_raw) AND (dropout bypassed) the
         * residual branch's contribution to dL/dhidden[l]. The attention
         * path needs dropout backward applied first, on a separate copy,
         * before entering multihead_attention_backward. */
        memcpy(model->ws_d_attn_dropout, model->ws_d_s1, seq_len * embedding_dim * sizeof(float));
        dropout_backward(model->ws_d_attn_dropout, model->cache_attn_dropout_mask[l],
                          seq_len * embedding_dim, model->dropout_rate);

        memcpy(model->ws_dhidden_out, model->ws_d_s1, seq_len * embedding_dim * sizeof(float));
        multihead_attention_backward(model, l, seq_len, model->ws_d_attn_dropout, model->ws_dhidden_out);

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

    if (out_loss) *out_loss = loss;
    return MODEL_SUCCESS;
}

model_errors_t model_apply_accumulated_gradients(neural_model_t *model,
                                                  size_t sample_count,
                                                  float average_loss) {
    if (!model || sample_count == 0) return MODEL_INVALID_INPUT;

    float inverse_count = 1.0f / (float)sample_count;
    for (size_t i = 0; i < model->total_param_count; i++) {
        model->grads[i] *= inverse_count;
    }

    /* ---- Optimizer step (SGD or AdamW, with optional grad-norm clipping) ---- */
    model_errors_t opt_rc = model_optimizer_step(model);
    if (opt_rc != MODEL_SUCCESS) {
        return opt_rc; /* e.g. lazy Adam moment-buffer allocation failed; params left untouched */
    }
    /* Every weight just changed - any GPU-resident cached copy (gpu_matmul.c)
     * is now stale and must be re-uploaded before its next use. */
    gpu_matmul_invalidate_weights();

    model->training_steps++;
    model->current_loss = average_loss;

    /* Phase 2: Update learning metrics */
    if (model->metrics.history_size < model->metrics.history_capacity) {
        model->metrics.loss_history[model->metrics.history_size] = average_loss;
        model->metrics.history_size++;
    }

    if (average_loss < model->metrics.best_loss) {
        model->metrics.best_loss = average_loss;
        model->metrics.steps_without_improvement = 0;
    } else {
        model->metrics.steps_without_improvement++;
    }

    if (average_loss > model->metrics.worst_loss) {
        model->metrics.worst_loss = average_loss;
    }

    model->metrics.avg_loss =
        (model->metrics.avg_loss * (model->training_steps - 1) + average_loss) /
        model->training_steps;

    /* LR schedule (warmup+cosine if configured, else plateau decay) for the *next* step. */
    model_lr_schedule_step(model);

    return MODEL_SUCCESS;
}

model_errors_t model_train_step(neural_model_t *model,
                                uint32_t *token_ids,
                                uint32_t target_id,
                                size_t seq_len) {
    if (!model) return MODEL_INVALID_INPUT;
    model_zero_gradients(model);
    float loss = 0.0f;
    model_errors_t rc = model_accumulate_gradients(
        model, token_ids, target_id, seq_len, &loss);
    if (rc != MODEL_SUCCESS) return rc;
    return model_apply_accumulated_gradients(model, 1, loss);
}

/* Predict next token */
uint32_t model_predict_next_token(neural_model_t *model,
                                  uint32_t *token_ids,
                                  size_t seq_len) {

    int was_training = model->is_training;
    model->is_training = 0; /* inference: dropout disabled */

    float *logits = model->ws_logits;
    model_forward(model, token_ids, seq_len, logits);

    model->is_training = was_training;

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
