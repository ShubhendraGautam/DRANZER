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
#include "core/lm_head.h"
#include "core/matmul_dispatch.h"
#include "backends/gpu/gpu_matmul.h"
#include "core/matmul.h"
#include "common/debug.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/* The backward matmul dispatch policy and its measured GPU thresholds moved
 * to core/matmul_dispatch.{c,h} when core/lm_head.c became a second caller;
 * call sites below use model_dispatch_backward_{input,weight}() directly. */

/* Backprop from dL/d(final hidden), which the caller has already placed in
 * model->ws_dhidden_in, down through every transformer layer and into the
 * token embedding gradients.
 *
 * This is the whole per-layer backward pass and it is shared verbatim by
 * both training entry points below. Only the head above it differs between
 * them: one seeds ws_dhidden_in at the last position, the other at every
 * position. Nothing in here inspects which - it is the same arithmetic on
 * whatever gradient arrives, which is exactly why supervising every position
 * needed no change to the layer stack. */
static void backward_layer_stack(neural_model_t *model,
                                 const uint32_t *token_ids,
                                 size_t seq_len) {
    size_t embedding_dim = model->embedding_dim;
    size_t ffn_dim = embedding_dim * 4;

    for (size_t li = 0; li < model->num_layers; li++) {
        size_t l = model->num_layers - 1 - li;
        transformer_layer_t *layer = &model->layers[l];

        if (model->cache_padding_mask_active) {
            for (size_t i = 0; i < seq_len; i++) {
                if (!model->cache_padding_mask[i]) {
                    memset(&model->ws_dhidden_in[i * embedding_dim], 0,
                           embedding_dim * sizeof(float));
                }
            }
        }

        /* FFN-LN backward: dL/dx2 (ws_dhidden_in) -> dL/d(post-dropout ffn raw) (ws_d_s2) */
        if (model_uses_rmsnorm(model)) {
            rms_norm_backward(
                model->ws_dhidden_in, model->cache_ffn_xhat[l],
                model->cache_ffn_std[l], layer->ln_gamma_ffn,
                layer->ln_gamma_ffn_grad, model->ws_d_s2,
                seq_len, embedding_dim);
        } else {
            layer_norm_backward(
                model->ws_dhidden_in, model->cache_ffn_xhat[l],
                model->cache_ffn_std[l], layer->ln_gamma_ffn,
                layer->ln_gamma_ffn_grad, layer->ln_beta_ffn_grad,
                model->ws_d_s2, seq_len, embedding_dim);
        }

        /* ws_d_s2 is dL/d(post-dropout ffn_raw) - the value that was added
         * to the residual branch directly (dropout only sits on the FFN
         * path, not the skip connection), so it is ALSO the residual
         * branch's contribution to dL/dx1 as-is. The FFN path itself needs
         * dropout backward applied first, on a separate copy, before
         * continuing through b_ff2/W_ff2/activation/W_ff1. */
        memcpy(model->ws_d_ffn_dropout, model->ws_d_s2, seq_len * embedding_dim * sizeof(float));
        dropout_backward(model->ws_d_ffn_dropout, model->cache_ffn_dropout_mask[l],
                          seq_len * embedding_dim, model->dropout_rate);

        for (size_t d = 0; d < embedding_dim; d++) {
            float sum = 0.0f;
            for (size_t i = 0; i < seq_len; i++) sum += model->ws_d_ffn_dropout[i * embedding_dim + d];
            layer->b_ff2_grad[d] += sum;
        }
        model_dispatch_backward_weight(model, model->cache_ff_hidden[l], model->ws_d_ffn_dropout, layer->W_ff2_grad,
                                seq_len, ffn_dim, embedding_dim);

        memset(model->ws_d_ff_hidden, 0, seq_len * ffn_dim * sizeof(float));
        model_dispatch_backward_input(model, model->ws_d_ffn_dropout, layer->W_ff2, model->ws_d_ff_hidden,
                               seq_len, ffn_dim, embedding_dim);

        /* Activation backward. SwiGLU splits the incoming gradient between
         * silu(u) and its parallel linear gate v. */
        if (model_uses_swiglu(model)) {
            for (size_t idx = 0; idx < seq_len * ffn_dim; idx++) {
                float upstream = model->ws_d_ff_hidden[idx];
                float pre_activation =
                    model->cache_ff_pre_activation[l][idx];
                float gate = model->cache_ff_gate[l][idx];
                model->ws_d_ff_gate[idx] = upstream * silu(pre_activation);
                model->ws_d_ff_hidden[idx] =
                    upstream * gate * silu_derivative(pre_activation);
            }
        } else {
            /* ReLU can use its cached output as the sign indicator; GELU
             * needs the pre-activation cached by forward. */
            for (size_t idx = 0; idx < seq_len * ffn_dim; idx++) {
                float derivative = model_uses_gelu(model)
                    ? gelu_derivative(model->cache_ff_pre_activation[l][idx])
                    : relu_derivative(model->cache_ff_hidden[l][idx]);
                model->ws_d_ff_hidden[idx] *= derivative;
            }
        }

        for (size_t d = 0; d < ffn_dim; d++) {
            float sum = 0.0f;
            for (size_t i = 0; i < seq_len; i++) sum += model->ws_d_ff_hidden[i * ffn_dim + d];
            layer->b_ff1_grad[d] += sum;
        }
        model_dispatch_backward_weight(model, model->cache_attn_ln_out[l], model->ws_d_ff_hidden, layer->W_ff1_grad,
                                seq_len, embedding_dim, ffn_dim);
        if (model_uses_swiglu(model)) {
            for (size_t d = 0; d < ffn_dim; d++) {
                float sum = 0.0f;
                for (size_t i = 0; i < seq_len; i++)
                    sum += model->ws_d_ff_gate[i * ffn_dim + d];
                layer->b_ff_gate_grad[d] += sum;
            }
            model_dispatch_backward_weight(
                model, model->cache_attn_ln_out[l], model->ws_d_ff_gate,
                layer->W_ff_gate_grad, seq_len, embedding_dim, ffn_dim);
        }

        memset(model->ws_d_x1_total, 0, seq_len * embedding_dim * sizeof(float));
        model_dispatch_backward_input(model, model->ws_d_ff_hidden, layer->W_ff1, model->ws_d_x1_total,
                               seq_len, embedding_dim, ffn_dim);
        if (model_uses_swiglu(model)) {
            model_dispatch_backward_input(
                model, model->ws_d_ff_gate, layer->W_ff_gate,
                model->ws_d_x1_total, seq_len, embedding_dim, ffn_dim);
        }
        for (size_t i = 0; i < seq_len * embedding_dim; i++) {
            model->ws_d_x1_total[i] += model->ws_d_s2[i]; /* + residual branch (bypasses dropout) */
        }

        /* Attn-LN backward: dL/dx1 (ws_d_x1_total) -> dL/d(post-dropout attn raw) (ws_d_s1) */
        if (model_uses_rmsnorm(model)) {
            rms_norm_backward(
                model->ws_d_x1_total, model->cache_attn_xhat[l],
                model->cache_attn_std[l], layer->ln_gamma_attn,
                layer->ln_gamma_attn_grad, model->ws_d_s1,
                seq_len, embedding_dim);
        } else {
            layer_norm_backward(
                model->ws_d_x1_total, model->cache_attn_xhat[l],
                model->cache_attn_std[l], layer->ln_gamma_attn,
                layer->ln_gamma_attn_grad, layer->ln_beta_attn_grad,
                model->ws_d_s1, seq_len, embedding_dim);
        }

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
                if (model->cache_padding_mask_active &&
                    !model->cache_padding_mask[i]) continue;
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
}

model_errors_t model_accumulate_gradients_all_masked(
    neural_model_t *model, uint32_t *token_ids, const uint32_t *targets,
    size_t seq_len, const model_attention_mask_t *mask,
    float *out_loss, size_t *out_supervised) {
    if (!model || model->params_read_only || !token_ids || !targets) {
        return MODEL_INVALID_INPUT;
    }
    if (seq_len == 0 || seq_len > model->max_seq_len) return MODEL_INVALID_INPUT;

    /* ---- Forward: layer stack, then the head over every position ---- */
    model->is_training = 1;
    model_errors_t rc = model_forward_hidden_masked(model, token_ids, seq_len,
                                                    mask);
    if (rc != MODEL_SUCCESS) return rc;

    rc = lm_head_forward_all(model, seq_len);
    if (rc != MODEL_SUCCESS) return rc;

    float loss = 0.0f;
    size_t supervised = 0;
    rc = lm_head_loss_and_grad_all_masked(
        model, targets, mask ? mask->padding_mask : NULL, seq_len,
        &loss, &supervised);
    if (rc != MODEL_SUCCESS) return rc;

    model->current_loss = loss;
    if (out_loss) *out_loss = loss;
    if (out_supervised) *out_supervised = supervised;

    if (supervised == 0) {
        /* No supervised position: the gradient this window contributes is
         * exactly zero, so running the backward would add nothing but would
         * still cost a full pass. */
        return MODEL_SUCCESS;
    }

    /* ---- Backward ---- */
    rc = lm_head_backward_all(model, seq_len);
    if (rc != MODEL_SUCCESS) return rc;

    backward_layer_stack(model, token_ids, seq_len);

    return MODEL_SUCCESS;
}

model_errors_t model_accumulate_gradients_all(neural_model_t *model,
                                              uint32_t *token_ids,
                                              const uint32_t *targets,
                                              size_t seq_len,
                                              float *out_loss,
                                              size_t *out_supervised) {
    return model_accumulate_gradients_all_masked(
        model, token_ids, targets, seq_len, NULL, out_loss, out_supervised);
}

model_errors_t model_accumulate_gradients(neural_model_t *model,
                                          uint32_t *token_ids,
                                          uint32_t target_id,
                                          size_t seq_len,
                                          float *out_loss) {

    if (!model || model->params_read_only || !token_ids ||
        target_id >= model->vocab_size) {
        return MODEL_INVALID_INPUT;
    }
    if (seq_len == 0 || seq_len > model->max_seq_len) {
        return MODEL_INVALID_INPUT;
    }

    size_t embedding_dim = model->embedding_dim;

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
    /* Only the LAST sequence position feeds the output head, so dL/dhidden
     * is zero everywhere else - zero the whole buffer, then fill just the
     * last row. This is what model_accumulate_gradients_all() above replaces:
     * there, every row is filled, and the rest of the backward is identical. */
    memset(model->ws_dhidden_in, 0, seq_len * embedding_dim * sizeof(float));
    model_errors_t head_rc = lm_head_project_backward(
        model, last_hidden, grad_logits,
        &model->ws_dhidden_in[(seq_len - 1) * embedding_dim], 1);
    if (head_rc != MODEL_SUCCESS) return head_rc;

    backward_layer_stack(model, token_ids, seq_len);

    if (out_loss) *out_loss = loss;
    return MODEL_SUCCESS;
}

model_errors_t model_apply_accumulated_gradients(neural_model_t *model,
                                                  size_t sample_count,
                                                  float average_loss) {
    if (!model || model->params_read_only || sample_count == 0) {
        return MODEL_INVALID_INPUT;
    }

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
