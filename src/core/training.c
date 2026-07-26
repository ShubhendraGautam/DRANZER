/*
 * Training step: forward pass (via transformer.c), cross-entropy loss,
 * full backpropagation through the output head, every transformer layer
 * (in reverse, including dropout and layer norm), and the token
 * embeddings, then an optimizer step (optimizer.c).
 */

#include "include/training.h"
#include "include/transformer.h"
#include "include/tensor_ops.h"
#include "include/optimizer.h"
#include "include/debug.h"
#include <string.h>
#include <math.h>

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
        matmul_backward_weight(model->cache_ff_hidden[l], model->ws_d_ffn_dropout, layer->W_ff2_grad,
                                seq_len, ffn_dim, embedding_dim);

        memset(model->ws_d_ff_hidden, 0, seq_len * ffn_dim * sizeof(float));
        matmul_backward_input(model->ws_d_ffn_dropout, layer->W_ff2, model->ws_d_ff_hidden,
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

    /* ---- Optimizer step (SGD or AdamW, with optional grad-norm clipping) ---- */
    model_errors_t opt_rc = model_optimizer_step(model);
    if (opt_rc != MODEL_SUCCESS) {
        return opt_rc; /* e.g. lazy Adam moment-buffer allocation failed; params left untouched */
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

    /* LR schedule (warmup+cosine if configured, else plateau decay) for the *next* step. */
    model_lr_schedule_step(model);

    return MODEL_SUCCESS;
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
