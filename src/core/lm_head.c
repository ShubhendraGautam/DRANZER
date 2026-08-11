/*
 * All-position output head: forward projection, cross-entropy, and backward.
 * See lm_head.h for why supervising every position is valid here and what it
 * is worth.
 */

#include "core/lm_head.h"
#include "core/matmul_dispatch.h"
#include "common/debug.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

model_errors_t lm_head_forward_all(neural_model_t *model, size_t seq_len) {
    if (!model || !model->ws_logits_all) return MODEL_INVALID_INPUT;
    if (seq_len == 0 || seq_len > model->max_seq_len) return MODEL_INVALID_INPUT;

    size_t embedding_dim = model->embedding_dim;
    size_t vocab_size = model->vocab_size;
    float *hidden = model->cache_hidden[model->num_layers];
    float *logits = model->ws_logits_all;

    /* One [seq_len x embedding_dim] @ [embedding_dim x vocab_size] instead of
     * the last-position path's [1 x embedding_dim] @ [...]. Same weights,
     * same dispatch, seq_len times the rows. */
    model_dispatch_matmul(model, hidden, model->output_projection, logits,
                          seq_len, embedding_dim, vocab_size);

    for (size_t i = 0; i < seq_len; i++) {
        float *row = &logits[i * vocab_size];
        for (size_t v = 0; v < vocab_size; v++) {
            row[v] += model->output_bias[v];
        }
    }

    return MODEL_SUCCESS;
}

model_errors_t lm_head_loss_and_grad_all_masked(
    neural_model_t *model, const uint32_t *targets,
    const uint8_t *position_mask, size_t seq_len,
    float *out_loss, size_t *out_supervised) {
    if (!model || !targets || !model->ws_logits_all || !model->ws_grad_logits_all) {
        return MODEL_INVALID_INPUT;
    }
    if (seq_len == 0 || seq_len > model->max_seq_len) return MODEL_INVALID_INPUT;

    size_t vocab_size = model->vocab_size;
    const float *logits = model->ws_logits_all;
    float *grad = model->ws_grad_logits_all;

    double total_loss = 0.0;
    size_t supervised = 0;

    for (size_t i = 0; i < seq_len; i++) {
        const float *row = &logits[i * vocab_size];
        float *grad_row = &grad[i * vocab_size];
        uint32_t target = targets[i];

        if ((position_mask && !position_mask[i]) ||
            target == LM_HEAD_IGNORE_TARGET || target >= vocab_size) {
            memset(grad_row, 0, vocab_size * sizeof(float));
            continue;
        }

        /* Max-subtracted log-sum-exp in double, as core/evaluation.c does.
         * The single-target path took softmax() then -log(max(p, 1e-7)),
         * which clamps rather than staying stable; with seq_len times as
         * many terms summed into one loss there is no reason to keep the
         * weaker form. */
        float max_logit = row[0];
        for (size_t v = 1; v < vocab_size; v++) {
            if (row[v] > max_logit) max_logit = row[v];
        }

        double exp_sum = 0.0;
        for (size_t v = 0; v < vocab_size; v++) {
            exp_sum += exp((double)row[v] - (double)max_logit);
        }
        double log_sum_exp = log(exp_sum) + (double)max_logit;

        total_loss += log_sum_exp - (double)row[target];

        /* dL/dlogits = softmax(logits) - onehot(target), recovered from the
         * same log-sum-exp rather than a second pass. */
        for (size_t v = 0; v < vocab_size; v++) {
            grad_row[v] = (float)exp((double)row[v] - log_sum_exp);
        }
        grad_row[target] -= 1.0f;

        supervised++;
    }

    if (supervised == 0) {
        /* Nothing to learn from this window. Every gradient row is already
         * zeroed by the loop above, so the caller's backward is a no-op
         * rather than a special case. */
        if (out_loss) *out_loss = 0.0f;
        if (out_supervised) *out_supervised = 0;
        return MODEL_SUCCESS;
    }

    /* Scale to the mean so the loss and gradient do not grow with window
     * length - a window of 128 and a window of 8 stay on the same scale, and
     * so does the learning rate that was tuned against them. */
    float inverse = 1.0f / (float)supervised;
    for (size_t i = 0; i < seq_len * vocab_size; i++) {
        grad[i] *= inverse;
    }

    if (out_loss) *out_loss = (float)(total_loss / (double)supervised);
    if (out_supervised) *out_supervised = supervised;

    DEBUG_PRINT("LM head: %zu/%zu positions supervised, mean loss=%.4f\n",
                supervised, seq_len, total_loss / (double)supervised);

    return MODEL_SUCCESS;
}

model_errors_t lm_head_loss_and_grad_all(neural_model_t *model,
                                         const uint32_t *targets,
                                         size_t seq_len,
                                         float *out_loss,
                                         size_t *out_supervised) {
    return lm_head_loss_and_grad_all_masked(model, targets, NULL, seq_len,
                                            out_loss, out_supervised);
}

model_errors_t lm_head_backward_all(neural_model_t *model, size_t seq_len) {
    if (!model || !model->ws_grad_logits_all) return MODEL_INVALID_INPUT;
    if (seq_len == 0 || seq_len > model->max_seq_len) return MODEL_INVALID_INPUT;

    size_t embedding_dim = model->embedding_dim;
    size_t vocab_size = model->vocab_size;
    float *grad = model->ws_grad_logits_all;
    float *hidden = model->cache_hidden[model->num_layers];

    /* Bias gradient: one column sum per vocabulary entry across positions. */
    for (size_t i = 0; i < seq_len; i++) {
        const float *grad_row = &grad[i * vocab_size];
        for (size_t v = 0; v < vocab_size; v++) {
            model->output_bias_grad[v] += grad_row[v];
        }
    }

    model_dispatch_backward_weight(model, hidden, grad,
                                   model->output_projection_grad,
                                   seq_len, embedding_dim, vocab_size);

    /* Every position now feeds the head, so unlike the single-target path
     * there is no zero-everywhere-but-one row structure to preserve - but
     * matmul_backward_input accumulates, so the destination still has to
     * start at zero. */
    memset(model->ws_dhidden_in, 0, seq_len * embedding_dim * sizeof(float));
    model_dispatch_backward_input(model, grad, model->output_projection,
                                  model->ws_dhidden_in,
                                  seq_len, embedding_dim, vocab_size);

    return MODEL_SUCCESS;
}
