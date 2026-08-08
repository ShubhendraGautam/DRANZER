/*
 * Optimizer: SGD, AdamW, global gradient-norm clipping, and the LR
 * schedule. Operates generically over the model's flat params/grads/adam_m
 * /adam_v buffers (model_types.h), so nothing here needs updating when a
 * new parameter array is added anywhere in the model.
 */

#include "core/optimizer.h"
#include "core/weight_decay.h"
#include "common/debug.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define PI_F 3.14159265358979323846f

void model_zero_gradients(neural_model_t *model) {
    memset(model->grads, 0, model->total_param_count * sizeof(float));
}

void sgd_update(float *param, float *grad, size_t size, float lr) {
    for (size_t i = 0; i < size; i++) {
        param[i] -= lr * grad[i];
    }
}

/* The adaptive step alone. Weight decay is NOT applied here: it is decoupled, so
 * it does not belong inside the moment update, and it is selective, so it cannot
 * be expressed over a flat range of floats at all - see core/weight_decay.h.
 * model_optimizer_step() applies it in its own pass immediately before this. */
void adam_update(float *param, float *grad, float *m, float *v, size_t size,
                  float lr, float beta1, float beta2, float eps, uint32_t t) {
    float bias_correction1 = 1.0f - powf(beta1, (float)t);
    float bias_correction2 = 1.0f - powf(beta2, (float)t);

    for (size_t i = 0; i < size; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];

        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;

        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void clip_gradients(neural_model_t *model, float max_norm) {
    if (max_norm <= 0.0f) return;

    double sum_sq = 0.0;
    for (size_t i = 0; i < model->total_param_count; i++) {
        double g = model->grads[i];
        sum_sq += g * g;
    }
    float norm = sqrtf((float)sum_sq);

    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < model->total_param_count; i++) {
            model->grads[i] *= scale;
        }
        DEBUG_PRINT("Gradient norm %.4f clipped to %.4f (scale=%.6f)\n", norm, max_norm, scale);
    }
}

model_errors_t model_optimizer_step(neural_model_t *model) {
    if (model->grad_clip_norm > 0.0f) {
        clip_gradients(model, model->grad_clip_norm);
    }

    if (model->optimizer_type == OPTIMIZER_ADAM) {
        if (!model->adam_m) {
            model->adam_m = calloc(model->total_param_count, sizeof(float));
            model->adam_v = calloc(model->total_param_count, sizeof(float));
            if (!model->adam_m || !model->adam_v) {
                free(model->adam_m);
                free(model->adam_v);
                model->adam_m = NULL;
                model->adam_v = NULL;
                return MODEL_ALLOCATION_FAILURE;
            }
            DEBUG_PRINT("Lazily allocated Adam moment buffers (%zu floats each)\n", model->total_param_count);
        }

        model->adam_t++;
        /* Decay first, then the adaptive step, matching the order the fused
         * version used and keeping this step's update identical to AdamW's for
         * the tensors that are decayed. */
        model_errors_t decay_rc = model_apply_weight_decay(model, model->learning_rate,
                                                           model->weight_decay);
        if (decay_rc != MODEL_SUCCESS) return decay_rc;

        adam_update(model->params, model->grads, model->adam_m, model->adam_v,
                    model->total_param_count, model->learning_rate,
                    model->adam_beta1, model->adam_beta2, model->adam_eps,
                    model->adam_t);
    } else {
        sgd_update(model->params, model->grads, model->total_param_count, model->learning_rate);
    }

    return MODEL_SUCCESS;
}

void model_lr_schedule_step(neural_model_t *model) {
    if (model->total_steps > 0 && model->total_steps > model->warmup_steps) {
        uint32_t step = model->training_steps;

        if (step < model->warmup_steps) {
            model->learning_rate = model->base_lr * ((float)(step + 1) / (float)model->warmup_steps);
        } else if (step < model->total_steps) {
            float progress = (float)(step - model->warmup_steps) /
                              (float)(model->total_steps - model->warmup_steps);
            float cosine = 0.5f * (1.0f + cosf(PI_F * progress));
            model->learning_rate = model->min_lr + (model->base_lr - model->min_lr) * cosine;
        } else {
            model->learning_rate = model->min_lr;
        }
        model->metrics.learning_rate = model->learning_rate;
        return;
    }

    /* No schedule configured: fall back to the original plateau decay. */
    if (model->metrics.steps_without_improvement > 20) {
        model->metrics.learning_rate *= 0.95f;
        model->learning_rate = model->metrics.learning_rate;
        model->metrics.steps_without_improvement = 0;
    }
}
