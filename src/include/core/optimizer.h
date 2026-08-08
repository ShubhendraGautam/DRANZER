#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "core/model_types.h"

/* Zero every gradient buffer before a fresh backward pass. Every trainable
 * parameter's gradient is a view into model->grads (see model_types.h), so
 * this is a single memset regardless of how many layers/params the model
 * has - no per-field loop to keep in sync when a new param type is added. */
void model_zero_gradients(neural_model_t *model);

/* param[i] -= lr * grad[i], for `size` elements. Generic: works for a
 * single field or (as model_optimizer_step uses it) the entire flat
 * params/grads buffer in one call. */
void sgd_update(float *param, float *grad, size_t size, float lr);

/* Adam's adaptive step: moment estimates, bias correction, and the update. `t`
 * is the 1-indexed step count, used for the bias correction.
 *
 * No weight decay here. AdamW's decay is decoupled from the adaptive step, and
 * in this project it is also selective - biases and normalization parameters are
 * excluded, which cannot be expressed over a flat range of floats. It is a
 * separate pass in model_optimizer_step(); the policy is core/weight_decay.h. */
void adam_update(float *param, float *grad, float *m, float *v, size_t size,
                  float lr, float beta1, float beta2, float eps, uint32_t t);

/* Global L2-norm gradient clipping across every parameter in
 * model->grads: if the norm exceeds max_norm, scales all gradients down
 * by max_norm/norm. No-op if max_norm <= 0 or the norm is already <= max_norm. */
void clip_gradients(neural_model_t *model, float max_norm);

/* Applies gradient clipping (if model->grad_clip_norm > 0) then the
 * configured optimizer (model->optimizer_type) over the model's entire
 * flat params/grads buffer in one call - adding a new parameter array
 * anywhere in the model requires no change here, since it's already part
 * of the flat buffer by construction.
 *
 * If model->optimizer_type is OPTIMIZER_ADAM and model->adam_m/adam_v
 * haven't been allocated yet, this allocates them on first use (they are
 * NOT allocated by model_new - see model_types.h) and returns
 * MODEL_ALLOCATION_FAILURE if that allocation fails, leaving the model
 * otherwise untouched (no partial update applied). Returns MODEL_SUCCESS
 * for SGD, which never allocates. */
model_errors_t model_optimizer_step(neural_model_t *model);

/* Advances model->learning_rate for the step about to happen. If
 * model->total_steps > 0, applies linear warmup (over warmup_steps) then
 * cosine decay (base_lr -> min_lr, over the remaining steps until
 * total_steps) using model->training_steps as the current step. If
 * total_steps == 0, the schedule is considered unconfigured and this
 * falls back to plateau decay (lr *= 0.95 after 20 steps without a new
 * best loss), matching the model's original behavior. */
void model_lr_schedule_step(neural_model_t *model);

#endif // OPTIMIZER_H
