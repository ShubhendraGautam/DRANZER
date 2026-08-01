#ifndef EVALUATION_H
#define EVALUATION_H

#include "core/model_types.h"

/**
 * Compute next-token cross-entropy without backpropagation or optimizer,
 * scheduler, RNG, parameter, gradient, or learning-metric updates.
 * Dropout is disabled for the forward pass and model->is_training is
 * restored before return.
 */
model_errors_t model_evaluate_step(neural_model_t *model,
                                   uint32_t *token_ids,
                                   uint32_t target_id,
                                   size_t seq_len,
                                   double *out_loss);

#endif /* EVALUATION_H */
