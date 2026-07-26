#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include "core/model_types.h"
#include <stdio.h>

/**
 * Save model weights to file
 */
model_errors_t model_save(neural_model_t *model, const char *filename);

/**
 * Load model weights from file
 */
model_errors_t model_load(neural_model_t *model, const char *filename);

/**
 * Write model dimensions + all weights (not gradients, not optimizer
 * state, not training metrics) to an already-open file stream. Shared by
 * model_save and checkpoint_save so the on-disk weight format has one
 * source of truth. Every trainable weight is a contiguous view into
 * model->params (model_types.h), so this is a single fwrite instead of
 * one line per named field.
 */
model_errors_t model_write_state(const neural_model_t *model, FILE *f);

/**
 * Read model dimensions + all weights from an already-open file stream,
 * reinitializing *model (via model_new) if its current dimensions don't
 * match what's in the stream. Shared by model_load and checkpoint_load.
 */
model_errors_t model_read_state(neural_model_t *model, FILE *f);

#endif // SERIALIZATION_H
