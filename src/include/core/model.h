#ifndef MODEL_H
#define MODEL_H

/* Facade header: external callers (main.c, cli.c, checkpoint.c) just
 * `#include "core/model.h"` and get the full model API, without needing to
 * know it's actually implemented across several focused modules:
 *   model_types.h   - neural_model_t, transformer_layer_t, enums
 *   tensor_ops.*     - matmul/softmax/layernorm/dropout primitives
 *   transformer.*    - attention + the stacked forward pass
 *   optimizer.*      - SGD/AdamW, grad clipping, LR schedule
 *   training.*       - model_train_step, model_predict_next_token
 *   serialization.*  - model_save/model_load and the shared write/read state
 *   metrics.*        - learning_metrics_t accessors
 *   model.c          - just model_new/model_free (this header's own declarations)
 * Internal modules should prefer including only the narrower header(s)
 * they actually need instead of this umbrella, to keep compile-time
 * coupling honest. */

#include "core/model_types.h"
#include "core/tensor_ops.h"
#include "core/transformer.h"
#include "core/optimizer.h"
#include "core/training.h"
#include "core/serialization.h"
#include "core/metrics.h"

/**
 * Initialize a new neural model with random weights. Every trainable
 * parameter (token embeddings, output head, every layer's weights) is
 * allocated as one contiguous flat buffer (model->params/model->grads);
 * the named fields on the model and on each transformer_layer_t are views
 * into it. model->adam_m/model->adam_v (Adam's moment estimates) are
 * allocated lazily on first use, not here - see optimizer.c.
 *
 * Optimizer/schedule/dropout are initialized to sane defaults (AdamW,
 * grad-norm clipping at 1.0, no LR schedule, no dropout) - override any of
 * those fields directly on the model before training starts, the same way
 * the caller already overrides learning_rate.
 */
model_errors_t model_new(neural_model_t *model,
                         size_t vocab_size,
                         size_t embedding_dim,
                         size_t num_heads,
                         size_t num_layers,
                         size_t max_seq_len);

/**
 * Free all model resources
 */
void model_free(neural_model_t *model);

#endif // MODEL_H
