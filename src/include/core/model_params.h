#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#include "core/model_types.h"
#include <stddef.h>

/* An addressable inventory of a model's trainable tensors.
 *
 * The flat `params` buffer is what makes the optimizer and serialization one
 * loop instead of one line per field, and it is exactly wrong for anything that
 * needs to treat tensors differently: a quantizer needs a tensor's shape to
 * pick a scale axis, and error attribution needs its name to report against.
 * This walks the same layout model_new() builds and hands back a description of
 * each tensor, so neither of those has to re-derive the layout and drift from
 * it.
 *
 * The views point into `model->params`; nothing here owns memory, and the
 * descriptors are invalidated by model_free(). */

typedef enum {
    /* A matmul operand: `rows` is the input channel count and `cols` the
     * output channel count, because this project computes C = A @ W. Anything
     * that quantizes per output channel wants the column axis - see
     * core/quantize.h, where getting this backwards is a documented trap. */
    PARAM_KIND_PROJECTION = 0,
    /* The token embedding table: two-dimensional, but indexed rather than
     * multiplied, so `rows` is the vocabulary and `cols` the embedding width.
     * Kept distinct from PROJECTION because a lookup has no reduction axis, so
     * the argument for one scale axis over another does not apply to it. */
    PARAM_KIND_EMBEDDING,
    /* One-dimensional additive term: `rows` is 1. */
    PARAM_KIND_BIAS,
    /* One-dimensional layer-norm scale or offset: `rows` is 1. Small, and
     * directly multiplying a normalized activation, which is why quantization
     * policies conventionally leave these alone. */
    PARAM_KIND_NORM,
    PARAM_KIND_COUNT
} param_kind_t;

typedef struct {
    /* Stable identifier, e.g. "token_embeddings", "layer0.W_q",
     * "layer3.b_ff1". Stable enough to key a CSV column on. */
    char name[48];
    float *values;      /* view into model->params */
    size_t rows;
    size_t cols;
    param_kind_t kind;
    size_t layer;       /* originating layer, or SIZE_MAX for global tensors */
} param_tensor_t;

/* Fill `out` with up to `capacity` descriptors and return how many the model
 * has. A return value greater than `capacity` means the array was too small and
 * only the first `capacity` were written; model_param_tensor_count() sizes it
 * without a dry run. `out` may be NULL when `capacity` is 0.
 *
 * The order is the layout order - globals, then each layer in turn - so a
 * report generated from it reads in the order the forward pass touches. */
size_t model_param_tensors(const neural_model_t *model,
                           param_tensor_t *out, size_t capacity);

/* How many descriptors model_param_tensors() would produce. */
size_t model_param_tensor_count(const neural_model_t *model);

/* Stable lowercase names ("projection", "embedding", "bias", "norm"). */
const char *param_kind_name(param_kind_t kind);

#endif /* MODEL_PARAMS_H */
