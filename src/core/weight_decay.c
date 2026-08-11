/*
 * The weight-decay policy - see core/weight_decay.h for why it is expressed over
 * the tensor inventory rather than over the flat parameter vector.
 */

#include "core/weight_decay.h"
#include <stdlib.h>

int weight_decay_applies_to(param_kind_t kind) {
    switch (kind) {
        case PARAM_KIND_PROJECTION:
        case PARAM_KIND_EMBEDDING:
            return 1;
        case PARAM_KIND_BIAS:
        case PARAM_KIND_NORM:
            return 0;
        case PARAM_KIND_COUNT:
        default:
            /* Not a kind any tensor has. Returning 0 means a kind added to the
             * enum and not added above is left undecayed, which is the
             * conservative direction: an omission changes the regularization of
             * a new tensor rather than silently shrinking something that should
             * not shrink. tests/core/test_weight_decay.c pins the whole table,
             * so the omission is caught rather than merely survived. */
            return 0;
    }
}

model_errors_t model_apply_weight_decay(neural_model_t *model, float lr,
                                        float weight_decay) {
    if (!model || model->params_read_only) return MODEL_INVALID_INPUT;
    if (weight_decay <= 0.0f) return MODEL_SUCCESS;

    /* One allocation per optimizer step, of a few dozen descriptors. That is
     * nothing beside the forward and backward passes that produced the gradient
     * this step is applying, and it keeps the policy reading over the inventory
     * itself rather than over a cached derivative of it that could drift. */
    size_t count = model_param_tensor_count(model);
    param_tensor_t *tensors = malloc(count * sizeof(*tensors));
    if (!tensors) return MODEL_ALLOCATION_FAILURE;
    if (model_param_tensors(model, tensors, count) != count) {
        free(tensors);
        return MODEL_INVALID_INPUT;
    }

    for (size_t t = 0; t < count; t++) {
        if (!weight_decay_applies_to(tensors[t].kind)) continue;
        size_t elements = tensors[t].rows * tensors[t].cols;
        float *values = tensors[t].values;
        for (size_t i = 0; i < elements; i++) {
            values[i] -= lr * weight_decay * values[i];
        }
    }

    free(tensors);
    return MODEL_SUCCESS;
}
