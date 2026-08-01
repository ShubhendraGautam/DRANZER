#include "core/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int initialize_sgd(neural_model_t *model, unsigned int seed) {
    srand(seed);
    if (model_new(model, 20, 8, 2, 1, 8) != MODEL_SUCCESS) return -1;
    model->optimizer_type = OPTIMIZER_SGD;
    model->learning_rate = 0.002f;
    model->metrics.learning_rate = model->learning_rate;
    model->metrics.initial_learning_rate = model->learning_rate;
    model->grad_clip_norm = 0.0f;
    model->weight_decay = 0.0f;
    return 0;
}

int main(void) {
    neural_model_t batched = {0};
    neural_model_t reference = {0};
    neural_model_t invalid = {0};
    uint32_t first[] = {1, 2, 3};
    uint32_t second[] = {4, 5, 6};
    int failed = model_new(&invalid, SIZE_MAX, SIZE_MAX, 1, SIZE_MAX,
                           SIZE_MAX) != MODEL_INVALID_INPUT;

    if (initialize_sgd(&batched, 91) != 0 || initialize_sgd(&reference, 91) != 0) {
        fprintf(stderr, "minibatch fixture initialization failed\n");
        failed = 1;
        goto cleanup;
    }

    size_t bytes = batched.total_param_count * sizeof(float);
    float *first_gradient = malloc(bytes);
    if (!first_gradient) {
        failed = 1;
        goto cleanup;
    }

    float loss_first = 0.0f, loss_second = 0.0f;
    model_zero_gradients(&reference);
    if (model_accumulate_gradients(&reference, first, 7, 3,
                                   &loss_first) != MODEL_SUCCESS) failed = 1;
    memcpy(first_gradient, reference.grads, bytes);
    model_zero_gradients(&reference);
    if (model_accumulate_gradients(&reference, second, 8, 3,
                                   &loss_second) != MODEL_SUCCESS) failed = 1;
    for (size_t i = 0; i < reference.total_param_count; i++) {
        float mean_gradient = (first_gradient[i] + reference.grads[i]) * 0.5f;
        reference.grads[i] = mean_gradient;
        reference.params[i] -= reference.learning_rate * mean_gradient;
    }

    model_zero_gradients(&batched);
    float actual_first = 0.0f, actual_second = 0.0f;
    if (model_accumulate_gradients(&batched, first, 7, 3,
                                   &actual_first) != MODEL_SUCCESS ||
        model_accumulate_gradients(&batched, second, 8, 3,
                                   &actual_second) != MODEL_SUCCESS ||
        model_apply_accumulated_gradients(
            &batched, 2, (actual_first + actual_second) * 0.5f) != MODEL_SUCCESS) {
        failed = 1;
    }

    float max_param_diff = 0.0f, max_gradient_diff = 0.0f;
    for (size_t i = 0; !failed && i < batched.total_param_count; i++) {
        float param_diff = fabsf(batched.params[i] - reference.params[i]);
        float gradient_diff = fabsf(batched.grads[i] - reference.grads[i]);
        if (param_diff > max_param_diff) max_param_diff = param_diff;
        if (gradient_diff > max_gradient_diff) max_gradient_diff = gradient_diff;
    }
    if (!failed && (max_param_diff > 1e-6f || max_gradient_diff > 1e-6f ||
                    batched.training_steps != 1 || batched.metrics.history_size != 1 ||
                    fabsf(batched.current_loss - (loss_first + loss_second) * 0.5f) > 1e-6f)) {
        fprintf(stderr, "minibatch update was not the mean of its sample gradients\n");
        fprintf(stderr, "max param diff %.9g, max gradient diff %.9g\n",
                max_param_diff, max_gradient_diff);
        failed = 1;
    }
    free(first_gradient);

    if (!failed) {
        printf("samples=2 optimizer_steps=%u average_loss=%.6f max_diff=%.3g\n",
               batched.training_steps, batched.current_loss,
               (double)fmaxf(max_param_diff, max_gradient_diff));
        printf("\nMINIBATCH ACCUMULATION CHECK PASSED\n");
    }

cleanup:
    model_free(&batched);
    model_free(&reference);
    return failed;
}
