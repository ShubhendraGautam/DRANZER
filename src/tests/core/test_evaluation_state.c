#include "core/model.h"
#include "common/fp_bits.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *copy_bytes(const void *source, size_t size) {
    void *copy = malloc(size);
    if (copy) memcpy(copy, source, size);
    return copy;
}

int main(void) {
    neural_model_t model = {0};
    uint32_t context[] = {1, 2, 3};
    if (model_new_seeded(&model, 12, 8, 2, 1, 8, 31) != MODEL_SUCCESS ||
        model_train_step(&model, context, 4, 3) != MODEL_SUCCESS) {
        fprintf(stderr, "failed to initialize evaluation fixture\n");
        model_free(&model);
        return 1;
    }

    size_t param_bytes = model.total_param_count * sizeof(float);
    size_t history_bytes = model.metrics.history_capacity * sizeof(float);
    float *params = copy_bytes(model.params, param_bytes);
    float *grads = copy_bytes(model.grads, param_bytes);
    float *adam_m = copy_bytes(model.adam_m, param_bytes);
    float *adam_v = copy_bytes(model.adam_v, param_bytes);
    float *history = copy_bytes(model.metrics.loss_history, history_bytes);
    if (!params || !grads || !adam_m || !adam_v || !history) {
        fprintf(stderr, "failed to snapshot model state\n");
        free(params); free(grads); free(adam_m); free(adam_v); free(history);
        model_free(&model);
        return 1;
    }

    uint32_t training_steps = model.training_steps;
    uint32_t adam_t = model.adam_t;
    float current_loss = model.current_loss;
    learning_metrics_t metrics = model.metrics;
    int was_training = model.is_training;
    double first_loss = 0.0;
    double second_loss = 0.0;

    int failed = model_evaluate_step(&model, context, 4, 3, &first_loss) != MODEL_SUCCESS ||
                 model_evaluate_step(&model, context, 4, 3, &second_loss) != MODEL_SUCCESS ||
                 /* Bit-pattern check, not isfinite(): -ffast-math folds
                  * isfinite() to true, so this line asserted nothing.
                  * See include/common/fp_bits.h. */
                 !dranzer_double_is_finite(first_loss) || first_loss <= 0.0 ||
                 first_loss != second_loss ||
                 model.training_steps != training_steps || model.adam_t != adam_t ||
                 model.current_loss != current_loss || model.is_training != was_training ||
                 memcmp(model.params, params, param_bytes) != 0 ||
                 memcmp(model.grads, grads, param_bytes) != 0 ||
                 memcmp(model.adam_m, adam_m, param_bytes) != 0 ||
                 memcmp(model.adam_v, adam_v, param_bytes) != 0 ||
                 memcmp(model.metrics.loss_history, history, history_bytes) != 0 ||
                 model.metrics.history_size != metrics.history_size ||
                 model.metrics.best_loss != metrics.best_loss ||
                 model.metrics.worst_loss != metrics.worst_loss ||
                 model.metrics.avg_loss != metrics.avg_loss ||
                 model.metrics.learning_rate != metrics.learning_rate ||
                 model.metrics.steps_without_improvement != metrics.steps_without_improvement;

    if (failed) {
        fprintf(stderr, "evaluation mutated persistent model or optimizer state\n");
    } else {
        printf("loss=%.9f persistent_state=unchanged\n", first_loss);
        printf("\nEVALUATION STATE CHECK PASSED\n");
    }

    free(params); free(grads); free(adam_m); free(adam_v); free(history);
    model_free(&model);
    return failed;
}
