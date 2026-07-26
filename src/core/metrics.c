/*
 * Learning metrics accessors - loss history, best/worst/avg loss, current
 * learning rate. The metrics themselves are updated in training.c as part
 * of model_train_step; this module just reports them.
 */

#include "core/metrics.h"
#include <stdio.h>

void model_get_metrics(neural_model_t *model, learning_metrics_t *out_metrics) {
    if (!model || !out_metrics) return;
    *out_metrics = model->metrics;
}

void model_print_metrics(neural_model_t *model) {
    if (!model) return;

    printf("\n=== Learning Metrics ===\n");
    printf("Training steps: %u\n", model->training_steps);
    printf("Current loss: %.6f\n", model->current_loss);
    printf("Best loss: %.6f\n", model->metrics.best_loss);
    printf("Worst loss: %.6f\n", model->metrics.worst_loss);
    printf("Average loss: %.6f\n", model->metrics.avg_loss);
    printf("Current learning rate: %.8f\n", model->metrics.learning_rate);
    printf("Steps without improvement: %u\n", model->metrics.steps_without_improvement);

    if (model->metrics.history_size > 0) {
        printf("\nLast %zu loss values: ",
               model->metrics.history_size < 10 ? model->metrics.history_size : 10);
        size_t start = model->metrics.history_size < 10 ? 0 : model->metrics.history_size - 10;
        for (size_t i = start; i < model->metrics.history_size; i++) {
            printf("%.4f ", model->metrics.loss_history[i]);
        }
        printf("\n");
    }

    printf("========================\n\n");
}
