/*
 * Learning metrics accessors - loss history, best/worst/avg loss, current
 * learning rate. The metrics themselves are updated in training.c as part
 * of model_train_step; this module just reports them.
 */

#include "core/metrics.h"
void model_get_metrics(neural_model_t *model, learning_metrics_t *out_metrics) {
    if (!model || !out_metrics) return;
    *out_metrics = model->metrics;
}
