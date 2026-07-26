#ifndef METRICS_H
#define METRICS_H

#include "core/model_types.h"

/**
 * Get learning metrics (loss history, stats)
 */
void model_get_metrics(neural_model_t *model, learning_metrics_t *out_metrics);

/**
 * Print training metrics and statistics
 */
void model_print_metrics(neural_model_t *model);

#endif // METRICS_H
