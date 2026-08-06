/*
 * Applies a quantization policy to every trainable tensor a model has, and
 * records what each one cost. See core/model_quantize.h.
 */

#include "core/model_quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void model_quantize_default_config(model_quant_config_t *config) {
    if (!config) return;
    config->bits = 0;
    config->granularity = QUANT_GRANULARITY_COLUMN;
    config->include_embeddings = 0;
    config->include_biases_and_norms = 0;
}

int model_quantize_includes(const model_quant_config_t *config, param_kind_t kind) {
    if (!config || config->bits == 0) return 0;
    switch (kind) {
        case PARAM_KIND_PROJECTION: return 1;
        case PARAM_KIND_EMBEDDING:  return config->include_embeddings != 0;
        case PARAM_KIND_BIAS:
        case PARAM_KIND_NORM:       return config->include_biases_and_norms != 0;
        default:                    return 0;
    }
}

/* Scales a tensor needs under a granularity: one, one per row, or one per
 * column. Only used for the storage estimate. */
static size_t scale_count(const param_tensor_t *tensor,
                          quant_granularity_t granularity) {
    switch (granularity) {
        case QUANT_GRANULARITY_TENSOR: return 1;
        case QUANT_GRANULARITY_ROW:    return tensor->rows;
        case QUANT_GRANULARITY_COLUMN: return tensor->cols;
        default:                       return 1;
    }
}

static int config_valid(const model_quant_config_t *config) {
    if (!config) return 0;
    if (config->bits != 0 &&
        (config->bits < QUANT_MIN_BITS || config->bits > QUANT_MAX_BITS)) {
        return 0;
    }
    if ((int)config->granularity < 0 ||
        (int)config->granularity >= (int)QUANT_GRANULARITY_COUNT) {
        return 0;
    }
    return 1;
}

int model_quantize_weights(neural_model_t *model,
                           const model_quant_config_t *config,
                           model_quant_report_t *report) {
    if (!model || !config_valid(config)) return -1;

    const size_t total = model_param_tensor_count(model);
    if (total == 0) return -1;

    /* The whole inventory at once. This runs once per model, not per token, so
     * one allocation is the right trade against carrying descriptor storage on
     * every model that never quantizes. Allocated before anything is modified,
     * so a failure leaves the weights untouched as the header promises. */
    param_tensor_t *tensors = malloc(total * sizeof(*tensors));
    if (!tensors) return -1;
    if (model_param_tensors(model, tensors, total) != total) {
        free(tensors);
        return -1;
    }

    model_quant_tensor_t *slots = NULL;
    size_t slot_capacity = 0;
    if (report) {
        slots = report->tensors;
        slot_capacity = report->tensor_capacity;
        memset(report, 0, sizeof(*report));
        report->tensors = slots;
        report->tensor_capacity = slot_capacity;
        report->config = *config;
        report->tensors_total = total;
    }

    /* Bits the policy would store, accumulated as tensors are visited so the
     * estimate cannot drift from what was actually quantized. */
    double stored_bits = 0.0;
    size_t stored_values = 0;

    for (size_t index = 0; index < total; index++) {
        const param_tensor_t *entry = &tensors[index];
        const int included = model_quantize_includes(config, entry->kind);
        const size_t count = entry->rows * entry->cols;

        quant_error_t error;
        memset(&error, 0, sizeof(error));

        if (included) {
            if (quantize_dequantize(entry->values, entry->rows, entry->cols,
                                    config->bits, config->granularity,
                                    &error) != 0) {
                /* A tensor the policy selected but the quantizer rejected means
                 * the inventory and the quantizer disagree about shapes, which
                 * is a bug rather than a runtime condition. Earlier tensors have
                 * already been modified, so the model is reported as unusable
                 * rather than silently half-quantized. */
                free(tensors);
                return -1;
            }
            stored_bits += (double)count * (double)config->bits +
                           (double)scale_count(entry, config->granularity) * 32.0;
        } else {
            stored_bits += (double)count * 32.0;
        }
        stored_values += count;

        if (report) {
            report->values_total += count;
            if (included) {
                report->tensors_quantized++;
                report->values_quantized += count;
                quant_error_accumulate(&report->combined, &error);
            }
            if (slots && index < slot_capacity) {
                model_quant_tensor_t *slot = &slots[index];
                snprintf(slot->name, sizeof(slot->name), "%s", entry->name);
                slot->kind = entry->kind;
                slot->rows = entry->rows;
                slot->cols = entry->cols;
                slot->quantized = included;
                slot->error = error;
            }
        }
    }

    if (report && stored_values > 0) {
        report->effective_bits_per_value = stored_bits / (double)stored_values;
    }

    free(tensors);
    return 0;
}
