#ifndef MODEL_QUANTIZE_H
#define MODEL_QUANTIZE_H

#include "core/model_params.h"
#include "core/model_types.h"
#include "core/quantize.h"

/* Applying a quantization grid to a whole model, and reporting what it cost.
 *
 * Simulated only: weights are quantized and mapped straight back to float in
 * place, so the model runs through exactly the same kernels afterwards. What
 * changes is the *values*, and that is the whole point at this stage - the
 * accuracy question is answered before any storage or kernel work exists to
 * confound it. See docs/quantization.md.
 *
 * This is destructive and not invertible. A caller comparing against the
 * unquantized model needs two models, or a saved copy of `params`. */

typedef struct {
    /* Grid width. 0 leaves the model alone, which is what makes "no
     * quantization" a configuration rather than a special case at every call
     * site. */
    int bits;
    quant_granularity_t granularity;

    /* Which tensors the grid is applied to.
     *
     * Projections are always included - they are the weights worth quantizing
     * and the ones every scheme in the literature targets. The other two are
     * off by default, and both defaults are conventions this project has
     * inherited rather than measured, which is precisely why they are flags:
     *
     *  - Embeddings are large enough to be worth compressing and are read by
     *    lookup rather than multiplied, so the argument for a particular scale
     *    axis does not apply to them.
     *  - Biases and norm parameters together are a fraction of a percent of the
     *    parameters, so quantizing them saves nothing measurable while sitting
     *    directly on a normalized activation. The expected result is "no
     *    benefit, some risk", and the flag exists so that can be shown rather
     *    than asserted. */
    int include_embeddings;
    int include_biases_and_norms;
} model_quant_config_t;

typedef struct {
    char name[48];
    param_kind_t kind;
    size_t rows, cols;
    int quantized;         /* 0 when the policy excluded it */
    quant_error_t error;   /* zeroed when not quantized */
} model_quant_tensor_t;

typedef struct {
    model_quant_config_t config;

    size_t tensors_total;
    size_t tensors_quantized;
    size_t values_total;
    size_t values_quantized;

    /* Error over the quantized tensors only, count-weighted. Including the
     * untouched tensors would dilute it toward zero by an amount that depends
     * on the policy, making two policies incomparable. */
    quant_error_t combined;

    /* Bits per parameter the policy would actually cost on disk, counting the
     * float32 scales and the tensors left at full width.
     *
     * Reported because the stated width is not the storage cost, and comparing
     * schemes at equal *stated* width is how a finer granularity gets credit
     * for accuracy it bought with space. Per-column 4-bit on a 256x1024 tensor
     * carries 1024 float scales - a fifth of a bit per weight - while the same
     * scheme on a 1x1024 bias carries one scale per value and costs more than
     * leaving it alone. */
    double effective_bits_per_value;

    /* Filled up to `tensor_capacity`; `tensors_total` says how many exist. */
    model_quant_tensor_t *tensors;
    size_t tensor_capacity;
} model_quant_report_t;

/* Defaults: no quantization, per-column scales, projections only. */
void model_quantize_default_config(model_quant_config_t *config);

/* Whether this policy would quantize a tensor of this kind. Exposed so a
 * report consumer and the quantizer cannot disagree about it. */
int model_quantize_includes(const model_quant_config_t *config, param_kind_t kind);

/* Apply `config` to `model` in place.
 *
 * Returns 0 on success, or -1 for a null model, a null config, an invalid bit
 * width, or an unknown granularity - in every failing case the weights are
 * untouched. `config->bits == 0` succeeds and changes nothing, filling the
 * report with zero error, so a "no quantization" baseline flows through the
 * same path as every other configuration instead of skipping it.
 *
 * `report` may be NULL. When it is not, `report->tensors` and
 * `report->tensor_capacity` are read as caller-provided storage and everything
 * else is overwritten; pass a zero capacity for aggregates only. */
int model_quantize_weights(neural_model_t *model,
                           const model_quant_config_t *config,
                           model_quant_report_t *report);

#endif /* MODEL_QUANTIZE_H */
