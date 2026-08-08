/* Which tensors weight decay touches, pinned.
 *
 * The decay used to run over the flat parameter vector, so it shrank every
 * LayerNorm gain and every bias along with the projections. That is a different
 * model from the AdamW every reference implementation uses, not a different
 * regularization strength: a LayerNorm gain starts at 1.0 and multiplies a
 * normalized activation, so pulling it towards zero shrinks the layer's output
 * scale every step with no gradient behind it.
 *
 * A test is needed rather than just the fixed code because the failure is silent.
 * Nothing about a model whose norms are being decayed looks wrong - it trains,
 * the loss descends, and the only visible consequence is that a comparison
 * against an external baseline is measuring two changes instead of one. So this
 * pins the policy both ways: the tensors that must move, and the tensors that
 * must not move at all.
 */
#include "core/model.h"
#include "core/model_params.h"
#include "core/weight_decay.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 20
#define EMB 8
#define HEADS 2
#define LAYERS 2
#define MAX_SEQ 8

int main(void) {
    int failures = 0;

    /* ---- 1. The policy table itself ---- */
    struct { param_kind_t kind; int decayed; const char *why; } expected[] = {
        { PARAM_KIND_PROJECTION, 1, "matmul operands are what \"weight\" decay means" },
        { PARAM_KIND_EMBEDDING,  1, "decayed with the output head it is compared against" },
        { PARAM_KIND_BIAS,       0, "an offset the data asks for, not a weight" },
        { PARAM_KIND_NORM,       0, "decaying a gain towards 0 changes the model" },
    };
    printf("decay policy by tensor kind:\n");
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        int actual = weight_decay_applies_to(expected[i].kind) ? 1 : 0;
        int ok = actual == expected[i].decayed;
        printf("  %-11s %-8s %s  (%s)\n", param_kind_name(expected[i].kind),
               actual ? "decayed" : "exempt", ok ? "" : "<-- FAIL",
               expected[i].why);
        if (!ok) failures++;
    }
    /* Every kind in the enum has to appear above, or a kind added later would go
     * unpinned - which is exactly how the flat-vector behaviour survived. */
    if (sizeof(expected) / sizeof(expected[0]) != (size_t)PARAM_KIND_COUNT) {
        printf("  FAIL: %zu kinds pinned but the enum has %d\n",
               sizeof(expected) / sizeof(expected[0]), (int)PARAM_KIND_COUNT);
        failures++;
    }

    /* ---- 2. What one optimizer step actually moves ---- */
    neural_model_t model = {0};
    if (model_new_seeded(&model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ, 5) !=
        MODEL_SUCCESS) {
        fprintf(stderr, "model_new_seeded failed\n");
        return 1;
    }

    /* A large decay and a zero gradient: with no gradient, the decay is the only
     * thing that can move a parameter, so any movement is attributable and any
     * absence of movement is conclusive. A realistic 0.01 with a real gradient
     * would leave the two effects tangled. */
    model.optimizer_type = OPTIMIZER_ADAM;
    model.learning_rate = 0.5f;
    model.weight_decay = 0.5f;
    model.grad_clip_norm = 0.0f;

    size_t tensor_count = model_param_tensor_count(&model);
    param_tensor_t *tensors = malloc(tensor_count * sizeof(*tensors));
    float *before = malloc(model.total_param_count * sizeof(float));
    if (!tensors || !before ||
        model_param_tensors(&model, tensors, tensor_count) != tensor_count) {
        fprintf(stderr, "could not snapshot the model\n");
        free(tensors);
        free(before);
        model_free(&model);
        return 1;
    }
    memcpy(before, model.params, model.total_param_count * sizeof(float));

    model_zero_gradients(&model);
    if (model_optimizer_step(&model) != MODEL_SUCCESS) {
        fprintf(stderr, "optimizer step failed\n");
        free(tensors);
        free(before);
        model_free(&model);
        return 1;
    }

    printf("\none step with zero gradients and weight_decay=0.5, lr=0.5:\n");
    printf("%-24s %-11s %10s %10s  %s\n", "tensor", "kind", "moved", "expected",
           "");
    for (size_t t = 0; t < tensor_count; t++) {
        param_tensor_t *tensor = &tensors[t];
        size_t elements = tensor->rows * tensor->cols;
        size_t offset = (size_t)(tensor->values - model.params);
        size_t moved = 0;
        for (size_t i = 0; i < elements; i++) {
            if (tensor->values[i] != before[offset + i]) moved++;
        }

        int should_decay = weight_decay_applies_to(tensor->kind);
        /* An exempt tensor must not move by a single bit. A decayed tensor must
         * move everywhere it is nonzero - a zero parameter times any decay is
         * still zero, which is why biases initialized to 0 could not be checked
         * this way and are covered by the exempt half instead. */
        size_t nonzero = 0;
        for (size_t i = 0; i < elements; i++) {
            if (before[offset + i] != 0.0f) nonzero++;
        }
        size_t want = should_decay ? nonzero : 0;
        int ok = moved == want;
        if (!ok) failures++;

        printf("%-24s %-11s %5zu/%-4zu %5zu/%-4zu  %s\n", tensor->name,
               param_kind_name(tensor->kind), moved, elements, want, elements,
               ok ? (should_decay ? "decayed" : "untouched") : "<-- FAIL");
    }

    /* ---- 3. The magnitude, for one decayed tensor ---- */
    /* p -= lr * wd * p leaves p * (1 - 0.25). Checking the value and not just
     * that it moved: a decay applied twice, or applied through the adaptive
     * step, would still move every element. */
    for (size_t t = 0; t < tensor_count; t++) {
        if (tensors[t].kind != PARAM_KIND_PROJECTION) continue;
        size_t offset = (size_t)(tensors[t].values - model.params);
        float original = before[offset];
        float expected_value = original - 0.5f * 0.5f * original;
        float actual = tensors[t].values[0];
        float gap = actual - expected_value;
        if (gap < 0.0f) gap = -gap;
        int ok = gap <= 1e-6f;
        printf("\n%s[0]: %.7f -> %.7f, decoupled decay predicts %.7f  %s\n",
               tensors[t].name, (double)original, (double)actual,
               (double)expected_value, ok ? "ok" : "<-- FAIL");
        if (!ok) failures++;
        break;
    }

    free(tensors);
    free(before);
    model_free(&model);

    printf("\n%s\n", failures == 0 ? "WEIGHT DECAY POLICY CHECK PASSED"
                                   : "WEIGHT DECAY POLICY CHECK FAILED");
    return failures == 0 ? 0 : 1;
}
