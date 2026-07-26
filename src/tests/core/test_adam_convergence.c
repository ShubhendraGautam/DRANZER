/* Sanity check that AdamW (model_new's default optimizer) actually drives
 * loss down on a single repeated example, and much faster than plain SGD
 * would - not a numerical-correctness proof (that's test_gradient_check.c's
 * job), just a regression guard against a mis-wired optimizer silently
 * training a model that never converges. */
#include "core/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB 20
#define EMB 16
#define HEADS 2
#define LAYERS 2
#define MAX_SEQ 8
#define STEPS 200

int main(void) {
    srand(7);
    neural_model_t model = {0};
    if (model_new(&model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        return 1;
    }

    int pass = 1;
    if (model.optimizer_type != OPTIMIZER_ADAM) {
        printf("FAIL: model_new's default optimizer_type is not OPTIMIZER_ADAM\n");
        pass = 0;
    }
    if (model.adam_m != NULL || model.adam_v != NULL) {
        printf("FAIL: adam_m/adam_v should be NULL until the first Adam step (lazy allocation)\n");
        pass = 0;
    }

    model.learning_rate = 0.01f;

    uint32_t tokens[4] = {1, 5, 3, 9};
    uint32_t target = 12;

    float first_loss = 0.0f, last_loss = 0.0f;
    for (int step = 0; step < STEPS; step++) {
        model_train_step(&model, tokens, target, 4);
        if (step == 0) first_loss = model.current_loss;
        if (step == STEPS - 1) last_loss = model.current_loss;
    }
    printf("loss: step0=%.6f -> step%d=%.6f\n", first_loss, STEPS - 1, last_loss);

    if (model.adam_m == NULL || model.adam_v == NULL) {
        printf("FAIL: adam_m/adam_v should be allocated after training with OPTIMIZER_ADAM\n");
        pass = 0;
    }
    /* Overfitting one example this many steps should crush the loss close
     * to zero; a generous threshold keeps this from being flaky while
     * still catching a genuinely broken optimizer (e.g. one that doesn't
     * move the loss at all, or diverges). */
    if (!(last_loss < 0.01f) || !(last_loss < first_loss)) {
        printf("FAIL: loss did not converge as expected (first=%.6f last=%.6f)\n", first_loss, last_loss);
        pass = 0;
    }

    model_free(&model);
    printf("\n%s\n", pass ? "ADAM CONVERGENCE CHECK PASSED" : "ADAM CONVERGENCE CHECK FAILED");
    return pass ? 0 : 1;
}
