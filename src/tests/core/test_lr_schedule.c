/* Checks the shape of model_lr_schedule_step's warmup+cosine schedule
 * (optimizer.c): linear ramp to base_lr over warmup_steps, monotonic
 * decay down to min_lr by total_steps, then flat at min_lr after. */
#include "core/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB 20
#define EMB 8
#define HEADS 2
#define LAYERS 1
#define MAX_SEQ 4

int main(void) {
    neural_model_t model = {0};
    if (model_new_seeded(&model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ, 1) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        return 1;
    }

    model.base_lr = 0.01f;
    model.min_lr = 0.0001f;
    model.warmup_steps = 10;
    model.total_steps = 50;
    model.learning_rate = 0.0f;

    uint32_t tokens[3] = {1, 2, 3};
    int pass = 1;
    float prev_lr = -1.0f;
    float lr_at_warmup_end = -1.0f, lr_at_horizon = -1.0f;

    for (int step = 0; step < 55; step++) {
        model_train_step(&model, tokens, 5, 3);
        float lr = model.learning_rate;

        if (step < (int)model.warmup_steps - 1) {
            /* Warmup: strictly increasing. */
            if (lr <= prev_lr) {
                printf("FAIL: lr did not increase during warmup at step %d (%.6f -> %.6f)\n", step, prev_lr, lr);
                pass = 0;
            }
        }
        if (step == (int)model.warmup_steps - 1) lr_at_warmup_end = lr;
        if (step == (int)model.total_steps - 2) lr_at_horizon = lr;
        prev_lr = lr;
    }

    printf("lr_at_warmup_end=%.6f (expect ~= base_lr=%.6f)\n", lr_at_warmup_end, model.base_lr);
    printf("lr_at_horizon=%.6f (expect close to min_lr=%.6f)\n", lr_at_horizon, model.min_lr);
    printf("lr_after_horizon=%.6f (expect == min_lr=%.6f)\n", model.learning_rate, model.min_lr);

    if (fabsf(lr_at_warmup_end - model.base_lr) > 0.001f) pass = 0;
    if (lr_at_horizon > lr_at_warmup_end) pass = 0; /* must have decayed */
    if (fabsf(model.learning_rate - model.min_lr) > 1e-6f) pass = 0;

    model_free(&model);
    printf("\n%s\n", pass ? "LR SCHEDULE CHECK PASSED" : "LR SCHEDULE CHECK FAILED");
    return pass ? 0 : 1;
}
