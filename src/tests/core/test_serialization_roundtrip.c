/* Save -> load roundtrip check: a model trained a few steps, saved to
 * disk, and loaded back into a fresh neural_model_t must produce identical
 * logits for the same input as the original - guards against the flat
 * params-buffer layout (model.c/model_new) and its serialization
 * (serialization.c) ever drifting out of sync. */
#include "../include/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB 15
#define EMB 8
#define HEADS 2
#define LAYERS 2
#define MAX_SEQ 6
#define SEQ_LEN 4

static const char *TMP_PATH = "/tmp/test_serialization_roundtrip.pth";

int main(void) {
    srand(3);
    neural_model_t model = {0};
    if (model_new(&model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        return 1;
    }
    model.learning_rate = 0.01f;

    uint32_t tokens[SEQ_LEN] = {1, 4, 2, 7};
    uint32_t target = 9;
    for (int i = 0; i < 25; i++) {
        model_train_step(&model, tokens, target, SEQ_LEN);
    }

    float logits_before[VOCAB];
    model.is_training = 0;
    model_forward(&model, tokens, SEQ_LEN, logits_before);

    if (model_save(&model, TMP_PATH) != MODEL_SUCCESS) {
        fprintf(stderr, "model_save failed\n");
        return 1;
    }

    neural_model_t loaded = {0};
    /* model_load/model_read_state reinitializes *loaded via model_new
     * internally once it reads the header, so an empty struct is fine. */
    if (model_load(&loaded, TMP_PATH) != MODEL_SUCCESS) {
        fprintf(stderr, "model_load failed\n");
        return 1;
    }
    loaded.is_training = 0;

    float logits_after[VOCAB];
    model_forward(&loaded, tokens, SEQ_LEN, logits_after);

    int pass = 1;
    float max_diff = 0.0f;
    for (size_t i = 0; i < VOCAB; i++) {
        float diff = fabsf(logits_before[i] - logits_after[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) pass = 0;
    }
    printf("max logit diff after save/load roundtrip: %.8f\n", max_diff);

    if (loaded.total_param_count != model.total_param_count) {
        printf("FAIL: total_param_count mismatch after reload (%zu vs %zu)\n",
               loaded.total_param_count, model.total_param_count);
        pass = 0;
    }

    model_free(&model);
    model_free(&loaded);
    remove(TMP_PATH);

    printf("\n%s\n", pass ? "SERIALIZATION ROUNDTRIP CHECK PASSED" : "SERIALIZATION ROUNDTRIP CHECK FAILED");
    return pass ? 0 : 1;
}
