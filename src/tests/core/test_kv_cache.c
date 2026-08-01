/* Verifies that incremental decoding with a per-layer KV cache produces
 * the same next-token logits as recomputing the complete prefix. */

#include "core/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define VOCAB_SIZE 37
#define EMBEDDING_DIM 12
#define NUM_HEADS 3
#define NUM_LAYERS 2
#define MAX_SEQ_LEN 8

int main(void) {
    srand(17);

    neural_model_t model = {0};
    if (model_new(&model, VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS,
                  NUM_LAYERS, MAX_SEQ_LEN) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        return 1;
    }
    model.is_training = 0;

    model_kv_cache_t cache = {0};
    if (model_kv_cache_init(&cache, &model) != MODEL_SUCCESS) {
        fprintf(stderr, "model_kv_cache_init failed\n");
        model_free(&model);
        return 1;
    }

    uint32_t tokens[MAX_SEQ_LEN] = {3, 11, 7, 19, 2, 31, 5, 13};
    float full_logits[VOCAB_SIZE];
    float cached_logits[VOCAB_SIZE];
    float worst_diff = 0.0f;

    for (size_t position = 0; position < MAX_SEQ_LEN; position++) {
        if (model_forward(&model, tokens, position + 1, full_logits) != MODEL_SUCCESS ||
            model_forward_token(&model, &cache, tokens[position], cached_logits) != MODEL_SUCCESS) {
            fprintf(stderr, "forward failed at position %zu\n", position);
            model_kv_cache_free(&cache);
            model_free(&model);
            return 1;
        }

        for (size_t i = 0; i < VOCAB_SIZE; i++) {
            float diff = fabsf(full_logits[i] - cached_logits[i]);
            if (diff > worst_diff) worst_diff = diff;
            if (diff > 2e-5f) {
                fprintf(stderr,
                        "logit mismatch at position=%zu vocab=%zu: full=%+.8f cached=%+.8f diff=%.8g\n",
                        position, i, full_logits[i], cached_logits[i], diff);
                model_kv_cache_free(&cache);
                model_free(&model);
                return 1;
            }
        }
    }

    if (cache.length != MAX_SEQ_LEN ||
        model_forward_token(&model, &cache, 1, cached_logits) != MODEL_INVALID_INPUT) {
        fprintf(stderr, "KV cache capacity bound was not enforced\n");
        model_kv_cache_free(&cache);
        model_free(&model);
        return 1;
    }

    model_kv_cache_reset(&cache);
    if (cache.length != 0 ||
        model_forward_token(&model, &cache, tokens[0], cached_logits) != MODEL_SUCCESS ||
        model_forward(&model, tokens, 1, full_logits) != MODEL_SUCCESS) {
        fprintf(stderr, "KV cache reset failed\n");
        model_kv_cache_free(&cache);
        model_free(&model);
        return 1;
    }

    for (size_t i = 0; i < VOCAB_SIZE; i++) {
        if (fabsf(full_logits[i] - cached_logits[i]) > 2e-5f) {
            fprintf(stderr, "KV cache reset retained stale state\n");
            model_kv_cache_free(&cache);
            model_free(&model);
            return 1;
        }
    }

    printf("max full-vs-cached logit diff: %.8g\n", worst_diff);
    printf("\nKV CACHE CHECK PASSED\n");
    model_kv_cache_free(&cache);
    model_free(&model);
    return 0;
}
