/* Verifies that incremental decoding with a per-layer KV cache produces
 * the same next-token logits as recomputing the complete prefix. */

#include "core/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_SIZE 37
#define EMBEDDING_DIM 12
#define NUM_HEADS 3
#define NUM_LAYERS 2
#define MAX_SEQ_LEN 8

static void normalize_cache(model_kv_cache_t *destination,
                            const model_kv_cache_t *source) {
    size_t row_bytes = source->embedding_dim * sizeof(float);
    destination->length = source->length;
    destination->start = 0;
    destination->total_tokens = source->total_tokens;
    for (size_t layer = 0; layer < source->num_layers; layer++) {
        for (size_t logical = 0; logical < source->length; logical++) {
            size_t physical = (source->start + logical) % source->capacity;
            memcpy(&destination->keys[layer][logical * source->embedding_dim],
                   &source->keys[layer][physical * source->embedding_dim],
                   row_bytes);
            memcpy(&destination->values[layer][logical * source->embedding_dim],
                   &source->values[layer][physical * source->embedding_dim],
                   row_bytes);
        }
    }
}

int main(void) {
    srand(17);

    neural_model_t model = {0};
    if (model_new(&model, VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS,
                  NUM_LAYERS, MAX_SEQ_LEN) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        return 1;
    }
    model.is_training = 0;

    float position_row[EMBEDDING_DIM];
    if (compute_positional_encoding_at(position_row, 5, EMBEDDING_DIM) != 0) {
        fprintf(stderr, "absolute positional encoding failed\n");
        model_free(&model);
        return 1;
    }
    for (size_t d = 0; d < EMBEDDING_DIM; d++) {
        if (fabsf(position_row[d] -
                  model.position_embeddings[5 * EMBEDDING_DIM + d]) > 1e-7f) {
            fprintf(stderr, "absolute positional formula changed at dim %zu\n", d);
            model_free(&model);
            return 1;
        }
    }

    model_kv_cache_t rejected = {0};
    if (model_kv_cache_init_with_capacity(&rejected, &model,
                                          MAX_SEQ_LEN + 1) != MODEL_INVALID_INPUT) {
        fprintf(stderr, "oversized KV window was accepted\n");
        model_kv_cache_free(&rejected);
        model_free(&model);
        return 1;
    }

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

    if (cache.length != MAX_SEQ_LEN || cache.total_tokens != MAX_SEQ_LEN ||
        cache.start != 0) {
        fprintf(stderr, "KV cache initial fill metadata is invalid\n");
        model_kv_cache_free(&cache);
        model_free(&model);
        return 1;
    }

    model_kv_cache_t normalized = {0};
    if (model_kv_cache_init(&normalized, &model) != MODEL_SUCCESS) {
        fprintf(stderr, "normalized KV cache allocation failed\n");
        model_kv_cache_free(&cache);
        model_free(&model);
        return 1;
    }
    const size_t extra_steps = MAX_SEQ_LEN * 2 + 3;
    for (size_t step = 0; step < extra_steps; step++) {
        uint32_t token = (uint32_t)((step * 7 + 23) % VOCAB_SIZE);
        normalize_cache(&normalized, &cache);
        if (model_forward_token(&model, &cache, token, cached_logits) != MODEL_SUCCESS ||
            model_forward_token(&model, &normalized, token, full_logits) != MODEL_SUCCESS) {
            fprintf(stderr, "sliding KV decode failed at extra step %zu\n", step);
            model_kv_cache_free(&normalized);
            model_kv_cache_free(&cache);
            model_free(&model);
            return 1;
        }
        for (size_t i = 0; i < VOCAB_SIZE; i++) {
            float diff = fabsf(full_logits[i] - cached_logits[i]);
            if (diff > 2e-5f) {
                fprintf(stderr,
                        "ring-layout mismatch at extra step=%zu vocab=%zu diff=%.8g\n",
                        step, i, diff);
                model_kv_cache_free(&normalized);
                model_kv_cache_free(&cache);
                model_free(&model);
                return 1;
            }
        }
    }
    if (cache.length != MAX_SEQ_LEN ||
        cache.total_tokens != MAX_SEQ_LEN + extra_steps ||
        cache.start != extra_steps % MAX_SEQ_LEN) {
        fprintf(stderr, "KV ring did not retain the newest fixed window\n");
        model_kv_cache_free(&normalized);
        model_kv_cache_free(&cache);
        model_free(&model);
        return 1;
    }
    model_kv_cache_free(&normalized);

    model_kv_cache_reset(&cache);
    if (cache.length != 0 || cache.start != 0 || cache.total_tokens != 0 ||
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
