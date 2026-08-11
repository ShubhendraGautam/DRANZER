#include "byte_pair_encoding.h"
#include "core/bundle.h"
#include "core/model.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VOCAB 264
#define EMBEDDING 8
#define HEADS 2
#define LAYERS 1
#define MAX_SEQUENCE 6

static int near(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <= tolerance;
}

static int check_rotation_contract(void) {
    float values[2 * EMBEDDING];
    float original[2 * EMBEDDING];
    for (size_t i = 0; i < 2 * EMBEDDING; i++)
        values[i] = (float)((int)i - 6) / 7.0f;
    memcpy(original, values, sizeof(values));

    if (model_rope_forward(values, 1, EMBEDDING, HEADS, 0) != MODEL_SUCCESS ||
        memcmp(values, original, EMBEDDING * sizeof(float)) != 0 ||
        model_rope_forward(values, 2, EMBEDDING, HEADS, 5) != MODEL_SUCCESS) {
        return 0;
    }
    for (size_t row = 0; row < 2; row++) {
        for (size_t head = 0; head < HEADS; head++) {
            for (size_t pair = 0; pair < EMBEDDING / HEADS / 2; pair++) {
                size_t index = row * EMBEDDING +
                               head * (EMBEDDING / HEADS) + 2 * pair;
                float before = original[index] * original[index] +
                               original[index + 1] * original[index + 1];
                float after = values[index] * values[index] +
                              values[index + 1] * values[index + 1];
                if (!near(after, before, 2e-5f)) return 0;
            }
        }
    }
    if (model_rope_backward(values, 2, EMBEDDING, HEADS, 5) != MODEL_SUCCESS)
        return 0;
    for (size_t i = 0; i < 2 * EMBEDDING; i++)
        if (!near(values[i], original[i], 2e-5f)) return 0;
    return model_rope_forward(values, 1, 6, 2, 0) == MODEL_INVALID_INPUT;
}

static int check_cached_equivalence(neural_model_t *model) {
    uint32_t tokens[4] = {1, 7, 3, 11};
    float full[VOCAB], cached[VOCAB];
    model_kv_cache_t cache = {0};
    if (model_kv_cache_init(&cache, model) != MODEL_SUCCESS) return 0;
    int ok = 1;
    model->is_training = 0;
    for (size_t length = 1; ok && length <= 4; length++) {
        if (model_forward(model, tokens, length, full) != MODEL_SUCCESS ||
            model_forward_token(model, &cache, tokens[length - 1], cached) !=
                MODEL_SUCCESS) {
            ok = 0;
            break;
        }
        for (size_t token = 0; token < VOCAB; token++)
            if (!near(full[token], cached[token], 2e-4f)) ok = 0;
    }
    model_kv_cache_free(&cache);
    return ok;
}

static int check_gradient(neural_model_t *model) {
    uint32_t tokens[3] = {2, 5, 9};
    float *parameter = &model->layers[0].W_q[1];
    float original = *parameter;
    const float epsilon = 1e-3f;
    float center = 0.0f, plus = 0.0f, minus = 0.0f;

    model->dropout_rate = 0.0f;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 13, 3, &center) != MODEL_SUCCESS)
        return 0;
    float analytic = model->layers[0].W_q_grad[1];

    *parameter = original + epsilon;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 13, 3, &plus) != MODEL_SUCCESS)
        return 0;
    *parameter = original - epsilon;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 13, 3, &minus) != MODEL_SUCCESS)
        return 0;
    *parameter = original;
    model_zero_gradients(model);

    float numerical = (plus - minus) / (2.0f * epsilon);
    float tolerance = 3e-2f + 3e-2f * fabsf(numerical);
    return isfinite(center) && isfinite(analytic) && isfinite(numerical) &&
           near(analytic, numerical, tolerance);
}

static int check_bundle_roundtrip(neural_model_t *model) {
    char path[160];
    snprintf(path, sizeof(path), "/tmp/dranzer-rope-%ld.bin", (long)getpid());
    bpe_encoder_t encoder = {0};
    bpe_encoder_t *loaded_encoder = NULL, *mapped_encoder = NULL;
    neural_model_t loaded = {0}, mapped = {0};
    model_bundle_metadata_t metadata = {.train_window = 4, .seed = 31};
    model_bundle_metadata_t loaded_metadata = {0}, mapped_metadata = {0};
    int ok = bpe_encoder_new_with_special_tokens(&encoder, VOCAB) == BPE_SUCCESS &&
             bpe_train(&encoder, "rotary position fixture", 23) == BPE_SUCCESS &&
             bpe_encoder_freeze(&encoder) == BPE_SUCCESS &&
             model_bundle_save(model, &encoder, &metadata, path) == BUNDLE_SUCCESS &&
             model_bundle_load(&loaded, &loaded_encoder, &loaded_metadata,
                               path) == BUNDLE_SUCCESS &&
             model_bundle_load_mmap(&mapped, &mapped_encoder, &mapped_metadata,
                                    path) == BUNDLE_SUCCESS;
    if (ok) {
        ok = loaded_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             mapped_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             loaded.architecture_flags == MODEL_ARCH_ROPE &&
             mapped.architecture_flags == MODEL_ARCH_ROPE &&
             loaded.output_projection && mapped.output_projection &&
             loaded.total_param_count == model->total_param_count &&
             memcmp(loaded.params, model->params,
                    model->total_param_count * sizeof(float)) == 0 &&
             memcmp(mapped.params, model->params,
                    model->total_param_count * sizeof(float)) == 0;
    }
    model_free(&loaded);
    model_free(&mapped);
    if (loaded_encoder) { bpe_encoder_free(loaded_encoder); free(loaded_encoder); }
    if (mapped_encoder) { bpe_encoder_free(mapped_encoder); free(mapped_encoder); }
    bpe_encoder_free(&encoder);
    remove(path);
    return ok;
}

int main(void) {
    neural_model_t model = {0}, rejected = {0}, ordinary = {0};
    int ok = check_rotation_contract() &&
             model_new_seeded_architecture(
                 &model, VOCAB, EMBEDDING, HEADS, LAYERS, MAX_SEQUENCE, 31,
                 MODEL_ARCH_ROPE) == MODEL_SUCCESS &&
             model_uses_rope(&model) && check_cached_equivalence(&model);

    uint32_t tokens[3] = {2, 5, 9};
    if (ok) {
        ok = model_forward_hidden(&model, tokens, 3) == MODEL_SUCCESS;
        for (size_t row = 0; ok && row < 3; row++) {
            for (size_t dim = 0; dim < EMBEDDING; dim++) {
                if (!near(model.cache_hidden[0][row * EMBEDDING + dim],
                          model.token_embeddings[tokens[row] * EMBEDDING + dim],
                          0.0f)) ok = 0;
            }
        }
    }
    if (ok) ok = check_gradient(&model) && check_bundle_roundtrip(&model);

    if (model_new_seeded(&ordinary, VOCAB, 6, 2, LAYERS,
                         MAX_SEQUENCE, 1) != MODEL_SUCCESS ||
        model_new_seeded_architecture(
            &rejected, VOCAB, 6, 2, LAYERS, MAX_SEQUENCE, 1,
            MODEL_ARCH_ROPE) != MODEL_INVALID_INPUT) ok = 0;

    model_free(&ordinary);
    model_free(&rejected);
    model_free(&model);
    if (!ok) fprintf(stderr, "RoPE contract failed\n");
    else printf("\nROPE CHECK PASSED\n");
    return ok ? 0 : 1;
}
