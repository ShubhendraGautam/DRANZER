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
#define MAX_SEQUENCE 5

static int near(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <= tolerance;
}

static int check_scalar_math(void) {
    const float probes[] = {-3.0f, -1.0f, 0.0f, 0.5f, 2.0f};
    const float epsilon = 1e-3f;
    if (gelu(0.0f) != 0.0f || !near(gelu_derivative(0.0f), 0.5f, 1e-7f))
        return 0;
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        float numerical = (gelu(probes[i] + epsilon) -
                           gelu(probes[i] - epsilon)) / (2.0f * epsilon);
        if (!near(gelu_derivative(probes[i]), numerical, 8e-4f)) return 0;
    }
    return 1;
}

static int check_gradient(neural_model_t *model) {
    uint32_t tokens[3] = {1, 4, 7};
    float *parameter = &model->layers[0].W_ff1[0];
    float original = *parameter;
    const float epsilon = 1e-3f;
    float center = 0.0f, plus = 0.0f, minus = 0.0f;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 9, 3, &center) != MODEL_SUCCESS)
        return 0;
    float analytic = model->layers[0].W_ff1_grad[0];
    *parameter = original + epsilon;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 9, 3, &plus) != MODEL_SUCCESS)
        return 0;
    *parameter = original - epsilon;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 9, 3, &minus) != MODEL_SUCCESS)
        return 0;
    *parameter = original;
    model_zero_gradients(model);
    float numerical = (plus - minus) / (2.0f * epsilon);
    return isfinite(center) && isfinite(analytic) && isfinite(numerical) &&
           near(analytic, numerical, 3e-2f + 3e-2f * fabsf(numerical));
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

static int check_bundle_roundtrip(neural_model_t *model) {
    char path[160];
    snprintf(path, sizeof(path), "/tmp/dranzer-gelu-%ld.bin", (long)getpid());
    const char fixture[] = "GELU bundle fixture";
    bpe_encoder_t encoder = {0};
    bpe_encoder_t *loaded_encoder = NULL, *mapped_encoder = NULL;
    neural_model_t loaded = {0}, mapped = {0};
    model_bundle_metadata_t metadata = {.train_window = 4, .seed = 51};
    model_bundle_metadata_t loaded_metadata = {0}, mapped_metadata = {0};
    int ok = bpe_encoder_new_with_special_tokens(&encoder, VOCAB) == BPE_SUCCESS &&
             bpe_train(&encoder, fixture, sizeof(fixture) - 1) == BPE_SUCCESS &&
             bpe_encoder_freeze(&encoder) == BPE_SUCCESS &&
             model_bundle_save(model, &encoder, &metadata, path) == BUNDLE_SUCCESS &&
             model_bundle_load(&loaded, &loaded_encoder, &loaded_metadata,
                               path) == BUNDLE_SUCCESS &&
             model_bundle_load_mmap(&mapped, &mapped_encoder, &mapped_metadata,
                                    path) == BUNDLE_SUCCESS;
    if (ok) {
        ok = loaded_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             mapped_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             loaded.architecture_flags == MODEL_ARCH_GELU &&
             mapped.architecture_flags == MODEL_ARCH_GELU &&
             loaded.cache_ff_pre_activation[0] &&
             mapped.cache_ff_pre_activation[0] &&
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
    neural_model_t ordinary = {0}, model = {0};
    int ok = check_scalar_math() &&
             model_new_seeded(&ordinary, VOCAB, EMBEDDING, HEADS, LAYERS,
                              MAX_SEQUENCE, 51) == MODEL_SUCCESS &&
             model_new_seeded_architecture(
                 &model, VOCAB, EMBEDDING, HEADS, LAYERS, MAX_SEQUENCE, 51,
                 MODEL_ARCH_GELU) == MODEL_SUCCESS &&
             model_uses_gelu(&model) &&
             !ordinary.cache_ff_pre_activation[0] &&
             model.cache_ff_pre_activation[0] &&
             ordinary.total_param_count == model.total_param_count &&
             check_cached_equivalence(&model) &&
             check_gradient(&model) &&
             check_bundle_roundtrip(&model);
    model_free(&ordinary);
    model_free(&model);
    if (!ok) fprintf(stderr, "GELU contract failed\n");
    else printf("\nGELU CHECK PASSED\n");
    return ok ? 0 : 1;
}
