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
#define LAYERS 2
#define MAX_SEQUENCE 6

static int near(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <= tolerance;
}

static float rms_loss(const float *input, const float *gamma,
                      const float *upstream) {
    float output[EMBEDDING];
    rms_normalize(input, output, EMBEDDING, gamma, 1e-6f);
    float loss = 0.0f;
    for (size_t d = 0; d < EMBEDDING; d++) loss += output[d] * upstream[d];
    return loss;
}

static int check_math(void) {
    float input[EMBEDDING], gamma[EMBEDDING], upstream[EMBEDDING];
    float output[EMBEDDING], xhat[EMBEDDING], rms[1];
    float dinput[EMBEDDING], dgamma[EMBEDDING] = {0};
    for (size_t d = 0; d < EMBEDDING; d++) {
        input[d] = (float)((int)d - 3) / 5.0f;
        gamma[d] = 0.7f + (float)d / 20.0f;
        upstream[d] = (float)((int)(d % 3) - 1) / 4.0f;
    }
    rms_norm_forward_cached(input, xhat, rms, output, gamma,
                            1, EMBEDDING, 1e-6f);
    float mean_square = 0.0f;
    for (size_t d = 0; d < EMBEDDING; d++) mean_square += input[d] * input[d];
    float expected_rms = sqrtf(mean_square / EMBEDDING + 1e-6f);
    if (!near(rms[0], expected_rms, 1e-6f)) return 0;
    for (size_t d = 0; d < EMBEDDING; d++) {
        if (!near(xhat[d], input[d] / expected_rms, 1e-6f) ||
            !near(output[d], gamma[d] * xhat[d], 1e-6f)) return 0;
    }
    rms_norm_backward(upstream, xhat, rms, gamma, dgamma, dinput,
                      1, EMBEDDING);

    const float epsilon = 1e-3f;
    for (size_t index = 0; index < EMBEDDING; index++) {
        float original = input[index];
        input[index] = original + epsilon;
        float plus = rms_loss(input, gamma, upstream);
        input[index] = original - epsilon;
        float minus = rms_loss(input, gamma, upstream);
        input[index] = original;
        if (!near(dinput[index], (plus - minus) / (2.0f * epsilon), 2e-3f))
            return 0;

        float expected_gamma = upstream[index] * xhat[index];
        if (!near(dgamma[index], expected_gamma, 1e-6f)) return 0;
    }
    return 1;
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
    snprintf(path, sizeof(path), "/tmp/dranzer-rmsnorm-%ld.bin", (long)getpid());
    bpe_encoder_t encoder = {0};
    bpe_encoder_t *loaded_encoder = NULL, *mapped_encoder = NULL;
    neural_model_t loaded = {0}, mapped = {0};
    model_bundle_metadata_t metadata = {.train_window = 4, .seed = 41};
    model_bundle_metadata_t loaded_metadata = {0}, mapped_metadata = {0};
    int ok = bpe_encoder_new_with_special_tokens(&encoder, VOCAB) == BPE_SUCCESS &&
             bpe_train(&encoder, "rms normalization fixture", 25) == BPE_SUCCESS &&
             bpe_encoder_freeze(&encoder) == BPE_SUCCESS &&
             model_bundle_save(model, &encoder, &metadata, path) == BUNDLE_SUCCESS &&
             model_bundle_load(&loaded, &loaded_encoder, &loaded_metadata,
                               path) == BUNDLE_SUCCESS &&
             model_bundle_load_mmap(&mapped, &mapped_encoder, &mapped_metadata,
                                    path) == BUNDLE_SUCCESS;
    if (ok) {
        ok = loaded_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             mapped_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             loaded.architecture_flags == MODEL_ARCH_RMSNORM &&
             mapped.architecture_flags == MODEL_ARCH_RMSNORM &&
             !loaded.layers[0].ln_beta_attn && !mapped.layers[0].ln_beta_ffn &&
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
    neural_model_t ordinary = {0}, rms = {0};
    size_t removed = 2 * LAYERS * EMBEDDING;
    int ok = check_math() &&
             model_new_seeded(&ordinary, VOCAB, EMBEDDING, HEADS, LAYERS,
                              MAX_SEQUENCE, 41) == MODEL_SUCCESS &&
             model_new_seeded_architecture(
                 &rms, VOCAB, EMBEDDING, HEADS, LAYERS, MAX_SEQUENCE, 41,
                 MODEL_ARCH_RMSNORM) == MODEL_SUCCESS &&
             model_uses_rmsnorm(&rms) &&
             !rms.layers[0].ln_beta_attn && !rms.layers[0].ln_beta_attn_grad &&
             !rms.layers[0].ln_beta_ffn && !rms.layers[0].ln_beta_ffn_grad &&
             ordinary.total_param_count == rms.total_param_count + removed &&
             check_cached_equivalence(&rms) && check_bundle_roundtrip(&rms);

    uint32_t tokens[3] = {2, 5, 9};
    float loss = 0.0f;
    model_zero_gradients(&rms);
    if (ok && (model_accumulate_gradients(&rms, tokens, 13, 3, &loss) !=
                   MODEL_SUCCESS || !isfinite(loss))) ok = 0;

    model_free(&ordinary);
    model_free(&rms);
    if (!ok) fprintf(stderr, "RMSNorm contract failed\n");
    else printf("\nRMSNORM CHECK PASSED\n");
    return ok ? 0 : 1;
}
