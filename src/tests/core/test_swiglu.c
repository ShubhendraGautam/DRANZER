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
#define FFN (EMBEDDING * 4)
#define HEADS 2
#define LAYERS 1
#define MAX_SEQUENCE 5

static int near(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <= tolerance;
}

static int check_scalar_math(void) {
    const float probes[] = {-4.0f, -1.0f, 0.0f, 0.75f, 3.0f};
    const float epsilon = 1e-3f;
    if (silu(0.0f) != 0.0f || !near(silu_derivative(0.0f), 0.5f, 1e-7f))
        return 0;
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        float numerical = (silu(probes[i] + epsilon) -
                           silu(probes[i] - epsilon)) / (2.0f * epsilon);
        if (!near(silu_derivative(probes[i]), numerical, 8e-4f)) return 0;
    }
    return near(silu(-100.0f), 0.0f, 1e-6f) &&
           near(silu(100.0f), 100.0f, 1e-5f);
}

static int check_activation_contract(neural_model_t *model) {
    uint32_t tokens[3] = {1, 4, 7};
    if (model_forward_hidden(model, tokens, 3) != MODEL_SUCCESS) return 0;
    transformer_layer_t *layer = &model->layers[0];
    for (size_t row = 0; row < 3; row++) {
        const float *input = &model->cache_attn_ln_out[0][row * EMBEDDING];
        for (size_t d = 0; d < FFN; d++) {
            float pre_activation = layer->b_ff1[d];
            float gate = layer->b_ff_gate[d];
            for (size_t input_dim = 0; input_dim < EMBEDDING; input_dim++) {
                pre_activation += input[input_dim] *
                    layer->W_ff1[input_dim * FFN + d];
                gate += input[input_dim] *
                    layer->W_ff_gate[input_dim * FFN + d];
            }
            size_t index = row * FFN + d;
            if (!near(model->cache_ff_pre_activation[0][index],
                      pre_activation, 2e-5f) ||
                !near(model->cache_ff_gate[0][index], gate, 2e-5f) ||
                !near(model->cache_ff_hidden[0][index],
                      silu(pre_activation) * gate, 2e-5f)) return 0;
        }
    }
    return 1;
}

static int check_one_gradient(neural_model_t *model, float *parameter,
                              float *gradient) {
    uint32_t tokens[3] = {1, 4, 7};
    float original = *parameter;
    const float epsilon = 1e-3f;
    float center = 0.0f, plus = 0.0f, minus = 0.0f;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 9, 3, &center) != MODEL_SUCCESS)
        return 0;
    float analytic = *gradient;
    *parameter = original + epsilon;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 9, 3, &plus) != MODEL_SUCCESS) {
        *parameter = original;
        return 0;
    }
    *parameter = original - epsilon;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 9, 3, &minus) != MODEL_SUCCESS) {
        *parameter = original;
        return 0;
    }
    *parameter = original;
    model_zero_gradients(model);
    float numerical = (plus - minus) / (2.0f * epsilon);
    return isfinite(center) && isfinite(analytic) && isfinite(numerical) &&
           near(analytic, numerical, 4e-2f + 4e-2f * fabsf(numerical));
}

static int check_cached_equivalence(neural_model_t *model) {
    uint32_t tokens[4] = {1, 7, 3, 11};
    float full[VOCAB], cached[VOCAB];
    model_kv_cache_t cache = {0};
    if (model_kv_cache_init(&cache, model) != MODEL_SUCCESS) return 0;
    int ok = cache.ff_gate != NULL;
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
    char path[160], quantized_path[160];
    snprintf(path, sizeof(path), "/tmp/dranzer-swiglu-%ld.bin", (long)getpid());
    snprintf(quantized_path, sizeof(quantized_path),
             "/tmp/dranzer-swiglu-quant-%ld.bin", (long)getpid());
    const char fixture[] = "SwiGLU bundle fixture";
    bpe_encoder_t encoder = {0};
    bpe_encoder_t *loaded_encoder = NULL, *mapped_encoder = NULL;
    bpe_encoder_t *quantized_encoder = NULL;
    neural_model_t loaded = {0}, mapped = {0}, quantized = {0};
    model_bundle_metadata_t metadata = {.train_window = 4, .seed = 61};
    model_bundle_metadata_t loaded_metadata = {0}, mapped_metadata = {0};
    model_bundle_metadata_t quantized_metadata = {0};
    model_bundle_storage_report_t storage_report = {0};
    model_quant_config_t quantized_config;
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
             loaded.architecture_flags == MODEL_ARCH_SWIGLU &&
             mapped.architecture_flags == MODEL_ARCH_SWIGLU &&
             loaded.layers[0].W_ff_gate && mapped.layers[0].W_ff_gate &&
             loaded.cache_ff_gate[0] && mapped.cache_ff_gate[0] &&
             loaded.total_param_count == model->total_param_count &&
             memcmp(loaded.params, model->params,
                    model->total_param_count * sizeof(float)) == 0 &&
             memcmp(mapped.params, model->params,
                    model->total_param_count * sizeof(float)) == 0;
    }
    model_quantize_default_config(&quantized_config);
    quantized_config.bits = 8;
    if (ok) {
        ok = model_bundle_save_quantized(
                 model, &encoder, &metadata, &quantized_config,
                 &storage_report, quantized_path) == BUNDLE_SUCCESS &&
             model_bundle_load(&quantized, &quantized_encoder,
                               &quantized_metadata,
                               quantized_path) == BUNDLE_SUCCESS &&
             quantized_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             quantized.architecture_flags == MODEL_ARCH_SWIGLU &&
             quantized.layers[0].W_ff_gate &&
             quantized.total_param_count == model->total_param_count &&
             storage_report.tensors_quantized > 0;
    }
    model_free(&loaded);
    model_free(&mapped);
    model_free(&quantized);
    if (loaded_encoder) { bpe_encoder_free(loaded_encoder); free(loaded_encoder); }
    if (mapped_encoder) { bpe_encoder_free(mapped_encoder); free(mapped_encoder); }
    if (quantized_encoder) {
        bpe_encoder_free(quantized_encoder);
        free(quantized_encoder);
    }
    bpe_encoder_free(&encoder);
    remove(path);
    remove(quantized_path);
    return ok;
}

int main(void) {
    neural_model_t ordinary = {0}, model = {0}, rejected = {0};
    size_t added = LAYERS * (EMBEDDING * FFN + FFN);
    int ok = check_scalar_math() &&
             model_new_seeded(&ordinary, VOCAB, EMBEDDING, HEADS, LAYERS,
                              MAX_SEQUENCE, 61) == MODEL_SUCCESS &&
             model_new_seeded_architecture(
                 &model, VOCAB, EMBEDDING, HEADS, LAYERS, MAX_SEQUENCE, 61,
                 MODEL_ARCH_SWIGLU) == MODEL_SUCCESS;
    if (ok) {
        model.use_scalar_matmul = 1;
        ok = model_uses_swiglu(&model) && !model_uses_gelu(&model) &&
             !ordinary.layers[0].W_ff_gate && !ordinary.cache_ff_gate[0] &&
             model.layers[0].W_ff_gate && model.layers[0].W_ff_gate_grad &&
             model.layers[0].b_ff_gate && model.layers[0].b_ff_gate_grad &&
             model.cache_ff_pre_activation[0] && model.cache_ff_gate[0] &&
             model.ws_d_ff_gate &&
             ordinary.total_param_count + added == model.total_param_count &&
             check_activation_contract(&model) &&
             check_cached_equivalence(&model) &&
             check_one_gradient(&model, &model.layers[0].W_ff1[0],
                                &model.layers[0].W_ff1_grad[0]) &&
             check_one_gradient(&model, &model.layers[0].W_ff_gate[0],
                                &model.layers[0].W_ff_gate_grad[0]) &&
             check_bundle_roundtrip(&model) &&
             model_new_seeded_architecture(
                 &rejected, VOCAB, EMBEDDING, HEADS, LAYERS, MAX_SEQUENCE, 61,
                 MODEL_ARCH_GELU | MODEL_ARCH_SWIGLU) == MODEL_INVALID_INPUT;
    }
    model_free(&rejected);
    model_free(&ordinary);
    model_free(&model);
    if (!ok) fprintf(stderr, "SwiGLU contract failed\n");
    else printf("\nSWIGLU CHECK PASSED\n");
    return ok ? 0 : 1;
}
