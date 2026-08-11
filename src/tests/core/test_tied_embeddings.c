#include "byte_pair_encoding.h"
#include "core/bundle.h"
#include "core/lm_head.h"
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
#define MAX_SEQUENCE 4

static int near(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <= tolerance;
}

static uint32_t read_u32(const uint8_t bytes[4]) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; i++) value |= (uint32_t)bytes[i] << (8 * i);
    return value;
}

static int is_v3_header(const char *path, uint32_t numeric) {
    uint8_t header[192];
    FILE *file = fopen(path, "rb");
    int ok = file && fread(header, 1, sizeof(header), file) == sizeof(header);
    if (file) fclose(file);
    return ok && memcmp(header, "DRNZBNDL", 8) == 0 &&
           read_u32(header + 8) == MODEL_BUNDLE_FORMAT_V3 &&
           read_u32(header + 16) == numeric && read_u32(header + 20) == 192 &&
           read_u32(header + 184) == MODEL_ARCH_TIED_EMBEDDINGS &&
           read_u32(header + 188) == 0;
}

static int check_projection_math(neural_model_t *model) {
    float hidden[2 * EMBEDDING];
    float logits[2 * VOCAB];
    float grad_logits[2 * VOCAB];
    float grad_hidden[2 * EMBEDDING] = {0};
    float expected_hidden[2 * EMBEDDING] = {0};
    float *expected_embeddings = calloc(VOCAB * EMBEDDING, sizeof(float));
    if (!expected_embeddings) return 0;

    for (size_t i = 0; i < 2 * EMBEDDING; i++) hidden[i] = (float)((int)i - 5) / 13.0f;
    for (size_t i = 0; i < VOCAB * EMBEDDING; i++)
        model->token_embeddings[i] = (float)((int)(i % 17) - 8) / 19.0f;
    for (size_t i = 0; i < 2 * VOCAB; i++)
        grad_logits[i] = (float)((int)(i % 11) - 5) / 23.0f;
    memset(model->token_embeddings_grad, 0,
           VOCAB * EMBEDDING * sizeof(float));

    if (lm_head_project(model, hidden, logits, 2) != MODEL_SUCCESS) {
        free(expected_embeddings);
        return 0;
    }
    for (size_t row = 0; row < 2; row++) {
        for (size_t token = 0; token < VOCAB; token++) {
            double expected = 0.0;
            for (size_t dim = 0; dim < EMBEDDING; dim++)
                expected += (double)hidden[row * EMBEDDING + dim] *
                            model->token_embeddings[token * EMBEDDING + dim];
            if (!near(logits[row * VOCAB + token], (float)expected, 1e-6f)) {
                free(expected_embeddings);
                return 0;
            }
        }
    }

    for (size_t row = 0; row < 2; row++) {
        for (size_t token = 0; token < VOCAB; token++) {
            float gradient = grad_logits[row * VOCAB + token];
            for (size_t dim = 0; dim < EMBEDDING; dim++) {
                expected_embeddings[token * EMBEDDING + dim] +=
                    hidden[row * EMBEDDING + dim] * gradient;
                expected_hidden[row * EMBEDDING + dim] +=
                    model->token_embeddings[token * EMBEDDING + dim] * gradient;
            }
        }
    }
    int ok = lm_head_project_backward(model, hidden, grad_logits,
                                      grad_hidden, 2) == MODEL_SUCCESS;
    for (size_t i = 0; ok && i < VOCAB * EMBEDDING; i++)
        ok = near(model->token_embeddings_grad[i], expected_embeddings[i], 1e-5f);
    for (size_t i = 0; ok && i < 2 * EMBEDDING; i++)
        ok = near(grad_hidden[i], expected_hidden[i], 1e-5f);
    free(expected_embeddings);
    return ok;
}

/* This parameter is used twice in one graph: token 1 is looked up at the
 * input, and every token row participates in the transposed output head.
 * A finite difference therefore checks that backward adds both paths to the
 * one stored gradient rather than dropping or double-applying either one. */
static int check_shared_end_to_end_gradient(neural_model_t *model) {
    uint32_t tokens[2] = {1, 7};
    const size_t index = EMBEDDING;
    const float original = model->token_embeddings[index];
    const float epsilon = 1e-3f;
    float center_loss = 0.0f, plus_loss = 0.0f, minus_loss = 0.0f;

    model->dropout_rate = 0.0f;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 11, 2,
                                   &center_loss) != MODEL_SUCCESS) return 0;
    float analytic = model->token_embeddings_grad[index];

    model->token_embeddings[index] = original + epsilon;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 11, 2,
                                   &plus_loss) != MODEL_SUCCESS) return 0;
    model->token_embeddings[index] = original - epsilon;
    model_zero_gradients(model);
    if (model_accumulate_gradients(model, tokens, 11, 2,
                                   &minus_loss) != MODEL_SUCCESS) return 0;
    model->token_embeddings[index] = original;
    model_zero_gradients(model);

    float numerical = (plus_loss - minus_loss) / (2.0f * epsilon);
    float tolerance = 2e-2f + 2e-2f * fabsf(numerical);
    return isfinite(center_loss) && isfinite(analytic) && isfinite(numerical) &&
           near(analytic, numerical, tolerance);
}

static int check_bundle_roundtrip(neural_model_t *model) {
    char float_path[160], quant_path[160];
    snprintf(float_path, sizeof(float_path), "/tmp/dranzer-tied-%ld.bin", (long)getpid());
    snprintf(quant_path, sizeof(quant_path), "/tmp/dranzer-tied-quant-%ld.bin",
             (long)getpid());
    bpe_encoder_t encoder = {0};
    neural_model_t loaded = {0}, mapped = {0}, quantized = {0};
    bpe_encoder_t *loaded_encoder = NULL, *mapped_encoder = NULL, *quant_encoder = NULL;
    model_bundle_metadata_t metadata = {.train_window = 4, .seed = 17};
    model_bundle_metadata_t loaded_metadata = {0}, mapped_metadata = {0}, quant_metadata = {0};
    model_quant_config_t quant_config;
    int ok = bpe_encoder_new_with_special_tokens(&encoder, VOCAB) == BPE_SUCCESS &&
             bpe_train(&encoder, "tied embeddings fixture", 23) == BPE_SUCCESS &&
             bpe_encoder_freeze(&encoder) == BPE_SUCCESS &&
             model_bundle_save(model, &encoder, &metadata, float_path) == BUNDLE_SUCCESS &&
             is_v3_header(float_path, 1) &&
             model_bundle_load(&loaded, &loaded_encoder, &loaded_metadata,
                               float_path) == BUNDLE_SUCCESS &&
             model_bundle_load_mmap(&mapped, &mapped_encoder, &mapped_metadata,
                                    float_path) == BUNDLE_SUCCESS;
    if (ok) {
        ok = loaded_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             mapped_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             loaded.architecture_flags == MODEL_ARCH_TIED_EMBEDDINGS &&
             mapped.architecture_flags == MODEL_ARCH_TIED_EMBEDDINGS &&
             !loaded.output_projection && !mapped.output_projection &&
             loaded.total_param_count == model->total_param_count &&
             memcmp(loaded.params, model->params,
                    model->total_param_count * sizeof(float)) == 0 &&
             memcmp(mapped.params, model->params,
                    model->total_param_count * sizeof(float)) == 0;
    }

    model_quantize_default_config(&quant_config);
    quant_config.bits = 8;
    if (ok) {
        ok = model_bundle_save_quantized(model, &encoder, &metadata,
                                         &quant_config, NULL,
                                         quant_path) == BUNDLE_SUCCESS &&
             is_v3_header(quant_path, 2) &&
             model_bundle_load(&quantized, &quant_encoder, &quant_metadata,
                               quant_path) == BUNDLE_SUCCESS &&
             quant_metadata.format_version == MODEL_BUNDLE_FORMAT_V3 &&
             quantized.architecture_flags == MODEL_ARCH_TIED_EMBEDDINGS &&
             !quantized.output_projection;
    }
    if (ok) ok = model_save(model, float_path) == MODEL_INVALID_INPUT;

    model_free(&loaded);
    model_free(&mapped);
    model_free(&quantized);
    if (loaded_encoder) { bpe_encoder_free(loaded_encoder); free(loaded_encoder); }
    if (mapped_encoder) { bpe_encoder_free(mapped_encoder); free(mapped_encoder); }
    if (quant_encoder) { bpe_encoder_free(quant_encoder); free(quant_encoder); }
    bpe_encoder_free(&encoder);
    remove(float_path);
    remove(quant_path);
    return ok;
}

int main(void) {
    neural_model_t untied = {0}, tied = {0};
    size_t removed = VOCAB * EMBEDDING;
    int ok = model_new_seeded(&untied, VOCAB, EMBEDDING, HEADS, LAYERS,
                              MAX_SEQUENCE, 9) == MODEL_SUCCESS &&
             model_new_seeded_architecture(
                 &tied, VOCAB, EMBEDDING, HEADS, LAYERS, MAX_SEQUENCE, 9,
                 MODEL_ARCH_TIED_EMBEDDINGS) == MODEL_SUCCESS &&
             model_uses_tied_embeddings(&tied) &&
             !tied.output_projection && !tied.output_projection_grad &&
             untied.total_param_count == tied.total_param_count + removed &&
             check_projection_math(&tied) &&
             check_shared_end_to_end_gradient(&tied) &&
             check_bundle_roundtrip(&tied);

    neural_model_t rejected = {0};
    if (model_new_seeded_architecture(
            &rejected, VOCAB, EMBEDDING, HEADS, LAYERS, MAX_SEQUENCE, 9,
            UINT32_C(0x80000000)) != MODEL_INVALID_INPUT) ok = 0;
    model_free(&rejected);
    model_free(&untied);
    model_free(&tied);
    if (!ok) fprintf(stderr, "tied embeddings contract failed\n");
    else printf("\nTIED EMBEDDINGS CHECK PASSED\n");
    return ok ? 0 : 1;
}
