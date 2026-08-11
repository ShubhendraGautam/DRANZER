/* Versioned model bundle contract: canonical weights, embedded frozen
 * tokenizer, metadata, legacy detection, and safe corruption rejection. */
#include "byte_pair_encoding.h"
#include "core/bundle.h"
#include "core/model.h"
#include "core/weight_decay.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VOCAB 264
#define EMBEDDING 8
#define HEADS 2
#define LAYERS 1
#define MAX_SEQUENCE 8
#define BUNDLE_HEADER_SIZE 152
#define QUANTIZED_HEADER_SIZE 184

static int write_blob(const char *path, const uint8_t *data, size_t size) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(data, 1, size, file) == size;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static uint8_t *read_blob(const char *path, size_t *out_size) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0) {
        if (file) fclose(file);
        return NULL;
    }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    uint8_t *data = malloc((size_t)length);
    if (!data || fread(data, 1, (size_t)length, file) != (size_t)length) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *out_size = (size_t)length;
    return data;
}

static void put_u64(uint8_t bytes[8], uint64_t value) {
    for (size_t i = 0; i < 8; i++) bytes[i] = (uint8_t)(value >> (8 * i));
}

static void put_u32(uint8_t bytes[4], uint32_t value) {
    for (size_t i = 0; i < 4; i++) bytes[i] = (uint8_t)(value >> (8 * i));
}

static uint32_t get_u32(const uint8_t bytes[4]) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; i++) value |= (uint32_t)bytes[i] << (8 * i);
    return value;
}

static uint64_t get_u64(const uint8_t bytes[8]) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; i++) value |= (uint64_t)bytes[i] << (8 * i);
    return value;
}

static uint64_t checksum_bytes(const uint8_t *bytes, size_t size) {
    uint64_t checksum = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; i++) {
        checksum ^= bytes[i];
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static void refresh_header_checksum(uint8_t *header, size_t header_size) {
    memset(header + 76, 0, 4);
    memset(header + 84, 0, 4);
    uint64_t checksum = checksum_bytes(header, header_size);
    put_u32(header + 76, (uint32_t)checksum);
    put_u32(header + 84, (uint32_t)(checksum >> 32));
}

static void refresh_weight_checksum(uint8_t *bundle, size_t header_size) {
    size_t weight_size = (size_t)get_u64(bundle + 120);
    uint64_t checksum = checksum_bytes(bundle + header_size, weight_size);
    put_u64(bundle + 136, checksum);
    refresh_header_checksum(bundle, header_size);
}

static bundle_errors_t load_and_release(const char *path) {
    neural_model_t loaded = {0};
    bpe_encoder_t *encoder = NULL;
    model_bundle_metadata_t metadata = {0};
    bundle_errors_t rc = model_bundle_load(&loaded, &encoder, &metadata, path);
    if (rc == BUNDLE_SUCCESS) {
        model_free(&loaded);
        bpe_encoder_free(encoder);
        free(encoder);
    }
    return rc;
}

static bundle_errors_t mmap_load_and_release(const char *path) {
    neural_model_t loaded = {0};
    bpe_encoder_t *encoder = NULL;
    model_bundle_metadata_t metadata = {0};
    bundle_errors_t rc = model_bundle_load_mmap(
        &loaded, &encoder, &metadata, path);
    if (rc == BUNDLE_SUCCESS) {
        model_free(&loaded);
        bpe_encoder_free(encoder);
        free(encoder);
    }
    return rc;
}

int main(void) {
    char bundle_path[160], quantized_path[160], corrupt_path[160], legacy_path[160];
    snprintf(bundle_path, sizeof(bundle_path), "/tmp/dranzer-bundle-%ld.bin", (long)getpid());
    snprintf(quantized_path, sizeof(quantized_path),
             "/tmp/dranzer-bundle-quantized-%ld.bin", (long)getpid());
    snprintf(corrupt_path, sizeof(corrupt_path), "/tmp/dranzer-corrupt-%ld.bin", (long)getpid());
    snprintf(legacy_path, sizeof(legacy_path), "/tmp/dranzer-legacy-%ld.pth", (long)getpid());
    int failed = 0;
    neural_model_t model = {0};
    bpe_encoder_t encoder = {0};
    bpe_tokens_t original_tokens = {0}, loaded_tokens = {0};
    uint8_t *baseline = NULL;
    size_t baseline_size = 0;
    const char *training_text = "banana bandana banana cabana";
    const char *probe_text = "banana cabana";

    if (model_new_seeded(&model, VOCAB, EMBEDDING, HEADS, LAYERS, MAX_SEQUENCE, 73) != MODEL_SUCCESS ||
        bpe_encoder_new_with_special_tokens(&encoder, VOCAB) != BPE_SUCCESS ||
        bpe_train(&encoder, training_text, strlen(training_text)) != BPE_SUCCESS ||
        bpe_encoder_freeze(&encoder) != BPE_SUCCESS) {
        fprintf(stderr, "bundle fixture setup failed\n");
        failed = 1;
        goto cleanup;
    }
    uint32_t context[] = {1, 2, 3, 4};
    for (int i = 0; i < 3; i++) {
        if (model_train_step(&model, context, 5, 4) != MODEL_SUCCESS) {
            fprintf(stderr, "bundle fixture training failed\n");
            failed = 1;
            goto cleanup;
        }
    }
    model_bundle_metadata_t metadata = {
        .train_window = 4,
        .seed = UINT64_C(0x123456789abcdef0),
        .input_fingerprint = UINT64_C(0xfedcba9876543210),
        .input_bytes = 987654,
    };
    if (model_bundle_save(&model, &encoder, &metadata, bundle_path) != BUNDLE_SUCCESS ||
        bpe_encode(&encoder, probe_text, strlen(probe_text), &original_tokens) != BPE_SUCCESS) {
        fprintf(stderr, "bundle save failed\n");
        failed = 1;
        goto cleanup;
    }

    neural_model_t loaded = {0};
    bpe_encoder_t *loaded_encoder = NULL;
    model_bundle_metadata_t loaded_metadata = {0};
    if (model_bundle_load(&loaded, &loaded_encoder, &loaded_metadata, bundle_path) !=
            BUNDLE_SUCCESS ||
        !loaded_encoder || !bpe_encoder_is_frozen(loaded_encoder) ||
        !bpe_encoder_has_special_tokens(loaded_encoder) ||
        loaded.total_param_count != model.total_param_count ||
        memcmp(loaded.params, model.params,
               model.total_param_count * sizeof(float)) != 0 ||
        loaded.training_steps != model.training_steps ||
        memcmp(&loaded.current_loss, &model.current_loss, sizeof(float)) != 0 ||
        loaded_metadata.train_window != metadata.train_window ||
        loaded_metadata.seed != metadata.seed ||
        loaded_metadata.input_fingerprint != metadata.input_fingerprint ||
        loaded_metadata.input_bytes != metadata.input_bytes ||
        loaded_metadata.format_version != MODEL_BUNDLE_FORMAT_V1 ||
        loaded_encoder->max_vocab_size != encoder.max_vocab_size ||
        loaded_encoder->vocab_size != encoder.vocab_size ||
        bpe_encode(loaded_encoder, probe_text, strlen(probe_text), &loaded_tokens) != BPE_SUCCESS ||
        loaded_tokens.token_count != original_tokens.token_count ||
        memcmp(loaded_tokens.token_ids, original_tokens.token_ids,
               original_tokens.token_count * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "bundle roundtrip changed model, tokenizer, or metadata\n");
        failed = 1;
    }
    model_free(&loaded);
    if (loaded_encoder) {
        bpe_encoder_free(loaded_encoder);
        free(loaded_encoder);
    }
    if (failed) goto cleanup;

    /* The mmap path preserves the exact version-1 representation while
     * borrowing its parameter bytes directly from a read-only file mapping.
     * Inference remains available; every training entry point must reject the
     * model before attempting to update those bytes. */
    neural_model_t mapped = {0};
    bpe_encoder_t *mapped_encoder = NULL;
    model_bundle_metadata_t mapped_metadata = {0};
    if (model_bundle_load_mmap(&mapped, &mapped_encoder, &mapped_metadata,
                               bundle_path) != BUNDLE_SUCCESS ||
        !mapped_encoder || !mapped.params_mapping ||
        mapped.params_mapping_size == 0 || mapped.params_owned ||
        !mapped.params_read_only ||
        mapped.params != (float *)((uint8_t *)mapped.params_mapping +
                                   BUNDLE_HEADER_SIZE) ||
        mapped.total_param_count != model.total_param_count ||
        memcmp(mapped.params, model.params,
               model.total_param_count * sizeof(float)) != 0 ||
        mapped.training_steps != model.training_steps ||
        memcmp(&mapped.current_loss, &model.current_loss, sizeof(float)) != 0 ||
        mapped_metadata.train_window != metadata.train_window ||
        mapped_metadata.seed != metadata.seed ||
        mapped_metadata.input_fingerprint != metadata.input_fingerprint ||
        mapped_metadata.input_bytes != metadata.input_bytes ||
        mapped_metadata.format_version != MODEL_BUNDLE_FORMAT_V1 ||
        mapped_encoder->vocab_size != encoder.vocab_size ||
        model_predict_next_token(&mapped, context, 4) !=
            model_predict_next_token(&model, context, 4)) {
        fprintf(stderr, "mmap bundle load changed model, tokenizer, or metadata\n");
        failed = 1;
    } else {
        uint32_t target = 5;
        model_quant_config_t rejected_quantization;
        model_quantize_default_config(&rejected_quantization);
        rejected_quantization.bits = 8;
        if (model_train_step(&mapped, context, target, 4) !=
                MODEL_INVALID_INPUT ||
            model_optimizer_step(&mapped) != MODEL_INVALID_INPUT ||
            model_apply_weight_decay(&mapped, 0.001f, 0.01f) !=
                MODEL_INVALID_INPUT ||
            model_quantize_weights(&mapped, &rejected_quantization, NULL) != -1 ||
            memcmp(mapped.params, model.params,
                   model.total_param_count * sizeof(float)) != 0) {
            fprintf(stderr, "mmap model was not enforced as inference-only\n");
            failed = 1;
        }
    }
    model_free(&mapped);
    if (mapped_encoder) {
        bpe_encoder_free(mapped_encoder);
        free(mapped_encoder);
    }
    if (failed) goto cleanup;

    baseline = read_blob(bundle_path, &baseline_size);
    size_t weights_size = model.total_param_count * sizeof(float);
    if (!baseline || baseline_size <= BUNDLE_HEADER_SIZE + weights_size + 8) {
        fprintf(stderr, "could not inspect saved bundle\n");
        failed = 1;
        goto cleanup;
    }
    if (memcmp(baseline, "DRNZBNDL", 8) != 0 ||
        get_u32(baseline + 8) != MODEL_BUNDLE_FORMAT_V1 ||
        get_u32(baseline + 16) != 1 ||
        get_u32(baseline + 20) != BUNDLE_HEADER_SIZE ||
        memcmp(baseline + baseline_size - 8, "DRNZDONE", 8) != 0) {
        fprintf(stderr, "version-1 fixed wire header changed\n");
        failed = 1;
        goto cleanup;
    }

    /* Version 2 stores only policy-selected tensors as packed codes. Loading
     * reconstructs the same float grid as the part-1 simulated quantizer, and
     * the report ties the measured artifact size to that accuracy policy. */
    model_quant_config_t quant_config;
    model_quantize_default_config(&quant_config);
    quant_config.bits = 4;
    neural_model_t expected_quantized = {0};
    model_quant_report_t accuracy_report = {0};
    model_bundle_storage_report_t storage_report = {0};
    if (model_new(&expected_quantized, VOCAB, EMBEDDING, HEADS, LAYERS,
                  MAX_SEQUENCE) != MODEL_SUCCESS) {
        fprintf(stderr, "quantized bundle reference allocation failed\n");
        failed = 1;
        goto cleanup;
    }
    memcpy(expected_quantized.params, model.params,
           model.total_param_count * sizeof(float));
    if (model_quantize_weights(&expected_quantized, &quant_config,
                               &accuracy_report) != 0 ||
        model_bundle_save_quantized(&model, &encoder, &metadata, &quant_config,
                                    &storage_report, quantized_path) != BUNDLE_SUCCESS) {
        fprintf(stderr, "quantized bundle save failed\n");
        model_free(&expected_quantized);
        failed = 1;
        goto cleanup;
    }
    size_t quantized_size = 0;
    uint8_t *quantized_blob = read_blob(quantized_path, &quantized_size);
    neural_model_t quantized_loaded = {0};
    bpe_encoder_t *quantized_encoder = NULL;
    model_bundle_metadata_t quantized_metadata = {0};
    if (!quantized_blob || quantized_size <= QUANTIZED_HEADER_SIZE + 8 ||
        storage_report.artifact_bytes != quantized_size ||
        storage_report.weight_payload_bytes >= weights_size ||
        quantized_size >= baseline_size ||
        memcmp(quantized_blob, "DRNZBNDL", 8) != 0 ||
        get_u32(quantized_blob + 8) != MODEL_BUNDLE_FORMAT_V2 ||
        get_u32(quantized_blob + 16) != 2 ||
        get_u32(quantized_blob + 20) != QUANTIZED_HEADER_SIZE ||
        memcmp(quantized_blob + quantized_size - 8, "DRNZDONE", 8) != 0 ||
        storage_report.values_quantized != accuracy_report.values_quantized ||
        storage_report.tensors_quantized != accuracy_report.tensors_quantized ||
        model_bundle_load(&quantized_loaded, &quantized_encoder,
                          &quantized_metadata, quantized_path) != BUNDLE_SUCCESS ||
        memcmp(quantized_loaded.params, expected_quantized.params,
               model.total_param_count * sizeof(float)) != 0 ||
        quantized_loaded.training_steps != model.training_steps ||
        quantized_metadata.seed != metadata.seed ||
        quantized_metadata.format_version != MODEL_BUNDLE_FORMAT_V2 ||
        !quantized_encoder || !bpe_encoder_is_frozen(quantized_encoder)) {
        fprintf(stderr, "quantized bundle roundtrip or size accounting failed\n");
        failed = 1;
    }
    if (mmap_load_and_release(quantized_path) != BUNDLE_UNSUPPORTED) {
        fprintf(stderr, "mmap loader did not reject unpacking-required v2 bundle\n");
        failed = 1;
    }
    model_free(&expected_quantized);
    model_free(&quantized_loaded);
    if (quantized_encoder) {
        bpe_encoder_free(quantized_encoder);
        free(quantized_encoder);
    }
    if (failed) {
        free(quantized_blob);
        goto cleanup;
    }

    quantized_blob[QUANTIZED_HEADER_SIZE + 7] ^= UINT8_C(0x20);
    if (!write_blob(corrupt_path, quantized_blob, quantized_size) ||
        load_and_release(corrupt_path) != BUNDLE_CHECKSUM_ERROR) {
        fprintf(stderr, "corrupt quantized payload was not rejected by checksum\n");
        free(quantized_blob);
        failed = 1;
        goto cleanup;
    }
    quantized_blob[QUANTIZED_HEADER_SIZE + 7] ^= UINT8_C(0x20);

    /* A forged-but-checksummed tensor shape must not redirect bytes into a
     * different parameter view. */
    quantized_blob[QUANTIZED_HEADER_SIZE + 8] ^= UINT8_C(1);
    refresh_weight_checksum(quantized_blob, QUANTIZED_HEADER_SIZE);
    if (!write_blob(corrupt_path, quantized_blob, quantized_size) ||
        load_and_release(corrupt_path) != BUNDLE_FORMAT_ERROR) {
        fprintf(stderr, "quantized tensor shape mismatch was not rejected\n");
        free(quantized_blob);
        failed = 1;
        goto cleanup;
    }
    free(quantized_blob);

    /* Weight and tokenizer payloads have independent checksums. */
    baseline[BUNDLE_HEADER_SIZE + 3] ^= UINT8_C(0x40);
    if (!write_blob(corrupt_path, baseline, baseline_size) ||
        load_and_release(corrupt_path) != BUNDLE_CHECKSUM_ERROR ||
        mmap_load_and_release(corrupt_path) != BUNDLE_CHECKSUM_ERROR) {
        fprintf(stderr, "corrupt weight payload was not rejected by checksum\n");
        failed = 1;
        goto cleanup;
    }
    baseline[BUNDLE_HEADER_SIZE + 3] ^= UINT8_C(0x40);
    baseline[BUNDLE_HEADER_SIZE + weights_size + 23] ^= UINT8_C(0x08);
    if (!write_blob(corrupt_path, baseline, baseline_size) ||
        load_and_release(corrupt_path) != BUNDLE_CHECKSUM_ERROR) {
        fprintf(stderr, "corrupt tokenizer payload was not rejected by checksum\n");
        failed = 1;
        goto cleanup;
    }
    baseline[BUNDLE_HEADER_SIZE + weights_size + 23] ^= UINT8_C(0x08);

    if (!write_blob(corrupt_path, baseline, baseline_size - 1) ||
        load_and_release(corrupt_path) != BUNDLE_FORMAT_ERROR) {
        fprintf(stderr, "truncated bundle was not rejected\n");
        failed = 1;
        goto cleanup;
    }
    baseline[8] = 2;
    refresh_header_checksum(baseline, BUNDLE_HEADER_SIZE);
    if (!write_blob(corrupt_path, baseline, baseline_size) ||
        load_and_release(corrupt_path) != BUNDLE_UNSUPPORTED) {
        fprintf(stderr, "unknown bundle version was not rejected\n");
        failed = 1;
        goto cleanup;
    }
    baseline[8] = 1;
    refresh_header_checksum(baseline, BUNDLE_HEADER_SIZE);

    /* A corrupted max-sequence header must fail shape validation before it
     * can request a huge quadratic attention cache. */
    memset(baseline + 56, 0xff, 8);
    refresh_header_checksum(baseline, BUNDLE_HEADER_SIZE);
    if (!write_blob(corrupt_path, baseline, baseline_size) ||
        load_and_release(corrupt_path) != BUNDLE_FORMAT_ERROR) {
        fprintf(stderr, "unsafe model shape was not rejected\n");
        failed = 1;
        goto cleanup;
    }
    free(baseline);
    baseline = read_blob(bundle_path, &baseline_size);
    if (!baseline) { failed = 1; goto cleanup; }

    /* Deterministic mutation sweep: every loader result is bounded and the
     * sanitizer matrix verifies that all error paths release allocations. */
    for (size_t i = 0; i < 96; i++) {
        size_t offset = (i * (baseline_size - 1)) / 95;
        baseline[offset] ^= (uint8_t)(1u << (i % 8));
        if (!write_blob(corrupt_path, baseline, baseline_size)) {
            failed = 1;
            break;
        }
        bundle_errors_t rc = load_and_release(corrupt_path);
        bundle_errors_t mmap_rc = mmap_load_and_release(corrupt_path);
        if (rc <= BUNDLE_SUCCESS || rc > BUNDLE_UNSUPPORTED ||
            mmap_rc <= BUNDLE_SUCCESS || mmap_rc > BUNDLE_UNSUPPORTED) {
            fprintf(stderr, "mutated bundle was accepted or produced invalid status\n");
            failed = 1;
            break;
        }
        baseline[offset] ^= (uint8_t)(1u << (i % 8));
    }
    if (failed) goto cleanup;

    /* The portable tokenizer decoder refuses a sparse billion-entry table
     * before allocating. */
    uint8_t malicious_tokenizer[24] = {0};
    put_u64(malicious_tokenizer, UINT64_C(0xffffffff));
    put_u64(malicious_tokenizer + 8, 256);
    malicious_tokenizer[16] = 1;
    bpe_encoder_t rejected = {0};
    if (bpe_encoder_deserialize_portable(
            &rejected, malicious_tokenizer, sizeof(malicious_tokenizer)) != BPE_FORMAT_ERROR) {
        fprintf(stderr, "oversized portable tokenizer was not rejected\n");
        bpe_encoder_free(&rejected);
        failed = 1;
        goto cleanup;
    }

    /* Compatibility fixture: old host-native weight files are detected as
     * non-bundles and remain readable by the legacy loader. */
    if (model_save(&model, legacy_path) != MODEL_SUCCESS ||
        load_and_release(legacy_path) != BUNDLE_NOT_BUNDLE ||
        mmap_load_and_release(legacy_path) != BUNDLE_NOT_BUNDLE) {
        fprintf(stderr, "legacy artifact detection failed\n");
        failed = 1;
        goto cleanup;
    }
    neural_model_t legacy = {0};
    if (model_load(&legacy, legacy_path) != MODEL_SUCCESS ||
        legacy.total_param_count != model.total_param_count ||
        memcmp(legacy.params, model.params,
               model.total_param_count * sizeof(float)) != 0) {
        fprintf(stderr, "legacy artifact no longer loads\n");
        failed = 1;
    }
    model_free(&legacy);

    /* Retained x86-64 little-endian fixture representing the exact
     * host-native layout written before bundle support. Designated bytes
     * keep it independent of the current legacy writer implementation. */
    const uint16_t endian_probe = 1;
    if (sizeof(size_t) == 8 && sizeof(float) == 4 &&
        *(const uint8_t *)&endian_probe == 1) {
        static const uint8_t release_fixture[152] = {
            [0] = 1, [8] = 1, [16] = 1, [24] = 1, [32] = 1,
            [40] = 7, [46] = 0x80, [47] = 0x3f, [48] = 24,
        };
        if (!write_blob(legacy_path, release_fixture, sizeof(release_fixture))) {
            failed = 1;
            goto cleanup;
        }
        neural_model_t fixture_model = {0};
        if (load_and_release(legacy_path) != BUNDLE_NOT_BUNDLE ||
            model_load(&fixture_model, legacy_path) != MODEL_SUCCESS ||
            fixture_model.vocab_size != 1 || fixture_model.embedding_dim != 1 ||
            fixture_model.num_heads != 1 || fixture_model.num_layers != 1 ||
            fixture_model.max_seq_len != 1 || fixture_model.total_param_count != 24 ||
            fixture_model.training_steps != 7 || fixture_model.current_loss != 1.0f) {
            fprintf(stderr, "retained legacy release fixture no longer loads\n");
            failed = 1;
        }
        model_free(&fixture_model);
    }

cleanup:
    free(baseline);
    bpe_tokens_free(&original_tokens);
    bpe_tokens_free(&loaded_tokens);
    bpe_encoder_free(&encoder);
    model_free(&model);
    remove(bundle_path);
    remove(quantized_path);
    remove(corrupt_path);
    remove(legacy_path);
    printf("\n%s\n", failed ? "MODEL BUNDLE CHECK FAILED" : "MODEL BUNDLE CHECK PASSED");
    return failed ? 1 : 0;
}
