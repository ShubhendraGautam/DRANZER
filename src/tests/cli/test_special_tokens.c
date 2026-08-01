#include "byte_pair_encoding.h"
#include "cli/generation.h"
#include "cli/sampling.h"
#include "core/model.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char path[160];
    snprintf(path, sizeof(path), "/tmp/dranzer-special-tokens-%ld.bin", (long)getpid());
    const char *text = "banana bandana banana bandana";
    bpe_encoder_t encoder = {0}, sidecar = {0}, portable = {0}, rejected = {0};
    bpe_encoder_t legacy = {0}, legacy_portable = {0};
    bpe_tokens_t encoded = {0}, sidecar_tokens = {0};
    uint8_t *portable_data = NULL;
    size_t portable_size = 0;
    uint8_t *legacy_portable_data = NULL;
    size_t legacy_portable_size = 0;
    char *decoded = NULL;
    size_t decoded_size = 0;
    int failed = 0;

    if (bpe_encoder_new_with_special_tokens(&rejected, 259) != BPE_INVALID_INPUT ||
        bpe_encoder_new_with_special_tokens(&encoder, 268) != BPE_SUCCESS ||
        !bpe_encoder_has_special_tokens(&encoder) || encoder.vocab_size != 260 ||
        bpe_encoder_first_learned_id(&encoder) != 260 ||
        bpe_encoder_special_token_id(&encoder, BPE_SPECIAL_PAD) != BPE_PAD_TOKEN_ID ||
        bpe_encoder_special_token_id(&encoder, BPE_SPECIAL_UNK) != BPE_UNK_TOKEN_ID ||
        bpe_encoder_special_token_id(&encoder, BPE_SPECIAL_BOS) != BPE_BOS_TOKEN_ID ||
        bpe_encoder_special_token_id(&encoder, BPE_SPECIAL_EOS) != BPE_EOS_TOKEN_ID ||
        bpe_train(&encoder, text, strlen(text)) != BPE_SUCCESS ||
        bpe_encoder_freeze(&encoder) != BPE_SUCCESS ||
        bpe_encode(&encoder, text, strlen(text), &encoded) != BPE_SUCCESS) {
        fprintf(stderr, "special tokenizer setup failed\n");
        failed = 1;
        goto cleanup;
    }
    for (size_t i = 0; i < encoded.token_count; i++) {
        if (bpe_token_is_control(&encoder, encoded.token_ids[i])) {
            fprintf(stderr, "raw text encoding emitted a control token\n");
            failed = 1;
            goto cleanup;
        }
    }

    size_t wrapped_count = encoded.token_count + 4;
    uint32_t *wrapped = malloc(wrapped_count * sizeof(*wrapped));
    if (!wrapped) { failed = 1; goto cleanup; }
    wrapped[0] = BPE_BOS_TOKEN_ID;
    memcpy(wrapped + 1, encoded.token_ids,
           encoded.token_count * sizeof(*wrapped));
    wrapped[encoded.token_count + 1] = BPE_EOS_TOKEN_ID;
    wrapped[encoded.token_count + 2] = BPE_PAD_TOKEN_ID;
    wrapped[encoded.token_count + 3] = BPE_UNK_TOKEN_ID;
    if (bpe_decode(&encoder, wrapped, wrapped_count, &decoded, &decoded_size) !=
            BPE_SUCCESS || decoded_size != strlen(text) || strcmp(decoded, text) != 0) {
        fprintf(stderr, "control tokens leaked into decoded text\n");
        free(wrapped);
        failed = 1;
        goto cleanup;
    }
    free(wrapped);

    if (bpe_encoder_save(&encoder, path) != BPE_SUCCESS ||
        bpe_encoder_load(&sidecar, path) != BPE_SUCCESS ||
        !bpe_encoder_has_special_tokens(&sidecar) ||
        sidecar.vocab_size != encoder.vocab_size ||
        bpe_encode(&sidecar, text, strlen(text), &sidecar_tokens) != BPE_SUCCESS ||
        sidecar_tokens.token_count != encoded.token_count ||
        memcmp(sidecar_tokens.token_ids, encoded.token_ids,
               encoded.token_count * sizeof(*encoded.token_ids)) != 0 ||
        bpe_encoder_serialize_portable(&encoder, &portable_data,
                                       &portable_size) != BPE_SUCCESS ||
        bpe_encoder_deserialize_portable(&portable, portable_data,
                                         portable_size) != BPE_SUCCESS ||
        !bpe_encoder_has_special_tokens(&portable) ||
        portable.vocab_size != encoder.vocab_size) {
        fprintf(stderr, "special-token persistence changed the contract\n");
        failed = 1;
        goto cleanup;
    }

    uint32_t prompt[] = {10, 11, 12, 13, 14};
    uint32_t prepared[4] = {0};
    size_t prepared_count = 0;
    if (generation_prepare_prompt(&encoder, prompt, 5, prepared, 4,
                                  &prepared_count) != 0 ||
        prepared_count != 4 || prepared[0] != BPE_BOS_TOKEN_ID ||
        prepared[1] != 12 || prepared[2] != 13 || prepared[3] != 14) {
        fprintf(stderr, "BOS prompt/truncation policy changed\n");
        failed = 1;
        goto cleanup;
    }

    float logits[268] = {0};
    logits[BPE_PAD_TOKEN_ID] = 100.0f;
    logits[BPE_UNK_TOKEN_ID] = 90.0f;
    logits[BPE_BOS_TOKEN_ID] = 80.0f;
    logits[BPE_EOS_TOKEN_ID] = 70.0f;
    generation_mask_control_logits(&encoder, logits, 268);
    uint32_t selected = sample_greedy(logits, 268);
    if (logits[BPE_PAD_TOKEN_ID] != -FLT_MAX ||
        logits[BPE_UNK_TOKEN_ID] != -FLT_MAX ||
        logits[BPE_BOS_TOKEN_ID] != -FLT_MAX ||
        selected != BPE_EOS_TOKEN_ID ||
        !generation_token_is_eos(&encoder, selected)) {
        fprintf(stderr, "generation did not mask structural controls or recognize EOS\n");
        failed = 1;
        goto cleanup;
    }

    neural_model_t eos_model = {0};
    uint32_t eos_sequence[8] = {0};
    size_t eos_prompt_count = 0;
    generation_result_t eos_result = {0};
    if (model_new(&eos_model, 268, 8, 2, 1, 8) != MODEL_SUCCESS) {
        failed = 1;
        goto cleanup;
    }
    memset(eos_model.params, 0,
           eos_model.total_param_count * sizeof(*eos_model.params));
    eos_model.output_bias[BPE_EOS_TOKEN_ID] = 100.0f;
    if (generation_prepare_prompt(&encoder, prompt, 2, eos_sequence, 8,
                                  &eos_prompt_count) != 0 ||
        generation_decode(&eos_model, &encoder, eos_sequence, eos_prompt_count,
                          5, SAMPLING_GREEDY, 1.0f, 5, 0.9f,
                          &eos_result) != GENERATION_SUCCESS ||
        eos_result.new_count != 1 || !eos_result.stopped_on_eos ||
        eos_result.total_count != eos_prompt_count + 1 ||
        eos_sequence[eos_result.total_count - 1] != BPE_EOS_TOKEN_ID) {
        fprintf(stderr, "EOS-biased model did not stop after one generated token\n");
        failed = 1;
    }
    model_free(&eos_model);
    if (failed) goto cleanup;

    if (bpe_encoder_new(&legacy, 264) != BPE_SUCCESS ||
        bpe_encoder_has_special_tokens(&legacy) ||
        bpe_encoder_first_learned_id(&legacy) != 256 ||
        bpe_encoder_special_token_id(&legacy, BPE_SPECIAL_EOS) != BPE_NO_TOKEN_ID ||
        bpe_encoder_freeze(&legacy) != BPE_SUCCESS ||
        bpe_encoder_serialize_portable(&legacy, &legacy_portable_data,
                                       &legacy_portable_size) != BPE_SUCCESS ||
        bpe_encoder_deserialize_portable(&legacy_portable,
                                         legacy_portable_data,
                                         legacy_portable_size) != BPE_SUCCESS ||
        bpe_encoder_has_special_tokens(&legacy_portable) ||
        legacy_portable.vocab_size != 256 ||
        bpe_encoder_first_learned_id(&legacy_portable) != 256) {
        fprintf(stderr, "legacy tokenizer IDs were reinterpreted\n");
        failed = 1;
    }
    if (failed) goto cleanup;

    const uint16_t endian_probe = 1;
    if (sizeof(uint64_t) == 8 && *(const uint8_t *)&endian_probe == 1) {
        static const uint8_t legacy_v1_fixture[24] = {
            'D', 'R', 'N', 'Z', 'B', 'P', 'E', '1',
            [9] = 1,  /* max_vocab_size = 256 */
            [17] = 1, /* vocab_size = 256 */
        };
        FILE *fixture_file = fopen(path, "wb");
        int fixture_ok = 0;
        if (fixture_file) {
            fixture_ok = fwrite(legacy_v1_fixture, 1,
                                sizeof(legacy_v1_fixture), fixture_file) ==
                         sizeof(legacy_v1_fixture);
            if (fclose(fixture_file) != 0) fixture_ok = 0;
            fixture_file = NULL;
        }
        if (!fixture_ok) {
            if (fixture_file) fclose(fixture_file);
            failed = 1;
            goto cleanup;
        }
        bpe_encoder_t legacy_fixture = {0};
        if (bpe_encoder_load(&legacy_fixture, path) != BPE_SUCCESS ||
            bpe_encoder_has_special_tokens(&legacy_fixture) ||
            legacy_fixture.vocab_size != 256 ||
            legacy_fixture.tokens[255].id != 255) {
            fprintf(stderr, "retained v1 tokenizer fixture was reinterpreted\n");
            failed = 1;
        }
        bpe_encoder_free(&legacy_fixture);
    }

cleanup:
    free(decoded);
    free(portable_data);
    free(legacy_portable_data);
    bpe_tokens_free(&encoded);
    bpe_tokens_free(&sidecar_tokens);
    bpe_encoder_free(&encoder);
    bpe_encoder_free(&sidecar);
    bpe_encoder_free(&portable);
    bpe_encoder_free(&legacy);
    bpe_encoder_free(&legacy_portable);
    remove(path);
    printf("\n%s\n", failed ? "SPECIAL TOKEN CONTRACT CHECK FAILED"
                              : "SPECIAL TOKEN CONTRACT CHECK PASSED");
    return failed ? 1 : 0;
}
