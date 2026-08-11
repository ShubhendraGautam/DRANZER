/* Public embedding surface: this test uses opaque handles for every runtime
 * object and only uses internal headers to create its temporary fixture. */

#include "dranzer.h"
#include "byte_pair_encoding.h"
#include "core/bundle.h"
#include "core/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VOCAB 264
#define EMBEDDING 8
#define HEADS 2
#define LAYERS 1
#define MAX_SEQUENCE 8

static int close_logits(const float *left, const float *right, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (fabsf(left[i] - right[i]) > 1e-5f) return 0;
    }
    return 1;
}

int main(void) {
    char path[160];
    snprintf(path, sizeof(path), "/tmp/dranzer-public-api-%ld.bin",
             (long)getpid());
    neural_model_t fixture_model = {0};
    bpe_encoder_t fixture_tokenizer = {0};
    int failed = 0;
    const uint8_t corpus[] = {'a','p','i',0,'b','y','t','e','s',' ','a','p','i'};
    if (model_new_seeded(&fixture_model, VOCAB, EMBEDDING, HEADS, LAYERS,
                         MAX_SEQUENCE, 44) != MODEL_SUCCESS ||
        bpe_encoder_new_with_special_tokens(&fixture_tokenizer, VOCAB) !=
            BPE_SUCCESS ||
        bpe_train(&fixture_tokenizer, (const char *)corpus,
                  sizeof(corpus)) != BPE_SUCCESS ||
        bpe_encoder_freeze(&fixture_tokenizer) != BPE_SUCCESS) {
        fprintf(stderr, "public API fixture setup failed\n");
        failed = 1;
        goto cleanup_fixture;
    }
    model_bundle_metadata_t metadata = {
        .train_window = 4, .seed = 44,
        .input_fingerprint = UINT64_C(123), .input_bytes = sizeof(corpus),
    };
    if (model_bundle_save(&fixture_model, &fixture_tokenizer, &metadata,
                          path) != BUNDLE_SUCCESS) {
        failed = 1;
        goto cleanup_fixture;
    }

    dranzer_model_t *model = NULL;
    dranzer_tokenizer_t *tokenizer = NULL;
    dranzer_bundle_info_t info = {0};
    if (dranzer_bundle_load(path, DRANZER_LOAD_COPY, &model, &tokenizer,
                            &info) != DRANZER_OK ||
        !model || !tokenizer || dranzer_model_vocab_size(model) != VOCAB ||
        dranzer_model_max_sequence(model) != MAX_SEQUENCE ||
        info.seed != metadata.seed ||
        strcmp(dranzer_status_string(DRANZER_CHECKSUM_ERROR),
               "checksum mismatch") != 0) {
        fprintf(stderr, "opaque bundle load or metadata failed\n");
        failed = 1;
        goto cleanup_public;
    }

    dranzer_model_t *mapped_model = NULL;
    dranzer_tokenizer_t *mapped_tokenizer = NULL;
    if (dranzer_bundle_load(path, DRANZER_LOAD_MMAP, &mapped_model,
                            &mapped_tokenizer, NULL) != DRANZER_OK) {
        fprintf(stderr, "public mmap load failed\n");
        failed = 1;
    }
    dranzer_model_free(mapped_model);
    dranzer_tokenizer_free(mapped_tokenizer);

    const uint8_t text[] = {'a','p','i',0,'b','y','t','e','s'};
    size_t token_count = 0;
    if (dranzer_tokenize(tokenizer, text, sizeof(text), NULL,
                         &token_count) != DRANZER_BUFFER_TOO_SMALL ||
        token_count == 0) {
        fprintf(stderr, "tokenizer size query failed\n");
        failed = 1;
        goto cleanup_public;
    }
    uint32_t *tokens = malloc(token_count * sizeof(*tokens));
    size_t token_capacity = token_count;
    if (!tokens || dranzer_tokenize(tokenizer, text, sizeof(text), tokens,
                                    &token_capacity) != DRANZER_OK ||
        token_capacity != token_count) {
        free(tokens);
        failed = 1;
        goto cleanup_public;
    }
    size_t decoded_size = 0;
    if (dranzer_detokenize(tokenizer, tokens, token_count, NULL,
                           &decoded_size) != DRANZER_BUFFER_TOO_SMALL ||
        decoded_size != sizeof(text)) {
        fprintf(stderr, "detokenizer size query failed\n");
        free(tokens);
        failed = 1;
        goto cleanup_public;
    }
    uint8_t *decoded = malloc(decoded_size);
    size_t decoded_capacity = decoded_size;
    if (!decoded || dranzer_detokenize(tokenizer, tokens, token_count, decoded,
                                       &decoded_capacity) != DRANZER_OK ||
        decoded_capacity != sizeof(text) ||
        memcmp(decoded, text, sizeof(text)) != 0) {
        fprintf(stderr, "binary-safe public tokenizer roundtrip failed\n");
        failed = 1;
    }
    free(decoded);

    size_t prefix_count = token_count < 3 ? token_count : 3;
    float full_logits[VOCAB], cached_logits[VOCAB];
    dranzer_cache_t *cache = NULL;
    if (dranzer_model_forward(model, tokens, prefix_count, full_logits,
                              VOCAB) != DRANZER_OK ||
        dranzer_cache_create(model, MAX_SEQUENCE, &cache) != DRANZER_OK) {
        fprintf(stderr, "public forward/cache creation failed\n");
        free(tokens);
        failed = 1;
        goto cleanup_public;
    }
    for (size_t i = 0; i < prefix_count; i++) {
        if (dranzer_cache_forward(cache, tokens[i], cached_logits, VOCAB) !=
            DRANZER_OK) failed = 1;
    }
    if (!close_logits(full_logits, cached_logits, VOCAB)) {
        fprintf(stderr, "public cached/full logits differ\n");
        failed = 1;
    }

    dranzer_generation_t *generation = NULL;
    if (dranzer_generation_create(model, tokenizer, MAX_SEQUENCE,
                                  &generation) != DRANZER_OK) {
        fprintf(stderr, "public generation creation failed\n");
        dranzer_cache_free(cache);
        free(tokens);
        failed = 1;
        goto cleanup_public;
    }

    /* Cache and generation retain their dependencies. */
    dranzer_model_free(model);
    model = NULL;
    dranzer_tokenizer_free(tokenizer);
    tokenizer = NULL;
    dranzer_cache_reset(cache);
    if (dranzer_cache_forward(cache, tokens[0], cached_logits, VOCAB) !=
            DRANZER_OK ||
        dranzer_generation_reset(generation, text, sizeof(text)) !=
            DRANZER_OK) {
        fprintf(stderr, "retained handle dependency failed\n");
        failed = 1;
    }
    uint8_t piece[64];
    size_t piece_size = sizeof(piece);
    uint32_t generated_token = 0;
    dranzer_status_t next_rc = dranzer_generation_next_greedy(
        generation, &generated_token, piece, &piece_size);
    if ((next_rc != DRANZER_OK && next_rc != DRANZER_FINISHED) ||
        generated_token >= VOCAB || piece_size > sizeof(piece)) {
        fprintf(stderr, "greedy generation step failed\n");
        failed = 1;
    }
    dranzer_generation_free(generation);
    dranzer_cache_free(cache);
    free(tokens);

cleanup_public:
    dranzer_model_free(model);
    dranzer_tokenizer_free(tokenizer);
cleanup_fixture:
    bpe_encoder_free(&fixture_tokenizer);
    model_free(&fixture_model);
    remove(path);
    printf("%s\n", failed ? "PUBLIC API CHECK FAILED"
                           : "PUBLIC API CHECK PASSED");
    return failed ? 1 : 0;
}
