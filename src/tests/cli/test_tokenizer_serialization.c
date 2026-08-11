#include "byte_pair_encoding.h"
#include "cli/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TMP_PATH "/tmp/dranzer_tokenizer_roundtrip.bin"
#define TMP_CORPUS_PATH "/tmp/dranzer_tokenizer_binary_input.bin"

int main(void) {
    static const char training_text[] = {
        'A', '\0', 'B', 'A', '\0', 'B', 'A', '\0', 'B', 'A', '\0', 'B'
    };
    static const char probe_text[] = {'A', '\0', 'B', 'A', '\0', 'B'};
    bpe_encoder_t original = {0};
    bpe_encoder_t loaded = {0};
    bpe_encoder_t portable = {0};
    bpe_tokens_t before = {0};
    bpe_tokens_t after = {0};
    bpe_tokens_t from_file = {0};
    bpe_tokens_t from_portable = {0};
    uint8_t *portable_data = NULL;
    size_t portable_size = 0;
    char *decoded = NULL;
    size_t decoded_len = 0;
    int failed = 0;

    if (bpe_encoder_new(&original, 280) != BPE_SUCCESS ||
        bpe_train(&original, training_text, sizeof(training_text)) != BPE_SUCCESS ||
        bpe_encode(&original, probe_text, sizeof(probe_text), &before) != BPE_SUCCESS ||
        bpe_encoder_freeze(&original) != BPE_SUCCESS ||
        bpe_encoder_save(&original, TMP_PATH) != BPE_SUCCESS ||
        bpe_encoder_load(&loaded, TMP_PATH) != BPE_SUCCESS ||
        bpe_encode(&loaded, probe_text, sizeof(probe_text), &after) != BPE_SUCCESS) {
        fprintf(stderr, "tokenizer save/load setup failed\n");
        failed = 1;
        goto cleanup;
    }

    if (original.max_vocab_size != loaded.max_vocab_size ||
        original.vocab_size != loaded.vocab_size ||
        before.token_count != after.token_count) {
        fprintf(stderr, "tokenizer metadata changed across roundtrip\n");
        failed = 1;
        goto cleanup;
    }

    int saw_binary_merge = 0;
    for (size_t i = 0; i < original.vocab_size; i++) {
        if (original.tokens[i].id != loaded.tokens[i].id ||
            original.tokens[i].frequency != loaded.tokens[i].frequency ||
            original.tokens[i].length != loaded.tokens[i].length ||
            (original.tokens[i].length > 0 &&
             memcmp(original.tokens[i].token, loaded.tokens[i].token,
                    original.tokens[i].length) != 0)) {
            fprintf(stderr, "token %zu changed across roundtrip\n", i);
            failed = 1;
            goto cleanup;
        }
        if (i >= BPE_BYTE_TOKEN_COUNT && original.tokens[i].token &&
            memchr(original.tokens[i].token, '\0', original.tokens[i].length)) {
            saw_binary_merge = 1;
        }
    }
    if (!saw_binary_merge) {
        fprintf(stderr, "fixture did not create a learned token containing NUL\n");
        failed = 1;
        goto cleanup;
    }
    if (memcmp(before.token_ids, after.token_ids,
               before.token_count * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "encoded IDs changed across roundtrip\n");
        failed = 1;
        goto cleanup;
    }

    if (bpe_decode(&loaded, after.token_ids, after.token_count,
                   &decoded, &decoded_len) != BPE_SUCCESS ||
        decoded_len != sizeof(probe_text) ||
        memcmp(decoded, probe_text, sizeof(probe_text)) != 0) {
        fprintf(stderr, "loaded tokenizer did not decode the original text\n");
        failed = 1;
        goto cleanup;
    }

    FILE *corpus = fopen(TMP_CORPUS_PATH, "wb");
    int corpus_written = corpus &&
                         fwrite(probe_text, 1, sizeof(probe_text), corpus) ==
                             sizeof(probe_text);
    if (corpus && fclose(corpus) != 0) corpus_written = 0;
    if (!corpus_written ||
        tokenizer_tokenize_file(&loaded, TMP_CORPUS_PATH, &from_file) !=
            TOKENIZER_SUCCESS ||
        from_file.token_count != after.token_count ||
        memcmp(from_file.token_ids, after.token_ids,
               after.token_count * sizeof(*after.token_ids)) != 0) {
        fprintf(stderr, "file tokenization truncated an embedded NUL\n");
        failed = 1;
        goto cleanup;
    }

    char magic[8] = {0};
    FILE *saved = fopen(TMP_PATH, "rb");
    if (!saved || fread(magic, 1, sizeof(magic), saved) != sizeof(magic) ||
        memcmp(magic, "DRNZBPE3", sizeof(magic)) != 0) {
        fprintf(stderr, "tokenizer sidecar did not carry binary format version 3\n");
        if (saved) fclose(saved);
        failed = 1;
        goto cleanup;
    }
    fclose(saved);

    if (bpe_encoder_serialize_portable(&original, &portable_data,
                                       &portable_size) != BPE_SUCCESS ||
        portable_size < 8 || memcmp(portable_data, "DRNZBPP2", 8) != 0 ||
        bpe_encoder_deserialize_portable(&portable, portable_data,
                                         portable_size) != BPE_SUCCESS ||
        bpe_encode(&portable, probe_text, sizeof(probe_text),
                   &from_portable) != BPE_SUCCESS ||
        from_portable.token_count != before.token_count ||
        memcmp(from_portable.token_ids, before.token_ids,
               before.token_count * sizeof(*before.token_ids)) != 0) {
        fprintf(stderr, "portable format changed a learned token containing NUL\n");
        failed = 1;
        goto cleanup;
    }

    printf("vocabulary=%zu encoded_tokens=%zu\n", loaded.vocab_size, after.token_count);
    printf("\nTOKENIZER SERIALIZATION CHECK PASSED\n");

cleanup:
    free(decoded);
    free(portable_data);
    bpe_tokens_free(&before);
    bpe_tokens_free(&after);
    bpe_tokens_free(&from_file);
    bpe_tokens_free(&from_portable);
    bpe_encoder_free(&original);
    bpe_encoder_free(&loaded);
    bpe_encoder_free(&portable);
    remove(TMP_PATH);
    remove(TMP_CORPUS_PATH);
    return failed;
}
