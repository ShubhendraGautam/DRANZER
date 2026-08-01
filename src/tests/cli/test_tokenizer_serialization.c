#include "byte_pair_encoding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TMP_PATH "/tmp/dranzer_tokenizer_roundtrip.bin"

int main(void) {
    const char *training_text = "banana bandana banana bandana";
    const char *probe_text = "banana bandana";
    bpe_encoder_t original = {0};
    bpe_encoder_t loaded = {0};
    bpe_tokens_t before = {0};
    bpe_tokens_t after = {0};
    char *decoded = NULL;
    size_t decoded_len = 0;
    int failed = 0;

    if (bpe_encoder_new(&original, 280) != BPE_SUCCESS ||
        bpe_train(&original, training_text, strlen(training_text)) != BPE_SUCCESS ||
        bpe_encode(&original, probe_text, strlen(probe_text), &before) != BPE_SUCCESS ||
        bpe_encoder_save(&original, TMP_PATH) != BPE_SUCCESS ||
        bpe_encoder_load(&loaded, TMP_PATH) != BPE_SUCCESS ||
        bpe_encode(&loaded, probe_text, strlen(probe_text), &after) != BPE_SUCCESS) {
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

    for (size_t i = 0; i < original.vocab_size; i++) {
        if (original.tokens[i].id != loaded.tokens[i].id ||
            original.tokens[i].frequency != loaded.tokens[i].frequency ||
            strcmp(original.tokens[i].token, loaded.tokens[i].token) != 0) {
            fprintf(stderr, "token %zu changed across roundtrip\n", i);
            failed = 1;
            goto cleanup;
        }
    }
    if (memcmp(before.token_ids, after.token_ids,
               before.token_count * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "encoded IDs changed across roundtrip\n");
        failed = 1;
        goto cleanup;
    }

    if (bpe_decode(&loaded, after.token_ids, after.token_count,
                   &decoded, &decoded_len) != BPE_SUCCESS ||
        decoded_len != strlen(probe_text) || strcmp(decoded, probe_text) != 0) {
        fprintf(stderr, "loaded tokenizer did not decode the original text\n");
        failed = 1;
        goto cleanup;
    }

    printf("vocabulary=%zu encoded_tokens=%zu\n", loaded.vocab_size, after.token_count);
    printf("\nTOKENIZER SERIALIZATION CHECK PASSED\n");

cleanup:
    free(decoded);
    bpe_tokens_free(&before);
    bpe_tokens_free(&after);
    bpe_encoder_free(&original);
    bpe_encoder_free(&loaded);
    remove(TMP_PATH);
    return failed;
}
