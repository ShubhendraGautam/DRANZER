#include "byte_pair_encoding.h"
#include "cli/tokenizer.h"
#include "core/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    const char *corpus = "banana bandana banana bandana\nbanana bandana\n";
    char path[128];
    snprintf(path, sizeof(path), "/tmp/dranzer_frozen_tokenizer_%ld.txt", (long)getpid());

    FILE *file = fopen(path, "wb");
    size_t written = file ? fwrite(corpus, 1, strlen(corpus), file) : 0;
    int close_rc = file ? fclose(file) : -1;
    if (!file || written != strlen(corpus) || close_rc != 0) {
        remove(path);
        fprintf(stderr, "failed to create tokenizer corpus fixture\n");
        return 1;
    }

    bpe_encoder_t *encoder = tokenizer_create_encoder(264);
    bpe_tokens_t before = {0};
    bpe_tokens_t after = {0};
    tokenizer_corpus_stats_t trained_stats = {0};
    tokenizer_corpus_stats_t rescanned_stats = {0};
    neural_model_t model = {0};
    int failed = 0;

    if (!encoder ||
        tokenizer_train_encoder_file(encoder, path, &trained_stats) != TOKENIZER_SUCCESS ||
        !bpe_encoder_is_frozen(encoder) ||
        trained_stats.byte_count != strlen(corpus) || trained_stats.chunk_count != 1 ||
        tokenizer_fingerprint_file(path, &rescanned_stats) != TOKENIZER_SUCCESS ||
        trained_stats.fingerprint != rescanned_stats.fingerprint ||
        bpe_encode(encoder, corpus, strlen(corpus), &before) != BPE_SUCCESS) {
        fprintf(stderr, "failed to prepare frozen tokenizer\n");
        failed = 1;
        goto cleanup;
    }

    size_t frozen_vocab_size = encoder->vocab_size;
    if (bpe_train(encoder, corpus, strlen(corpus)) != BPE_ENCODER_FROZEN ||
        encoder->vocab_size != frozen_vocab_size) {
        fprintf(stderr, "frozen tokenizer accepted additional training\n");
        failed = 1;
        goto cleanup;
    }

    srand(17);
    if (model_new(&model, encoder->max_vocab_size, 8, 2, 1, 8) != MODEL_SUCCESS) {
        fprintf(stderr, "failed to create model fixture\n");
        failed = 1;
        goto cleanup;
    }
    for (size_t i = 0; i + 1 < before.token_count && i < 12; i++) {
        size_t context_len = i + 1 < 8 ? i + 1 : 8;
        uint32_t *context = &before.token_ids[i + 1 - context_len];
        if (model_train_step(&model, context, before.token_ids[i + 1], context_len) !=
            MODEL_SUCCESS) {
            fprintf(stderr, "model training fixture failed\n");
            failed = 1;
            goto cleanup;
        }
    }

    if (bpe_encode(encoder, corpus, strlen(corpus), &after) != BPE_SUCCESS ||
        after.token_count != before.token_count || encoder->vocab_size != frozen_vocab_size ||
        memcmp(after.token_ids, before.token_ids,
               before.token_count * sizeof(*before.token_ids)) != 0) {
        fprintf(stderr, "token stream changed after model optimization\n");
        failed = 1;
        goto cleanup;
    }

    printf("fingerprint=%016llx bytes=%zu vocabulary=%zu tokens=%zu\n",
           (unsigned long long)trained_stats.fingerprint, trained_stats.byte_count,
           frozen_vocab_size, before.token_count);
    printf("\nFROZEN TOKENIZER TRAINING CHECK PASSED\n");

cleanup:
    model_free(&model);
    bpe_tokens_free(&before);
    bpe_tokens_free(&after);
    tokenizer_free_encoder(encoder);
    remove(path);
    return failed;
}
