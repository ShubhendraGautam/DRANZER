#include "cli/tokenizer.h"
#include "byte_pair_encoding.h"
#include "common/debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILE_SIZE (10 * 1024 * 1024) // 10MB max file size
#define TOKENIZER_TRAIN_CHUNK_SIZE (256 * 1024)
#define FNV1A_OFFSET UINT64_C(14695981039346656037)
#define FNV1A_PRIME UINT64_C(1099511628211)

static tokenizer_errors_t scan_corpus(const char *filename, bpe_encoder_t *encoder,
                                      tokenizer_corpus_stats_t *out_stats) {
    if (!filename || !out_stats) return TOKENIZER_INVALID_INPUT;

    FILE *file = fopen(filename, "rb");
    if (!file) return TOKENIZER_FILE_NOT_FOUND;

    unsigned char *buffer = malloc(TOKENIZER_TRAIN_CHUNK_SIZE);
    if (!buffer) {
        fclose(file);
        return TOKENIZER_ALLOCATION_FAILURE;
    }

    tokenizer_corpus_stats_t stats = {FNV1A_OFFSET, 0, 0};
    tokenizer_errors_t rc = TOKENIZER_SUCCESS;
    size_t bytes_read = 0;
    while ((bytes_read = fread(buffer, 1, TOKENIZER_TRAIN_CHUNK_SIZE, file)) > 0) {
        for (size_t i = 0; i < bytes_read; i++) {
            stats.fingerprint ^= (uint64_t)buffer[i];
            stats.fingerprint *= FNV1A_PRIME;
        }
        stats.byte_count += bytes_read;
        stats.chunk_count++;

        if (encoder && encoder->vocab_size < encoder->max_vocab_size) {
            bpe_errors_t train_rc = bpe_train(encoder, (const char *)buffer, bytes_read);
            if (train_rc != BPE_SUCCESS) {
                rc = train_rc == BPE_ALLOCATION_FAILURE
                         ? TOKENIZER_ALLOCATION_FAILURE
                         : TOKENIZER_ENCODING_ERROR;
                break;
            }
        }
    }

    if (rc == TOKENIZER_SUCCESS && ferror(file)) rc = TOKENIZER_IO_ERROR;
    if (rc == TOKENIZER_SUCCESS && stats.byte_count == 0) rc = TOKENIZER_INVALID_INPUT;

    free(buffer);
    fclose(file);
    if (rc == TOKENIZER_SUCCESS) *out_stats = stats;
    return rc;
}

bpe_encoder_t* tokenizer_create_encoder(size_t vocab_size) {
    bpe_encoder_t *encoder = malloc(sizeof(bpe_encoder_t));
    if (encoder == NULL) {
        return NULL;
    }

    if (bpe_encoder_new(encoder, vocab_size) != BPE_SUCCESS) {
        free(encoder);
        return NULL;
    }

    return encoder;
}

bpe_encoder_t* tokenizer_create_special_encoder(size_t vocab_size) {
    bpe_encoder_t *encoder = malloc(sizeof(*encoder));
    if (!encoder) return NULL;
    if (bpe_encoder_new_with_special_tokens(encoder, vocab_size) != BPE_SUCCESS) {
        free(encoder);
        return NULL;
    }
    return encoder;
}

void tokenizer_free_encoder(bpe_encoder_t *encoder) {
    if (!encoder) return;
    bpe_encoder_free(encoder);
    free(encoder);
}

tokenizer_errors_t tokenizer_save_encoder(const bpe_encoder_t *encoder, const char *filename) {
    if (!encoder || !filename) return TOKENIZER_INVALID_INPUT;
    bpe_errors_t rc = bpe_encoder_save(encoder, filename);
    if (rc == BPE_SUCCESS) return TOKENIZER_SUCCESS;
    if (rc == BPE_IO_ERROR) return TOKENIZER_IO_ERROR;
    if (rc == BPE_FORMAT_ERROR) return TOKENIZER_FORMAT_ERROR;
    return rc == BPE_ALLOCATION_FAILURE ? TOKENIZER_ALLOCATION_FAILURE : TOKENIZER_INVALID_INPUT;
}

tokenizer_errors_t tokenizer_load_encoder(const char *filename, bpe_encoder_t **out_encoder) {
    if (!filename || !out_encoder) return TOKENIZER_INVALID_INPUT;
    *out_encoder = NULL;

    bpe_encoder_t *encoder = malloc(sizeof(*encoder));
    if (!encoder) return TOKENIZER_ALLOCATION_FAILURE;

    bpe_errors_t rc = bpe_encoder_load(encoder, filename);
    if (rc != BPE_SUCCESS) {
        free(encoder);
        if (rc == BPE_IO_ERROR) return TOKENIZER_FILE_NOT_FOUND;
        if (rc == BPE_FORMAT_ERROR) return TOKENIZER_FORMAT_ERROR;
        return rc == BPE_ALLOCATION_FAILURE ? TOKENIZER_ALLOCATION_FAILURE : TOKENIZER_INVALID_INPUT;
    }

    *out_encoder = encoder;
    return TOKENIZER_SUCCESS;
}

tokenizer_errors_t tokenizer_default_path(const char *model_path, char *output, size_t output_size) {
    if (!model_path || !output || output_size == 0) return TOKENIZER_INVALID_INPUT;
    int written = snprintf(output, output_size, "%s.tokenizer", model_path);
    if (written < 0 || (size_t)written >= output_size) return TOKENIZER_INVALID_INPUT;
    return TOKENIZER_SUCCESS;
}

tokenizer_errors_t tokenizer_train_encoder_file(bpe_encoder_t *encoder, const char *filename,
                                                 tokenizer_corpus_stats_t *out_stats) {
    if (!encoder || bpe_encoder_is_frozen(encoder)) return TOKENIZER_INVALID_INPUT;
    tokenizer_errors_t rc = scan_corpus(filename, encoder, out_stats);
    if (rc != TOKENIZER_SUCCESS) return rc;
    return bpe_encoder_freeze(encoder) == BPE_SUCCESS
               ? TOKENIZER_SUCCESS
               : TOKENIZER_ENCODING_ERROR;
}

tokenizer_errors_t tokenizer_fingerprint_file(const char *filename,
                                               tokenizer_corpus_stats_t *out_stats) {
    return scan_corpus(filename, NULL, out_stats);
}

tokenizer_errors_t tokenizer_tokenize(bpe_encoder_t *encoder, const char *text, bpe_tokens_t *output) {
    if (encoder == NULL || text == NULL || output == NULL) {
        return TOKENIZER_INVALID_INPUT;
    }

    if (bpe_encode(encoder, text, strlen(text), output) != BPE_SUCCESS) {
        return TOKENIZER_ENCODING_ERROR;
    }

    return TOKENIZER_SUCCESS;
}

tokenizer_errors_t tokenizer_tokenize_file(bpe_encoder_t *encoder, const char *filename, bpe_tokens_t *output) {
    DEBUG_PRINT("Tokenizing file: %s\n", filename);
    if (encoder == NULL || filename == NULL || output == NULL) {
        return TOKENIZER_INVALID_INPUT;
    }

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        DEBUG_PRINT("Could not open file: %s\n", filename);
        return TOKENIZER_FILE_NOT_FOUND;
    }

    // Read entire file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size <= 0 || file_size > MAX_FILE_SIZE) {
        fclose(file);
        return TOKENIZER_INVALID_INPUT;
    }

    char *buffer = malloc(file_size + 1);
    if (buffer == NULL) {
        fclose(file);
        return TOKENIZER_ALLOCATION_FAILURE;
    }

    size_t read_size = fread(buffer, 1, file_size, file);
    fclose(file);

    if (read_size != (size_t)file_size) {
        free(buffer);
        return TOKENIZER_ENCODING_ERROR;
    }

    buffer[file_size] = '\0';

    // Tokenize the exact bytes read; buffer may contain embedded NULs.
    DEBUG_PRINT("File read successfully. Size: %ld bytes. Tokenizing...\n", file_size);
    tokenizer_errors_t result =
        bpe_encode(encoder, buffer, read_size, output) == BPE_SUCCESS
            ? TOKENIZER_SUCCESS : TOKENIZER_ENCODING_ERROR;
    if (result == TOKENIZER_SUCCESS) {
        DEBUG_PRINT("Tokenization complete. Token count: %zu\n", output->token_count);
    }
    free(buffer);

    return result;
}

tokenizer_errors_t tokenizer_detokenize(bpe_encoder_t *encoder, const uint32_t *token_ids, size_t token_count, char **output) {
    if (encoder == NULL || token_ids == NULL || output == NULL) {
        return TOKENIZER_INVALID_INPUT;
    }

    size_t output_len;
    if (bpe_decode(encoder, token_ids, token_count, output, &output_len) != BPE_SUCCESS) {
        return TOKENIZER_ENCODING_ERROR;
    }

    return TOKENIZER_SUCCESS;
}
