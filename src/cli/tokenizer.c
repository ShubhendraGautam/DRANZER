#include "cli/tokenizer.h"
#include "byte_pair_encoding.h"
#include "common/debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILE_SIZE (10 * 1024 * 1024) // 10MB max file size

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

    FILE *file = fopen(filename, "r");
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

    // Tokenize the buffer
    DEBUG_PRINT("File read successfully. Size: %ld bytes. Tokenizing...\n", file_size);
    tokenizer_errors_t result = tokenizer_tokenize(encoder, buffer, output);
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

