#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "byte_pair_encoding.h"
#include <stdio.h>

typedef enum {
    TOKENIZER_SUCCESS = 0,
    TOKENIZER_FILE_NOT_FOUND,
    TOKENIZER_ALLOCATION_FAILURE,
    TOKENIZER_INVALID_INPUT,
    TOKENIZER_ENCODING_ERROR,
    TOKENIZER_IO_ERROR,
    TOKENIZER_FORMAT_ERROR,
} tokenizer_errors_t;

typedef struct {
    bpe_encoder_t *encoder;
    uint32_t *token_ids;
    size_t token_count;
} tokenized_text_t;

/**
 * Creates a new tokenizer from a BPE encoder.
 * @param encoder: Pointer to a trained BPE encoder
 * @return Allocated encoder, or NULL on failure
 */
bpe_encoder_t* tokenizer_create_encoder(size_t vocab_size);

/** Free an encoder returned by tokenizer_create_encoder/load_encoder. */
void tokenizer_free_encoder(bpe_encoder_t *encoder);

/** Save/load the trained BPE vocabulary used alongside a model file. */
tokenizer_errors_t tokenizer_save_encoder(const bpe_encoder_t *encoder, const char *filename);
tokenizer_errors_t tokenizer_load_encoder(const char *filename, bpe_encoder_t **out_encoder);

/** Derive the default sidecar path: "<model path>.tokenizer". */
tokenizer_errors_t tokenizer_default_path(const char *model_path, char *output, size_t output_size);

/**
 * Tokenizes a text string using BPE.
 * @param encoder: Pointer to the BPE encoder
 * @param text: Input text
 * @param output: Pointer to store token IDs (allocated by function)
 * @return TOKENIZER_SUCCESS on success, error code otherwise
 */
tokenizer_errors_t tokenizer_tokenize(bpe_encoder_t *encoder, const char *text, bpe_tokens_t *output);

/**
 * Tokenizes text from a file.
 * @param encoder: Pointer to the BPE encoder
 * @param filename: Path to the text file
 * @param output: Pointer to store token IDs (allocated by function)
 * @return TOKENIZER_SUCCESS on success, error code otherwise
 */
tokenizer_errors_t tokenizer_tokenize_file(bpe_encoder_t *encoder, const char *filename, bpe_tokens_t *output);

/**
 * Detokenizes token IDs back to text.
 * @param encoder: Pointer to the BPE encoder
 * @param token_ids: Array of token IDs
 * @param token_count: Number of tokens
 * @param output: Pointer to store decoded text
 * @return TOKENIZER_SUCCESS on success, error code otherwise
 */
tokenizer_errors_t tokenizer_detokenize(bpe_encoder_t *encoder, const uint32_t *token_ids, size_t token_count, char **output);

#endif /* TOKENIZER_H */
