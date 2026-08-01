#ifndef BYTE_PAIR_ENCODING_H
#define BYTE_PAIR_ENCODING_H

#include <stddef.h>
#include <stdint.h>
#include "hashmap.h"

typedef enum {
    BPE_SUCCESS = 0,
    BPE_ALLOCATION_FAILURE,
    BPE_INVALID_INPUT,
    BPE_NULL_ENCODER,
    BPE_ENCODING_ERROR,
    BPE_IO_ERROR,
    BPE_FORMAT_ERROR,
} bpe_errors_t;

typedef struct {
    char *token;           // The token string (could be multi-char like "ab" or "ing")
    uint32_t id;           // Token ID
    int frequency;         // Frequency of this token in the current vocabulary
} bpe_token_t;

typedef struct {
    bpe_token_t *tokens;   // Array of tokens
    size_t vocab_size;     // Current vocabulary size
    size_t max_vocab_size; // Maximum vocabulary size to build to
    hashmap_t token_to_id; // Map from token string to token ID
    hashmap_t id_to_token; // Map from token ID to token string
} bpe_encoder_t;

typedef struct {
    uint32_t *token_ids;   // Array of token IDs
    size_t token_count;    // Number of tokens
} bpe_tokens_t;

/* Function declarations */

/**
 * Creates and initializes a new BPE encoder.
 * @param encoder: Pointer to the encoder structure to initialize
 * @param max_vocab_size: Maximum vocabulary size to build (e.g., 1000, 10000)
 * @return BPE_SUCCESS on success, error code otherwise
 */
bpe_errors_t bpe_encoder_new(bpe_encoder_t *encoder, size_t max_vocab_size);

/**
 * Trains the BPE encoder on input text by building the vocabulary.
 * @param encoder: Pointer to the encoder
 * @param input: Input text to train on
 * @param input_len: Length of input text
 * @return BPE_SUCCESS on success, error code otherwise
 */
bpe_errors_t bpe_train(bpe_encoder_t *encoder, const char *input, size_t input_len);

/**
 * Encodes input text into token IDs using the trained BPE encoder.
 * @param encoder: Pointer to the encoder
 * @param input: Input text to encode
 * @param input_len: Length of input text
 * @param tokens: Pointer to store the resulting tokens (allocated by function)
 * @return BPE_SUCCESS on success, error code otherwise
 */
bpe_errors_t bpe_encode(bpe_encoder_t *encoder, const char *input, size_t input_len, bpe_tokens_t *tokens);

/**
 * Decodes token IDs back into text.
 * @param encoder: Pointer to the encoder
 * @param token_ids: Array of token IDs
 * @param token_count: Number of tokens
 * @param output: Pointer to buffer to store decoded text (allocated by function)
 * @param output_len: Pointer to store length of output
 * @return BPE_SUCCESS on success, error code otherwise
 */
bpe_errors_t bpe_decode(bpe_encoder_t *encoder, const uint32_t *token_ids, size_t token_count, char **output, size_t *output_len);

/**
 * Save a trained vocabulary, including merge order, to a binary sidecar.
 * The model weights deliberately remain a separate format so existing
 * .pth files stay readable.
 */
bpe_errors_t bpe_encoder_save(const bpe_encoder_t *encoder, const char *filename);

/**
 * Load a vocabulary saved by bpe_encoder_save into an uninitialized
 * encoder. The caller owns the result and must call bpe_encoder_free.
 */
bpe_errors_t bpe_encoder_load(bpe_encoder_t *encoder, const char *filename);

/**
 * Frees the BPE encoder and all associated resources.
 * @param encoder: Pointer to the encoder
 * @return BPE_SUCCESS on success, error code otherwise
 */
bpe_errors_t bpe_encoder_free(bpe_encoder_t *encoder);

/**
 * Frees the tokens structure.
 * @param tokens: Pointer to the tokens structure
 * @return BPE_SUCCESS on success, error code otherwise
 */
bpe_errors_t bpe_tokens_free(bpe_tokens_t *tokens);

#endif /* BYTE_PAIR_ENCODING_H */
