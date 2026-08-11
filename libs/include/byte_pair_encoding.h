#ifndef BYTE_PAIR_ENCODING_H
#define BYTE_PAIR_ENCODING_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "hashmap.h"

typedef enum {
    BPE_SUCCESS = 0,
    BPE_ALLOCATION_FAILURE,
    BPE_INVALID_INPUT,
    BPE_NULL_ENCODER,
    BPE_ENCODING_ERROR,
    BPE_IO_ERROR,
    BPE_FORMAT_ERROR,
    BPE_ENCODER_FROZEN,
} bpe_errors_t;

typedef struct {
    char *token;           // Token bytes; sentinel-terminated for diagnostics only
    size_t length;         // Authoritative length; token bytes may contain NUL
    uint32_t id;           // Token ID
    int frequency;         // Frequency of this token in the current vocabulary
} bpe_token_t;

typedef struct {
    bpe_token_t *tokens;   // Array of tokens
    size_t vocab_size;     // Current vocabulary size
    size_t max_vocab_size; // Maximum vocabulary size to build to
    int is_frozen;         // Frozen vocabularies reject further training
    int has_special_tokens;// IDs 256..259 are reserved control tokens
    hashmap_t token_to_id; // Map from length-delimited token bytes to token ID
    hashmap_t id_to_token; // Compatibility mirror of the byte-keyed token map
} bpe_encoder_t;

typedef enum {
    BPE_SPECIAL_PAD = 0,
    BPE_SPECIAL_UNK,
    BPE_SPECIAL_BOS,
    BPE_SPECIAL_EOS,
} bpe_special_token_t;

#define BPE_BYTE_TOKEN_COUNT UINT32_C(256)
#define BPE_PAD_TOKEN_ID UINT32_C(256)
#define BPE_UNK_TOKEN_ID UINT32_C(257)
#define BPE_BOS_TOKEN_ID UINT32_C(258)
#define BPE_EOS_TOKEN_ID UINT32_C(259)
#define BPE_FIRST_SPECIAL_AWARE_MERGE_ID UINT32_C(260)
#define BPE_NO_TOKEN_ID UINT32_MAX

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

/* New tokenizer mode. Byte IDs remain 0..255 for compatibility; four
 * non-text control IDs are reserved before learned merges. */
bpe_errors_t bpe_encoder_new_with_special_tokens(bpe_encoder_t *encoder,
                                                  size_t max_vocab_size);
int bpe_encoder_has_special_tokens(const bpe_encoder_t *encoder);
uint32_t bpe_encoder_special_token_id(const bpe_encoder_t *encoder,
                                      bpe_special_token_t token);
size_t bpe_encoder_first_learned_id(const bpe_encoder_t *encoder);
int bpe_token_is_control(const bpe_encoder_t *encoder, uint32_t token_id);

/**
 * Trains the BPE encoder on input text by building the vocabulary.
 * @param encoder: Pointer to the encoder
 * @param input: Input text to train on
 * @param input_len: Length of input text
 * @return BPE_SUCCESS on success, error code otherwise
 */
bpe_errors_t bpe_train(bpe_encoder_t *encoder, const char *input, size_t input_len);

/** Freeze token IDs and merge order. This operation is irreversible. */
bpe_errors_t bpe_encoder_freeze(bpe_encoder_t *encoder);

/** Return nonzero when further bpe_train calls will be rejected. */
int bpe_encoder_is_frozen(const bpe_encoder_t *encoder);

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
 * Save a trained vocabulary, including merge order, to a legacy/tooling
 * sidecar. The current CLI also embeds this state in its model bundle.
 */
bpe_errors_t bpe_encoder_save(const bpe_encoder_t *encoder, const char *filename);

/** Write/read a vocabulary at the current position of an open stream. */
bpe_errors_t bpe_encoder_write(const bpe_encoder_t *encoder, FILE *file);
bpe_errors_t bpe_encoder_read(bpe_encoder_t *encoder, FILE *file);

/* Canonical little-endian tokenizer payload used inside versioned model
 * bundles. Version 2 payloads begin with DRNZBPP2 and store binary-safe token
 * bytes; the reader retains support for the unmarked version 1 payload. The
 * returned byte buffer is owned by the caller. */
bpe_errors_t bpe_encoder_serialize_portable(const bpe_encoder_t *encoder,
                                            uint8_t **out_data,
                                            size_t *out_size);
bpe_errors_t bpe_encoder_deserialize_portable(bpe_encoder_t *encoder,
                                              const uint8_t *data,
                                              size_t size);

/* Read only the allocation-driving max-vocabulary field after validating the
 * portable payload version and fixed header. */
bpe_errors_t bpe_encoder_portable_max_vocab(const uint8_t *data, size_t size,
                                            uint64_t *out_max_vocab);

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
