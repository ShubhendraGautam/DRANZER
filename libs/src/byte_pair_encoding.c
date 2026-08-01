/*
 * This file implements byte pair encoding (BPE) for tokenization.
 * BPE is a data compression technique used in NLP/LLMs to build a vocabulary
 * by iteratively merging the most frequent character or token pairs.
 */

#include "byte_pair_encoding.h"
#include "debug.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

#define MAX_PAIR_LEN 256
#define BPE_FILE_MAGIC_V1 "DRNZBPE1"
#define BPE_FILE_MAGIC_V2 "DRNZBPE2"
#define BPE_FILE_MAGIC_SIZE 8

typedef struct {
    char pair[MAX_PAIR_LEN];
    int frequency;
} pair_frequency_t;

/* Helper function to compare pair frequencies for sorting */
static int compare_pair_frequency(const void *a, const void *b) {
    const pair_frequency_t *pf1 = (const pair_frequency_t *)a;
    const pair_frequency_t *pf2 = (const pair_frequency_t *)b;
    return pf2->frequency - pf1->frequency; // Descending order
}

/* Count all pair frequencies in the token sequence */
static pair_frequency_t* count_pair_frequencies(char **tokens, size_t token_count, size_t *unique_pairs_count) {
    if (token_count < 2) {
        *unique_pairs_count = 0;
        return NULL;
    }

    hashmap_t pair_map;
    if (hashmap_new(&pair_map, 1000) != HASHMAP_SUCCESS) {
        return NULL;
    }

    // Count all adjacent pairs
    for (size_t i = 0; i < token_count - 1; i++) {
        char pair[MAX_PAIR_LEN];
        snprintf(pair, MAX_PAIR_LEN, "%s%s", tokens[i], tokens[i + 1]);

        int *count = malloc(sizeof(int));
        if (count == NULL) {
            hashmap_free(&pair_map);
            return NULL;
        }

        hashmap_value_type_t type;
        if (hashmap_get(&pair_map, pair, count, &type) == HASHMAP_SUCCESS) {
            (*count)++;
            hashmap_insert(&pair_map, pair, count, HASHMAP_VALUE_TYPE_INT);
        } else {
            *count = 1;
            hashmap_insert(&pair_map, pair, count, HASHMAP_VALUE_TYPE_INT);
        }
        free(count);
    }

    // Convert hashmap to sorted array
    // For simplicity, we'll iterate through and collect all pairs
    // Note: This is a simplified implementation; a full version would iterate the hashmap
    pair_frequency_t *pairs = malloc(sizeof(pair_frequency_t) * pair_map.bucket_count);
    if (pairs == NULL) {
        hashmap_free(&pair_map);
        return NULL;
    }

    size_t pair_idx = 0;
    for (size_t i = 0; i < pair_map.bucket_count; i++) {
        hashmap_entry_t *entry = pair_map.buckets[i];
        while (entry != NULL && pair_idx < pair_map.bucket_count) {
            strncpy(pairs[pair_idx].pair, entry->key, MAX_PAIR_LEN - 1);
            pairs[pair_idx].pair[MAX_PAIR_LEN - 1] = '\0';
            pairs[pair_idx].frequency = *(int *)entry->value;
            pair_idx++;
            entry = entry->next;
        }
    }

    *unique_pairs_count = pair_idx;

    // Sort by frequency (descending)
    qsort(pairs, pair_idx, sizeof(pair_frequency_t), compare_pair_frequency);

    hashmap_free(&pair_map);
    return pairs;
}

/* Merge the highest frequency pair in the token sequence */
static void merge_pair(char **tokens, size_t *token_count, const char *pair) {
    size_t i = 0;
    while (i + 1 < *token_count) {
        char combined[MAX_PAIR_LEN];
        snprintf(combined, MAX_PAIR_LEN, "%s%s", tokens[i], tokens[i + 1]);

        if (strcmp(combined, pair) == 0) {
            /* Every working token is allocated at MAX_PAIR_LEN, so no
             * resize is needed here. Keeping the allocation stable also
             * avoids losing the original pointer if realloc fails. */
            strncpy(tokens[i], combined, MAX_PAIR_LEN - 1);
            tokens[i][MAX_PAIR_LEN - 1] = '\0';

            free(tokens[i + 1]);
            for (size_t j = i + 1; j + 1 < *token_count; j++) {
                tokens[j] = tokens[j + 1];
            }
            (*token_count)--;
            continue; /* Recheck this position for another adjacent merge. */
        }
        i++;
    }
}

static int insert_token_id(bpe_encoder_t *encoder, const char *token, uint32_t id) {
    int token_rc = hashmap_insert(&encoder->token_to_id, token, &id, HASHMAP_VALUE_TYPE_INT);
    int id_rc = hashmap_insert(&encoder->id_to_token, token, &id, HASHMAP_VALUE_TYPE_INT);
    return (token_rc == HASHMAP_SUCCESS || token_rc == HASHMAP_KEY_EXISTS) &&
           (id_rc == HASHMAP_SUCCESS || id_rc == HASHMAP_KEY_EXISTS);
}

static bpe_errors_t bpe_encoder_new_mode(bpe_encoder_t *encoder,
                                         size_t max_vocab_size,
                                         int special_tokens) {
    DEBUG_PRINT("Creating BPE encoder with max vocab size: %zu\n", max_vocab_size);
    size_t minimum = special_tokens ? BPE_FIRST_SPECIAL_AWARE_MERGE_ID
                                    : BPE_BYTE_TOKEN_COUNT;
    if (encoder == NULL || max_vocab_size < minimum || max_vocab_size > UINT32_MAX ||
        max_vocab_size > SIZE_MAX / sizeof(bpe_token_t)) {
        return BPE_INVALID_INPUT;
    }

    memset(encoder, 0, sizeof(*encoder));

    if (hashmap_new(&encoder->token_to_id, 1000) != HASHMAP_SUCCESS) {
        return BPE_ALLOCATION_FAILURE;
    }
    if (hashmap_new(&encoder->id_to_token, 1000) != HASHMAP_SUCCESS) {
        hashmap_free(&encoder->token_to_id);
        return BPE_ALLOCATION_FAILURE;
    }

    encoder->tokens = calloc(max_vocab_size, sizeof(bpe_token_t));
    if (encoder->tokens == NULL) {
        hashmap_free(&encoder->token_to_id);
        hashmap_free(&encoder->id_to_token);
        return BPE_ALLOCATION_FAILURE;
    }

    encoder->vocab_size = 0;
    encoder->max_vocab_size = max_vocab_size;
    encoder->has_special_tokens = special_tokens;

    // Initialize with single-character tokens (0-255)
    for (int i = 0; i < 256 && encoder->vocab_size < max_vocab_size; i++) {
        char byte_char[2] = {(char)i, '\0'};
        encoder->tokens[encoder->vocab_size].token = malloc(2);
        if (encoder->tokens[encoder->vocab_size].token == NULL) {
            bpe_encoder_free(encoder);
            return BPE_ALLOCATION_FAILURE;
        }
        strcpy(encoder->tokens[encoder->vocab_size].token, byte_char);
        encoder->tokens[encoder->vocab_size].id = encoder->vocab_size;
        encoder->tokens[encoder->vocab_size].frequency = 0;

        /* hashmap_insert duplicates scalar values, so a stack value is
         * sufficient. The old heap temporary leaked once per token. */
        uint32_t id = (uint32_t)encoder->vocab_size;
        encoder->vocab_size++;
        if (!insert_token_id(encoder, byte_char, id)) {
            bpe_encoder_free(encoder);
            return BPE_ALLOCATION_FAILURE;
        }
    }

    if (special_tokens) {
        for (uint32_t id = BPE_PAD_TOKEN_ID; id <= BPE_EOS_TOKEN_ID; id++) {
            encoder->tokens[id].id = id;
            encoder->tokens[id].frequency = 0;
        }
        encoder->vocab_size = BPE_FIRST_SPECIAL_AWARE_MERGE_ID;
    }

    return BPE_SUCCESS;
}

bpe_errors_t bpe_encoder_new(bpe_encoder_t *encoder, size_t max_vocab_size) {
    return bpe_encoder_new_mode(encoder, max_vocab_size, 0);
}

bpe_errors_t bpe_encoder_new_with_special_tokens(bpe_encoder_t *encoder,
                                                  size_t max_vocab_size) {
    return bpe_encoder_new_mode(encoder, max_vocab_size, 1);
}

int bpe_encoder_has_special_tokens(const bpe_encoder_t *encoder) {
    return encoder && encoder->has_special_tokens;
}

uint32_t bpe_encoder_special_token_id(const bpe_encoder_t *encoder,
                                      bpe_special_token_t token) {
    if (!bpe_encoder_has_special_tokens(encoder)) return BPE_NO_TOKEN_ID;
    switch (token) {
        case BPE_SPECIAL_PAD: return BPE_PAD_TOKEN_ID;
        case BPE_SPECIAL_UNK: return BPE_UNK_TOKEN_ID;
        case BPE_SPECIAL_BOS: return BPE_BOS_TOKEN_ID;
        case BPE_SPECIAL_EOS: return BPE_EOS_TOKEN_ID;
        default: return BPE_NO_TOKEN_ID;
    }
}

size_t bpe_encoder_first_learned_id(const bpe_encoder_t *encoder) {
    return bpe_encoder_has_special_tokens(encoder)
               ? BPE_FIRST_SPECIAL_AWARE_MERGE_ID : BPE_BYTE_TOKEN_COUNT;
}

int bpe_token_is_control(const bpe_encoder_t *encoder, uint32_t token_id) {
    return bpe_encoder_has_special_tokens(encoder) &&
           token_id >= BPE_PAD_TOKEN_ID && token_id <= BPE_EOS_TOKEN_ID;
}

bpe_errors_t bpe_train(bpe_encoder_t *encoder, const char *input, size_t input_len) {
    DEBUG_PRINT("Starting BPE training with input_len=%zu\n", input_len);
    if (encoder == NULL || input == NULL || input_len == 0) {
        return BPE_INVALID_INPUT;
    }
    if (encoder->is_frozen) {
        return BPE_ENCODER_FROZEN;
    }

    // Split input into individual characters as initial tokens
    char **tokens = malloc(input_len * sizeof(char *));
    if (tokens == NULL) {
        return BPE_ALLOCATION_FAILURE;
    }

    for (size_t i = 0; i < input_len; i++) {
        tokens[i] = malloc(MAX_PAIR_LEN);
        if (tokens[i] == NULL) {
            return BPE_ALLOCATION_FAILURE;
        }
        snprintf(tokens[i], MAX_PAIR_LEN, "%c", input[i]);
    }
    size_t token_count = input_len;

    // Iteratively merge the most frequent pairs
    while (encoder->vocab_size < encoder->max_vocab_size) {
        size_t unique_pairs_count;
        pair_frequency_t *pairs = count_pair_frequencies(tokens, token_count, &unique_pairs_count);

        if (unique_pairs_count == 0 || pairs == NULL) {
            if (pairs != NULL) free(pairs);
            break;
        }

        // Get the most frequent pair
        pair_frequency_t best_pair = pairs[0];
        if (best_pair.frequency < 2) {
            free(pairs);
            break; // No more pairs worth merging
        }

        // Create new token from merged pair
        bpe_token_t *new_token = &encoder->tokens[encoder->vocab_size];
        new_token->token = malloc(MAX_PAIR_LEN);
        if (new_token->token == NULL) {
            free(pairs);
            return BPE_ALLOCATION_FAILURE;
        }

        strncpy(new_token->token, best_pair.pair, MAX_PAIR_LEN - 1);
        new_token->token[MAX_PAIR_LEN - 1] = '\0';
        new_token->id = encoder->vocab_size;
        new_token->frequency = best_pair.frequency;

        uint32_t id = (uint32_t)encoder->vocab_size;
        if (!insert_token_id(encoder, new_token->token, id)) {
            free(new_token->token);
            new_token->token = NULL;
            free(pairs);
            return BPE_ALLOCATION_FAILURE;
        }

        encoder->vocab_size++;

        // Merge the pair in the token sequence
        merge_pair(tokens, &token_count, best_pair.pair);

        free(pairs);
    }

    // Cleanup tokens - use token_count, not input_len, since tokens were merged
    for (size_t i = 0; i < token_count; i++) {
        free(tokens[i]);
    }
    free(tokens);

    return BPE_SUCCESS;
}

bpe_errors_t bpe_encoder_freeze(bpe_encoder_t *encoder) {
    if (!encoder || !encoder->tokens ||
        encoder->vocab_size < bpe_encoder_first_learned_id(encoder)) {
        return BPE_INVALID_INPUT;
    }
    encoder->is_frozen = 1;
    return BPE_SUCCESS;
}

int bpe_encoder_is_frozen(const bpe_encoder_t *encoder) {
    return encoder && encoder->is_frozen;
}

bpe_errors_t bpe_encode(bpe_encoder_t *encoder, const char *input, size_t input_len, bpe_tokens_t *tokens) {
    if (encoder == NULL || input == NULL || input_len == 0 || tokens == NULL) {
        return BPE_INVALID_INPUT;
    }
    tokens->token_ids = NULL;
    tokens->token_count = 0;

    // Start with individual characters
    char **working_tokens = malloc(input_len * sizeof(char *));
    if (working_tokens == NULL) {
        return BPE_ALLOCATION_FAILURE;
    }

    for (size_t i = 0; i < input_len; i++) {
        working_tokens[i] = malloc(MAX_PAIR_LEN);
        if (working_tokens[i] == NULL) {
            for (size_t j = 0; j < i; j++) free(working_tokens[j]);
            free(working_tokens);
            return BPE_ALLOCATION_FAILURE;
        }
        snprintf(working_tokens[i], MAX_PAIR_LEN, "%c", input[i]);
    }
    size_t token_count = input_len;

    // Greedily apply merges from vocabulary
    for (size_t v = bpe_encoder_first_learned_id(encoder);
         v < encoder->vocab_size; v++) {
        size_t i = 0;
        while (i < token_count - 1) {
            char combined[MAX_PAIR_LEN];
            snprintf(combined, MAX_PAIR_LEN, "%s%s", working_tokens[i], working_tokens[i + 1]);

            if (strcmp(combined, encoder->tokens[v].token) == 0) {
                strncpy(working_tokens[i], combined, MAX_PAIR_LEN - 1);
                working_tokens[i][MAX_PAIR_LEN - 1] = '\0';

                // Shift tokens
                for (size_t j = i + 1; j < token_count - 1; j++) {
                    strcpy(working_tokens[j], working_tokens[j + 1]);
                }
                token_count--;
            } else {
                i++;
            }
        }
    }

    // Convert tokens to IDs
    tokens->token_ids = malloc(token_count * sizeof(uint32_t));
    if (tokens->token_ids == NULL) {
        for (size_t i = 0; i < input_len; i++) free(working_tokens[i]);
        free(working_tokens);
        return BPE_ALLOCATION_FAILURE;
    }

    for (size_t i = 0; i < token_count; i++) {
        uint32_t id;
        hashmap_value_type_t type;
        if (hashmap_get(&encoder->token_to_id, working_tokens[i], &id, &type) == HASHMAP_SUCCESS) {
            tokens->token_ids[i] = id;
        } else {
            tokens->token_ids[i] = bpe_encoder_has_special_tokens(encoder)
                                       ? BPE_UNK_TOKEN_ID : UINT32_C(255);
        }
    }
    tokens->token_count = token_count;

    // Cleanup
    for (size_t i = 0; i < input_len; i++) {
        free(working_tokens[i]);
    }
    free(working_tokens);

    return BPE_SUCCESS;
}

bpe_errors_t bpe_decode(bpe_encoder_t *encoder, const uint32_t *token_ids, size_t token_count, char **output, size_t *output_len) {
    if (encoder == NULL || token_ids == NULL || output == NULL || output_len == NULL) {
        return BPE_INVALID_INPUT;
    }

    // Calculate required size
    size_t required_size = 1;
    for (size_t i = 0; i < token_count; i++) {
        uint32_t id = token_ids[i];
        if (id < encoder->vocab_size && !bpe_token_is_control(encoder, id) &&
            encoder->tokens[id].token) {
            size_t length = strlen(encoder->tokens[id].token);
            if (length > SIZE_MAX - required_size) return BPE_INVALID_INPUT;
            required_size += length;
        }
    }

    *output = malloc(required_size);
    if (*output == NULL) {
        return BPE_ALLOCATION_FAILURE;
    }

    size_t offset = 0;
    for (size_t i = 0; i < token_count; i++) {
        uint32_t id = token_ids[i];
        if (id < encoder->vocab_size && !bpe_token_is_control(encoder, id) &&
            encoder->tokens[id].token) {
            size_t length = strlen(encoder->tokens[id].token);
            memcpy(*output + offset, encoder->tokens[id].token, length);
            offset += length;
        }
    }
    (*output)[offset] = '\0';
    *output_len = offset;
    return BPE_SUCCESS;
}

static int write_exact(FILE *file, const void *data, size_t size) {
    return fwrite(data, 1, size, file) == size;
}

static int read_exact(FILE *file, void *data, size_t size) {
    return fread(data, 1, size, file) == size;
}

static void portable_put_u32(uint8_t *data, uint32_t value) {
    for (size_t i = 0; i < 4; i++) data[i] = (uint8_t)(value >> (i * 8));
}

static void portable_put_u64(uint8_t *data, uint64_t value) {
    for (size_t i = 0; i < 8; i++) data[i] = (uint8_t)(value >> (i * 8));
}

static uint32_t portable_get_u32(const uint8_t *data) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; i++) value |= (uint32_t)data[i] << (i * 8);
    return value;
}

static uint64_t portable_get_u64(const uint8_t *data) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; i++) value |= (uint64_t)data[i] << (i * 8);
    return value;
}

/* Portable artifacts are untrusted input. A frozen vocabulary larger than
 * this would allocate a disproportionately large sparse token table before
 * a single merge is decoded; such models are outside this runtime's scope. */
#define BPE_PORTABLE_MAX_VOCAB UINT64_C(1048576)

bpe_errors_t bpe_encoder_serialize_portable(const bpe_encoder_t *encoder,
                                            uint8_t **out_data,
                                            size_t *out_size) {
    if (!encoder || !out_data || !out_size || !encoder->tokens ||
        encoder->vocab_size < 256 || encoder->vocab_size > encoder->max_vocab_size ||
        encoder->max_vocab_size > BPE_PORTABLE_MAX_VOCAB) return BPE_INVALID_INPUT;
    *out_data = NULL;
    *out_size = 0;

    size_t first_learned = bpe_encoder_first_learned_id(encoder);
    if (encoder->vocab_size < first_learned) return BPE_INVALID_INPUT;
    size_t size = 24;
    for (size_t i = first_learned; i < encoder->vocab_size; i++) {
        size_t length = strlen(encoder->tokens[i].token);
        int64_t frequency = encoder->tokens[i].frequency;
        if (length == 0 || length >= MAX_PAIR_LEN || length > UINT32_MAX ||
            frequency > INT32_MAX || frequency < INT32_MIN ||
            length > SIZE_MAX - size - 8) return BPE_INVALID_INPUT;
        size += 8 + length;
    }
    uint8_t *data = malloc(size);
    if (!data) return BPE_ALLOCATION_FAILURE;

    portable_put_u64(data, encoder->max_vocab_size);
    portable_put_u64(data + 8, encoder->vocab_size);
    portable_put_u32(data + 16, encoder->is_frozen ? 1u : 0u);
    portable_put_u32(data + 20, encoder->has_special_tokens ? 1u : 0u);
    size_t offset = 24;
    for (size_t i = first_learned; i < encoder->vocab_size; i++) {
        size_t length = strlen(encoder->tokens[i].token);
        portable_put_u32(data + offset, (uint32_t)length);
        portable_put_u32(data + offset + 4,
                         (uint32_t)(int32_t)encoder->tokens[i].frequency);
        memcpy(data + offset + 8, encoder->tokens[i].token, length);
        offset += 8 + length;
    }
    *out_data = data;
    *out_size = size;
    return BPE_SUCCESS;
}

bpe_errors_t bpe_encoder_deserialize_portable(bpe_encoder_t *encoder,
                                              const uint8_t *data,
                                              size_t size) {
    if (!encoder || !data || size < 24) return BPE_INVALID_INPUT;
    uint64_t stored_max = portable_get_u64(data);
    uint64_t stored_vocab = portable_get_u64(data + 8);
    uint32_t frozen = portable_get_u32(data + 16);
    uint32_t flags = portable_get_u32(data + 20);
    size_t first_learned = flags == 1 ? BPE_FIRST_SPECIAL_AWARE_MERGE_ID
                                      : BPE_BYTE_TOKEN_COUNT;
    if (flags > 1 || stored_max < first_learned || stored_vocab < first_learned ||
        stored_vocab > stored_max ||
        stored_max > BPE_PORTABLE_MAX_VOCAB || stored_max > SIZE_MAX ||
        frozen != 1) {
        return BPE_FORMAT_ERROR;
    }
    bpe_errors_t rc = flags == 1
                          ? bpe_encoder_new_with_special_tokens(
                                encoder, (size_t)stored_max)
                          : bpe_encoder_new(encoder, (size_t)stored_max);
    if (rc != BPE_SUCCESS) return rc;

    size_t offset = 24;
    for (size_t i = first_learned; i < (size_t)stored_vocab; i++) {
        if (size - offset < 8) { rc = BPE_FORMAT_ERROR; break; }
        uint32_t length = portable_get_u32(data + offset);
        int32_t frequency = (int32_t)portable_get_u32(data + offset + 4);
        offset += 8;
        if (length == 0 || length >= MAX_PAIR_LEN || (size_t)length > size - offset ||
            memchr(data + offset, '\0', length) != NULL) {
            rc = BPE_FORMAT_ERROR;
            break;
        }
#if INT_MAX < INT32_MAX || INT_MIN > INT32_MIN
        if (frequency > INT_MAX || frequency < INT_MIN) {
            rc = BPE_FORMAT_ERROR;
            break;
        }
#endif
        bpe_token_t *token = &encoder->tokens[i];
        token->token = malloc((size_t)length + 1);
        if (!token->token) { rc = BPE_ALLOCATION_FAILURE; break; }
        encoder->vocab_size = i + 1;
        memcpy(token->token, data + offset, length);
        token->token[length] = '\0';
        token->id = (uint32_t)i;
        token->frequency = (int)frequency;
        if (!insert_token_id(encoder, token->token, (uint32_t)i)) {
            rc = BPE_ALLOCATION_FAILURE;
            break;
        }
        offset += length;
    }
    if (rc == BPE_SUCCESS && offset != size) rc = BPE_FORMAT_ERROR;
    if (rc != BPE_SUCCESS) bpe_encoder_free(encoder);
    else encoder->is_frozen = 1;
    return rc;
}

bpe_errors_t bpe_encoder_write(const bpe_encoder_t *encoder, FILE *file) {
    if (!encoder || !file || !encoder->tokens || encoder->vocab_size < 256 ||
        encoder->vocab_size > encoder->max_vocab_size) {
        return BPE_INVALID_INPUT;
    }

    uint64_t max_vocab_size = (uint64_t)encoder->max_vocab_size;
    uint64_t vocab_size = (uint64_t)encoder->vocab_size;
    size_t first_learned = bpe_encoder_first_learned_id(encoder);
    if (encoder->vocab_size < first_learned) return BPE_INVALID_INPUT;
    const char *magic = encoder->has_special_tokens
                            ? BPE_FILE_MAGIC_V2 : BPE_FILE_MAGIC_V1;
    int ok = write_exact(file, magic, BPE_FILE_MAGIC_SIZE) &&
             write_exact(file, &max_vocab_size, sizeof(max_vocab_size)) &&
             write_exact(file, &vocab_size, sizeof(vocab_size));

    /* Byte tokens 0..255 are deterministic and need not be repeated in
     * the file. Merge order is the array order from index 256 onward. */
    for (size_t i = first_learned; ok && i < encoder->vocab_size; i++) {
        size_t token_len = strlen(encoder->tokens[i].token);
        if (token_len == 0 || token_len >= MAX_PAIR_LEN || token_len > UINT32_MAX) {
            ok = 0;
            break;
        }
        uint32_t stored_len = (uint32_t)token_len;
        int32_t frequency = (int32_t)encoder->tokens[i].frequency;
        ok = write_exact(file, &stored_len, sizeof(stored_len)) &&
             write_exact(file, &frequency, sizeof(frequency)) &&
             write_exact(file, encoder->tokens[i].token, token_len);
    }

    return ok ? BPE_SUCCESS : BPE_IO_ERROR;
}

bpe_errors_t bpe_encoder_read(bpe_encoder_t *encoder, FILE *file) {
    if (!encoder || !file) return BPE_INVALID_INPUT;
    char magic[BPE_FILE_MAGIC_SIZE];
    uint64_t stored_max_vocab = 0;
    uint64_t stored_vocab = 0;
    if (!read_exact(file, magic, sizeof(magic))) return BPE_FORMAT_ERROR;
    int special_tokens = memcmp(magic, BPE_FILE_MAGIC_V2, sizeof(magic)) == 0;
    if (!special_tokens && memcmp(magic, BPE_FILE_MAGIC_V1, sizeof(magic)) != 0) {
        return BPE_FORMAT_ERROR;
    }
    size_t first_learned = special_tokens
                               ? BPE_FIRST_SPECIAL_AWARE_MERGE_ID
                               : BPE_BYTE_TOKEN_COUNT;
    if (!read_exact(file, &stored_max_vocab, sizeof(stored_max_vocab)) ||
        !read_exact(file, &stored_vocab, sizeof(stored_vocab)) ||
        stored_max_vocab < first_learned || stored_vocab < first_learned ||
        stored_vocab > stored_max_vocab ||
        stored_max_vocab > BPE_PORTABLE_MAX_VOCAB ||
        stored_max_vocab > SIZE_MAX) {
        return BPE_FORMAT_ERROR;
    }

    bpe_errors_t rc = special_tokens
                          ? bpe_encoder_new_with_special_tokens(
                                encoder, (size_t)stored_max_vocab)
                          : bpe_encoder_new(encoder, (size_t)stored_max_vocab);
    if (rc != BPE_SUCCESS) {
        return rc;
    }

    for (size_t i = first_learned; i < (size_t)stored_vocab; i++) {
        uint32_t token_len = 0;
        int32_t frequency = 0;
        if (!read_exact(file, &token_len, sizeof(token_len)) ||
            !read_exact(file, &frequency, sizeof(frequency)) ||
            token_len == 0 || token_len >= MAX_PAIR_LEN) {
            rc = BPE_FORMAT_ERROR;
            break;
        }

        bpe_token_t *token = &encoder->tokens[i];
        token->token = malloc((size_t)token_len + 1);
        if (!token->token) {
            rc = BPE_ALLOCATION_FAILURE;
            break;
        }
        /* Count the slot as owned immediately so error cleanup also frees
         * a token whose payload turns out to be truncated. */
        encoder->vocab_size = i + 1;
        if (!read_exact(file, token->token, token_len)) {
            rc = BPE_FORMAT_ERROR;
            break;
        }
        token->token[token_len] = '\0';
        token->id = (uint32_t)i;
        token->frequency = (int)frequency;

        uint32_t id = (uint32_t)i;
        if (!insert_token_id(encoder, token->token, id)) {
            rc = BPE_ALLOCATION_FAILURE;
            break;
        }
    }

    if (rc != BPE_SUCCESS) {
        bpe_encoder_free(encoder);
    } else {
        encoder->is_frozen = 1;
    }
    return rc;
}

bpe_errors_t bpe_encoder_save(const bpe_encoder_t *encoder, const char *filename) {
    if (!filename) return BPE_INVALID_INPUT;
    FILE *file = fopen(filename, "wb");
    if (!file) return BPE_IO_ERROR;
    bpe_errors_t rc = bpe_encoder_write(encoder, file);
    if (fclose(file) != 0 && rc == BPE_SUCCESS) rc = BPE_IO_ERROR;
    return rc;
}

bpe_errors_t bpe_encoder_load(bpe_encoder_t *encoder, const char *filename) {
    if (!encoder || !filename) return BPE_INVALID_INPUT;
    FILE *file = fopen(filename, "rb");
    if (!file) return BPE_IO_ERROR;
    bpe_errors_t rc = bpe_encoder_read(encoder, file);
    if (rc == BPE_SUCCESS && fgetc(file) != EOF) {
        bpe_encoder_free(encoder);
        rc = BPE_FORMAT_ERROR;
    }
    fclose(file);
    return rc;
}

bpe_errors_t bpe_encoder_free(bpe_encoder_t *encoder) {
    if (encoder == NULL) {
        return BPE_NULL_ENCODER;
    }

    if (encoder->tokens != NULL) {
        for (size_t i = 0; i < encoder->vocab_size; i++) {
            if (encoder->tokens[i].token != NULL) {
                free(encoder->tokens[i].token);
            }
        }
        free(encoder->tokens);
        encoder->tokens = NULL;
    }

    hashmap_free(&encoder->token_to_id);
    hashmap_free(&encoder->id_to_token);

    encoder->vocab_size = 0;
    encoder->max_vocab_size = 0;
    encoder->is_frozen = 0;
    encoder->has_special_tokens = 0;

    return BPE_SUCCESS;
}

bpe_errors_t bpe_tokens_free(bpe_tokens_t *tokens) {
    if (tokens == NULL) {
        return BPE_INVALID_INPUT;
    }

    if (tokens->token_ids != NULL) {
        free(tokens->token_ids);
        tokens->token_ids = NULL;
    }

    tokens->token_count = 0;
    return BPE_SUCCESS;
}
