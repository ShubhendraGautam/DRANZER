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

#define MAX_PAIR_LEN 256

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
    size_t pair_len1 = strlen(tokens[0]);
    for (size_t i = 0; i < *token_count; i++) {
        int token_len = strlen(tokens[i]);
        if (i < *token_count - 1) {
            int next_len = strlen(tokens[i + 1]);
            char combined[MAX_PAIR_LEN];
            snprintf(combined, MAX_PAIR_LEN, "%s%s", tokens[i], tokens[i + 1]);

            if (strcmp(combined, pair) == 0) {
                tokens[i] = realloc(tokens[i], MAX_PAIR_LEN);
                strncpy(tokens[i], combined, MAX_PAIR_LEN - 1);
                tokens[i][MAX_PAIR_LEN - 1] = '\0';

                // Free the token we're about to remove
                free(tokens[i + 1]);
                
                // Shift remaining tokens
                for (size_t j = i + 1; j < *token_count - 1; j++) {
                    tokens[j] = tokens[j + 1];
                }
                (*token_count)--;
                i--; // Check this position again in case of multiple merges
            }
        }
    }
}

bpe_errors_t bpe_encoder_new(bpe_encoder_t *encoder, size_t max_vocab_size) {
    DEBUG_PRINT("Creating BPE encoder with max vocab size: %zu\n", max_vocab_size);
    if (encoder == NULL || max_vocab_size < 256) {
        return BPE_INVALID_INPUT;
    }

    if (hashmap_new(&encoder->token_to_id, 1000) != HASHMAP_SUCCESS ||
        hashmap_new(&encoder->id_to_token, 1000) != HASHMAP_SUCCESS) {
        return BPE_ALLOCATION_FAILURE;
    }

    encoder->tokens = malloc(sizeof(bpe_token_t) * max_vocab_size);
    if (encoder->tokens == NULL) {
        hashmap_free(&encoder->token_to_id);
        hashmap_free(&encoder->id_to_token);
        return BPE_ALLOCATION_FAILURE;
    }

    encoder->vocab_size = 0;
    encoder->max_vocab_size = max_vocab_size;

    // Initialize with single-character tokens (0-255)
    for (int i = 0; i < 256 && encoder->vocab_size < max_vocab_size; i++) {
        char byte_char[2] = {(char)i, '\0'};
        encoder->tokens[encoder->vocab_size].token = malloc(2);
        if (encoder->tokens[encoder->vocab_size].token == NULL) {
            return BPE_ALLOCATION_FAILURE;
        }
        strcpy(encoder->tokens[encoder->vocab_size].token, byte_char);
        encoder->tokens[encoder->vocab_size].id = encoder->vocab_size;
        encoder->tokens[encoder->vocab_size].frequency = 0;

        // Allocate ID values on heap for safe storage in hashmap
        uint32_t *id_ptr = malloc(sizeof(uint32_t));
        if (id_ptr == NULL) {
            return BPE_ALLOCATION_FAILURE;
        }
        *id_ptr = encoder->vocab_size;
        
        hashmap_insert(&encoder->token_to_id, byte_char, id_ptr, HASHMAP_VALUE_TYPE_INT);
        hashmap_insert(&encoder->id_to_token, byte_char, id_ptr, HASHMAP_VALUE_TYPE_INT);

        encoder->vocab_size++;
    }

    return BPE_SUCCESS;
}

bpe_errors_t bpe_train(bpe_encoder_t *encoder, const char *input, size_t input_len) {
    DEBUG_PRINT("Starting BPE training with input_len=%zu\n", input_len);
    if (encoder == NULL || input == NULL || input_len == 0) {
        return BPE_INVALID_INPUT;
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

        // Allocate ID values on heap for safe storage in hashmap
        uint32_t *id_ptr = malloc(sizeof(uint32_t));
        if (id_ptr == NULL) {
            free(pairs);
            return BPE_ALLOCATION_FAILURE;
        }
        *id_ptr = encoder->vocab_size;
        
        hashmap_insert(&encoder->token_to_id, new_token->token, id_ptr, HASHMAP_VALUE_TYPE_INT);
        hashmap_insert(&encoder->id_to_token, new_token->token, id_ptr, HASHMAP_VALUE_TYPE_INT);

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

bpe_errors_t bpe_encode(bpe_encoder_t *encoder, const char *input, size_t input_len, bpe_tokens_t *tokens) {
    if (encoder == NULL || input == NULL || tokens == NULL) {
        return BPE_INVALID_INPUT;
    }

    // Start with individual characters
    char **working_tokens = malloc(input_len * sizeof(char *));
    if (working_tokens == NULL) {
        return BPE_ALLOCATION_FAILURE;
    }

    for (size_t i = 0; i < input_len; i++) {
        working_tokens[i] = malloc(MAX_PAIR_LEN);
        if (working_tokens[i] == NULL) {
            return BPE_ALLOCATION_FAILURE;
        }
        snprintf(working_tokens[i], MAX_PAIR_LEN, "%c", input[i]);
    }
    size_t token_count = input_len;

    // Greedily apply merges from vocabulary
    for (size_t v = 256; v < encoder->vocab_size; v++) {
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
        return BPE_ALLOCATION_FAILURE;
    }

    for (size_t i = 0; i < token_count; i++) {
        uint32_t id;
        hashmap_value_type_t type;
        if (hashmap_get(&encoder->token_to_id, working_tokens[i], &id, &type) == HASHMAP_SUCCESS) {
            tokens->token_ids[i] = id;
        } else {
            // Token not in vocabulary, use unknown token id (255)
            tokens->token_ids[i] = 255;
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
    if (encoder == NULL || token_ids == NULL || output == NULL) {
        return BPE_INVALID_INPUT;
    }

    // Calculate required size
    size_t required_size = 1;
    for (size_t i = 0; i < token_count; i++) {
        if (token_ids[i] < encoder->vocab_size) {
            required_size += strlen(encoder->tokens[token_ids[i]].token);
        }
    }

    *output = malloc(required_size);
    if (*output == NULL) {
        return BPE_ALLOCATION_FAILURE;
    }

    (*output)[0] = '\0';

    for (size_t i = 0; i < token_count; i++) {
        if (token_ids[i] < encoder->vocab_size) {
            strcat(*output, encoder->tokens[token_ids[i]].token);
        }
    }

    *output_len = strlen(*output);
    return BPE_SUCCESS;
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
    }

    hashmap_free(&encoder->token_to_id);
    hashmap_free(&encoder->id_to_token);

    encoder->vocab_size = 0;
    encoder->max_vocab_size = 0;

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

