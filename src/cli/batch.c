/*
 * Phase 3: Batch processing implementation
 * Handles multiple sequences efficiently
 */

#include "cli/batch.h"
#include "common/debug.h"
#include <stdlib.h>
#include <string.h>

/* Create a new batch */
batch_t *batch_create(size_t batch_size, size_t max_seq_len) {
    batch_t *batch = malloc(sizeof(batch_t));
    if (!batch) return NULL;
    
    batch->batch_size = batch_size;
    batch->current_idx = 0;
    
    /* Pre-allocate storage for sequences */
    batch->token_sequences = malloc(batch_size * sizeof(uint32_t *));
    batch->target_tokens = malloc(batch_size * sizeof(uint32_t));
    batch->sequence_lengths = malloc(batch_size * sizeof(size_t));
    
    if (!batch->token_sequences || !batch->target_tokens || !batch->sequence_lengths) {
        free(batch->token_sequences);
        free(batch->target_tokens);
        free(batch->sequence_lengths);
        free(batch);
        return NULL;
    }
    
    /* Allocate individual sequences */
    for (size_t i = 0; i < batch_size; i++) {
        batch->token_sequences[i] = malloc(max_seq_len * sizeof(uint32_t));
        if (!batch->token_sequences[i]) {
            /* Cleanup on partial failure */
            for (size_t j = 0; j < i; j++) {
                free(batch->token_sequences[j]);
            }
            free(batch->token_sequences);
            free(batch->target_tokens);
            free(batch->sequence_lengths);
            free(batch);
            return NULL;
        }
    }
    
    /* Initialize */
    memset(batch->target_tokens, 0, batch_size * sizeof(uint32_t));
    memset(batch->sequence_lengths, 0, batch_size * sizeof(size_t));
    
    DEBUG_PRINT("Batch created with size %zu, max_seq_len %zu\n", batch_size, max_seq_len);
    
    return batch;
}

/* Add sequence to batch */
int batch_add_sequence(batch_t *batch, uint32_t *tokens, size_t seq_len, uint32_t target_token) {
    if (!batch || !tokens || seq_len == 0) return -1;
    
    if (batch->current_idx >= batch->batch_size) {
        DEBUG_PRINT("Batch is full (current_idx=%zu, batch_size=%zu)\n", batch->current_idx, batch->batch_size);
        return -1;
    }
    
    memcpy(batch->token_sequences[batch->current_idx], tokens, seq_len * sizeof(uint32_t));
    batch->sequence_lengths[batch->current_idx] = seq_len;
    batch->target_tokens[batch->current_idx] = target_token;
    
    batch->current_idx++;
    
    return 0;
}

/* Check if batch is full */
int batch_is_full(const batch_t *batch) {
    if (!batch) return -1;
    return batch->current_idx >= batch->batch_size ? 1 : 0;
}

/* Check if batch has any sequences */
int batch_has_data(const batch_t *batch) {
    if (!batch) return -1;
    return batch->current_idx > 0 ? 1 : 0;
}

/* Get size of batch */
size_t batch_get_size(const batch_t *batch) {
    if (!batch) return 0;
    return batch->current_idx;
}

/* Reset batch for reuse */
void batch_reset(batch_t *batch) {
    if (!batch) return;
    
    batch->current_idx = 0;
    memset(batch->target_tokens, 0, batch->batch_size * sizeof(uint32_t));
    memset(batch->sequence_lengths, 0, batch->batch_size * sizeof(size_t));
}

/* Free batch resources */
void batch_free(batch_t *batch) {
    if (!batch) return;
    
    for (size_t i = 0; i < batch->batch_size; i++) {
        free(batch->token_sequences[i]);
    }
    free(batch->token_sequences);
    free(batch->target_tokens);
    free(batch->sequence_lengths);
    free(batch);
}
