/*
 * Phase 3: Batch processing for efficient training
 * Processes multiple sequences in parallel
 */

#ifndef BATCH_H
#define BATCH_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t **token_sequences;  // Array of token ID sequences
    uint32_t *target_tokens;     // Target tokens for each sequence
    size_t *sequence_lengths;    // Length of each sequence
    size_t batch_size;           // Number of sequences in batch
    size_t current_idx;          // Current position for iteration
} batch_t;

/**
 * Create a new batch
 * @param batch_size: Number of sequences per batch
 * @param max_seq_len: Maximum sequence length
 * @return Allocated batch structure
 */
batch_t *batch_create(size_t batch_size, size_t max_seq_len);

/**
 * Add sequence to batch
 * @param batch: Batch to add to
 * @param tokens: Token IDs to add
 * @param seq_len: Length of sequence
 * @param target_token: Target/label token
 * @return 0 on success, -1 if batch full
 */
int batch_add_sequence(batch_t *batch, uint32_t *tokens, size_t seq_len, uint32_t target_token);

/**
 * Check if batch is full
 */
int batch_is_full(const batch_t *batch);

/**
 * Check if batch has any sequences
 */
int batch_has_data(const batch_t *batch);

/**
 * Get size of batch
 */
size_t batch_get_size(const batch_t *batch);

/**
 * Reset batch for reuse
 */
void batch_reset(batch_t *batch);

/**
 * Free batch resources
 */
void batch_free(batch_t *batch);

#endif // BATCH_H
