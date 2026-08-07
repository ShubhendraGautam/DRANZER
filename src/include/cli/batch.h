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
    /* Per-POSITION targets, parallel to token_sequences: for example i,
     * target_sequences[i][p] is the token that should follow position p.
     * This was a single uint32_t per example while only the last position
     * of a window was supervised; every position is now (core/lm_head.h),
     * so a window of length L carries L targets. */
    uint32_t **target_sequences;
    size_t *sequence_lengths;    // Length of each sequence
    size_t batch_size;           // Number of sequences in batch
    size_t current_idx;          // Current position for iteration
    size_t max_seq_len;          // Capacity of each token/target sequence
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
 * @param targets: seq_len targets, one per position - targets[p] is the
 *   token that should follow tokens[p]. Entries may be
 *   LM_HEAD_IGNORE_TARGET to leave a position unsupervised.
 * @return 0 on success, -1 if batch full
 */
int batch_add_sequence(batch_t *batch, const uint32_t *tokens, size_t seq_len,
                       const uint32_t *targets);

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

/* Deterministic in-place Fisher-Yates permutation of the populated
 * examples. This RNG is local to the batch and never consumes model RNG. */
void batch_shuffle(batch_t *batch, uint64_t seed);

/**
 * Reset batch for reuse
 */
void batch_reset(batch_t *batch);

/**
 * Free batch resources
 */
void batch_free(batch_t *batch);

#endif // BATCH_H
