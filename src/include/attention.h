#ifndef ATTENTION_H
#define ATTENTION_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    ATTENTION_SUCCESS = 0,
    ATTENTION_INVALID_INPUT,
    ATTENTION_ALLOCATION_FAILURE,
} attention_errors_t;

typedef struct {
    float *weights;      // Flattened 2D attention weight matrix (seq_len x seq_len)
    size_t seq_len;      // Sequence length
} attention_weights_t;

/**
 * Computes self-attention weights for a sequence of tokens.
 * Uses simple dot-product attention with scaled scores.
 * 
 * @param token_ids: Array of token IDs
 * @param seq_len: Length of the sequence
 * @param embedding_dim: Dimension of embeddings (for scaling)
 * @param output: Pointer to store attention weights matrix
 * @return ATTENTION_SUCCESS on success, error code otherwise
 */
attention_errors_t attention_compute_self_attention(uint32_t *token_ids, 
                                                    size_t seq_len, 
                                                    size_t embedding_dim,
                                                    attention_weights_t *output);

/**
 * Prints attention weights matrix (for debugging/visualization).
 * 
 * @param weights: Pointer to attention weights
 * @param max_print: Maximum number of rows/cols to print
 */
void attention_print_weights(attention_weights_t *weights, size_t max_print);

/**
 * Frees attention weights structure.
 * 
 * @param weights: Pointer to attention weights
 * @return ATTENTION_SUCCESS on success
 */
attention_errors_t attention_weights_free(attention_weights_t *weights);

#endif /* ATTENTION_H */
