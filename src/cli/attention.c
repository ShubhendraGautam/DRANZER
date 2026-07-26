/*
 * Attention mechanism implementation
 * Provides simple self-attention computation for token sequences
 */

#include "cli/attention.h"
#include "common/debug.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Simple softmax implementation - OPTIMIZED */
static void softmax(float *values, size_t size) {
    if (size == 0) return;

    // Find max for numerical stability
    float max_val = values[0];
    for (size_t i = 1; i < size; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    // Compute exp and sum in single pass
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float exp_val = expf(values[i] - max_val);
        values[i] = exp_val;
        sum += exp_val;
    }

    // Normalize using reciprocal (faster than division)
    if (sum > 0) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < size; i++) {
            values[i] *= inv_sum;
        }
    }
}

/* Compute dot product attention score between two embeddings */
static float compute_attention_score(float *query, float *key, size_t dim) {
    float score = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        score += query[i] * key[i];
    }
    return score / sqrtf((float)dim); // Scale by sqrt(dim)
}

/* Generate simple pseudo-random embedding from token ID */
static void get_embedding(uint32_t token_id, float *embedding, size_t dim) {
    // Pseudo-random embedding based on token ID using sine function
    for (size_t i = 0; i < dim; i++) {
        // Simple hash-like function for reproducible embeddings
        float val = sinf((float)(token_id * 12.9898f + i * 78.233f));
        embedding[i] = (val - floorf(val)); // Normalize to [0, 1)
    }
}

attention_errors_t attention_compute_self_attention(uint32_t *token_ids,
                                                    size_t seq_len,
                                                    size_t embedding_dim,
                                                    attention_weights_t *output) {
    DEBUG_PRINT("Computing self-attention for seq_len=%zu, embedding_dim=%zu\n", seq_len, embedding_dim);
    if (token_ids == NULL || output == NULL || seq_len == 0 || embedding_dim == 0) {
        return ATTENTION_INVALID_INPUT;
    }

    // Allocate embeddings for all tokens
    float *embeddings = malloc(seq_len * embedding_dim * sizeof(float));
    if (embeddings == NULL) {
        return ATTENTION_ALLOCATION_FAILURE;
    }

    // Generate embeddings for each token
    for (size_t i = 0; i < seq_len; i++) {
        get_embedding(token_ids[i], &embeddings[i * embedding_dim], embedding_dim);
    }

    // Allocate attention weights matrix
    output->weights = malloc(seq_len * seq_len * sizeof(float));
    if (output->weights == NULL) {
        free(embeddings);
        return ATTENTION_ALLOCATION_FAILURE;
    }

    output->seq_len = seq_len;

    // Compute attention weights for each query position
    for (size_t i = 0; i < seq_len; i++) {
        float *scores = malloc(seq_len * sizeof(float));
        if (scores == NULL) {
            free(embeddings);
            free(output->weights);
            return ATTENTION_ALLOCATION_FAILURE;
        }

        // Compute scores between token i and all other tokens
        for (size_t j = 0; j < seq_len; j++) {
            scores[j] = compute_attention_score(&embeddings[i * embedding_dim],
                                               &embeddings[j * embedding_dim],
                                               embedding_dim);
        }

        // Apply softmax to get attention probabilities
        softmax(scores, seq_len);

        // Store attention weights for query position i
        for (size_t j = 0; j < seq_len; j++) {
            output->weights[i * seq_len + j] = scores[j];
        }

        free(scores);
    }

    free(embeddings);
    return ATTENTION_SUCCESS;
}

void attention_print_weights(attention_weights_t *weights, size_t max_print) {
    if (weights == NULL || weights->weights == NULL) {
        printf("Invalid attention weights\n");
        return;
    }

    size_t print_rows = weights->seq_len > max_print ? max_print : weights->seq_len;
    size_t print_cols = weights->seq_len > max_print ? max_print : weights->seq_len;

    printf("Attention Weights (%zu x %zu sample):\n", print_rows, print_cols);
    for (size_t i = 0; i < print_rows; i++) {
        printf("  [%2zu] ", i);
        for (size_t j = 0; j < print_cols; j++) {
            printf("%.3f ", weights->weights[i * weights->seq_len + j]);
        }
        printf("\n");
    }
}

attention_errors_t attention_weights_free(attention_weights_t *weights) {
    if (weights == NULL) {
        return ATTENTION_INVALID_INPUT;
    }

    if (weights->weights != NULL) {
        free(weights->weights);
        weights->weights = NULL;
    }

    weights->seq_len = 0;
    return ATTENTION_SUCCESS;
}
