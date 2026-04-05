/*
 * Phase 3: Advanced sampling strategies for token generation
 * Implements top-k, top-p (nucleus), and beam search
 */

#ifndef SAMPLING_H
#define SAMPLING_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    SAMPLING_GREEDY,      // Argmax selection
    SAMPLING_TOPK,        // Restrict to top-k highest probabilities
    SAMPLING_TOPP,        // Nucleus sampling (restrict to cumulative prob p)
    SAMPLING_BEAM,        // Beam search (multiple hypotheses)
} sampling_strategy_t;

typedef struct {
    uint32_t *tokens;     // Token IDs for this beam
    size_t length;        // Number of tokens
    float score;          // Cumulative score (log probability)
} beam_t;

typedef struct {
    beam_t *beams;         // Array of beam hypotheses
    size_t num_beams;      // Number of active beams
    size_t beam_width;     // Max beams to maintain
} beam_search_t;

/**
 * Greedy sampling: select highest probability token
 * @param logits: Output logits from model (vocab_size length)
 * @param vocab_size: Size of vocabulary
 * @return Selected token ID
 */
uint32_t sample_greedy(float *logits, size_t vocab_size);

/**
 * Top-k sampling: restrict to k highest probability tokens, then sample
 * @param logits: Output logits from model
 * @param vocab_size: Size of vocabulary
 * @param k: Number of top tokens to consider
 * @return Selected token ID
 */
uint32_t sample_topk(float *logits, size_t vocab_size, size_t k);

/**
 * Top-p (nucleus) sampling: restrict to minimum set of tokens with cumulative prob >= p
 * @param logits: Output logits from model
 * @param vocab_size: Size of vocabulary
 * @param p: Cumulative probability threshold (typically 0.9)
 * @return Selected token ID
 */
uint32_t sample_topp(float *logits, size_t vocab_size, float p);

/**
 * Initialize beam search
 * @param beam_width: Number of beams to maintain
 * @param initial_token: Starting token
 * @return Initialized beam search structure
 */
beam_search_t beam_search_init(size_t beam_width, uint32_t initial_token);

/**
 * Generate next tokens using beam search
 * @param logits: Output logits from model
 * @param vocab_size: Size of vocabulary
 * @param beam_search: Current beam search state (modified in-place)
 * @return Best token ID from beam search
 */
uint32_t beam_search_step(float *logits, size_t vocab_size, beam_search_t *beam_search);

/**
 * Get best hypothesis from beam search
 * @param beam_search: Completed beam search
 * @return Best beam hypothesis
 */
beam_t beam_search_best(const beam_search_t *beam_search);

/**
 * Free beam search resources
 */
void beam_search_free(beam_search_t *beam_search);

#endif // SAMPLING_H
