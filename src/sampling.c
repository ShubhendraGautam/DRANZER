/*
 * Phase 3: Advanced sampling strategies implementation
 * Implements top-k, top-p (nucleus), and beam search
 */

#include "include/sampling.h"
#include "include/debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Compare function for qsort (descending order) */
typedef struct {
    float prob;
    uint32_t idx;
} prob_idx_t;

static int compare_prob_desc(const void *a, const void *b) {
    float prob_a = ((const prob_idx_t *)a)->prob;
    float prob_b = ((const prob_idx_t *)b)->prob;
    return (prob_a < prob_b) ? 1 : (prob_a > prob_b) ? -1 : 0;
}

/* Greedy sampling: select highest probability token */
uint32_t sample_greedy(float *logits, size_t vocab_size) {
    if (!logits || vocab_size == 0) return 0;
    
    uint32_t best_idx = 0;
    float best_logit = logits[0];
    
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_idx = i;
        }
    }
    
    DEBUG_PRINT("Greedy sampling selected token %u\n", best_idx);
    return best_idx;
}

/* Top-k sampling */
uint32_t sample_topk(float *logits, size_t vocab_size, size_t k) {
    if (!logits || vocab_size == 0 || k == 0) return 0;
    
    if (k > vocab_size) k = vocab_size;
    
    /* Create probability-index pairs */
    prob_idx_t *pairs = malloc(vocab_size * sizeof(prob_idx_t));
    for (size_t i = 0; i < vocab_size; i++) {
        pairs[i].prob = logits[i];
        pairs[i].idx = i;
    }
    
    /* Sort by probability (descending) */
    qsort(pairs, vocab_size, sizeof(prob_idx_t), compare_prob_desc);
    
    /* Keep only top-k, zero out rest */
    float best_prob = pairs[k-1].prob;
    float sum = 0.0f;
    for (size_t i = 0; i < k; i++) {
        sum += expf(pairs[i].prob - best_prob);
    }
    
    /* Normalize top-k */
    float r = (float)rand() / RAND_MAX * sum;
    float acc = 0.0f;
    uint32_t selected = pairs[0].idx;
    
    for (size_t i = 0; i < k; i++) {
        acc += expf(pairs[i].prob - best_prob);
        if (acc >= r) {
            selected = pairs[i].idx;
            break;
        }
    }
    
    free(pairs);
    DEBUG_PRINT("Top-k (k=%zu) sampling selected token %u\n", k, selected);
    
    return selected;
}

/* Top-p (nucleus) sampling */
uint32_t sample_topp(float *logits, size_t vocab_size, float p) {
    if (!logits || vocab_size == 0 || p <= 0.0f || p > 1.0f) return 0;
    
    /* Create probability-index pairs */
    prob_idx_t *pairs = malloc(vocab_size * sizeof(prob_idx_t));
    for (size_t i = 0; i < vocab_size; i++) {
        pairs[i].prob = logits[i];
        pairs[i].idx = i;
    }
    
    /* Sort by probability (descending) */
    qsort(pairs, vocab_size, sizeof(prob_idx_t), compare_prob_desc);
    
    /* Find threshold where cumulative prob reaches p */
    float max_prob = pairs[0].prob;
    float sum_all = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        sum_all += expf(pairs[i].prob - max_prob);
    }
    
    float cumsum = 0.0f;
    size_t cutoff = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        cumsum += expf(pairs[i].prob - max_prob) / sum_all;
        if (cumsum > p) {
            cutoff = i;
            break;
        }
    }
    
    if (cutoff == 0) cutoff = 1; /* At least keep top-1 */
    
    /* Sample from nucleus */
    float nucleus_sum = 0.0f;
    for (size_t i = 0; i <= cutoff; i++) {
        nucleus_sum += expf(pairs[i].prob - max_prob);
    }
    
    float r = (float)rand() / RAND_MAX * nucleus_sum;
    float acc = 0.0f;
    uint32_t selected = pairs[0].idx;
    
    for (size_t i = 0; i <= cutoff; i++) {
        acc += expf(pairs[i].prob - max_prob);
        if (acc >= r) {
            selected = pairs[i].idx;
            break;
        }
    }
    
    free(pairs);
    DEBUG_PRINT("Top-p (p=%.2f) sampling selected token %u from %zu candidates\n", p, selected, cutoff + 1);
    
    return selected;
}

/* Initialize beam search */
beam_search_t beam_search_init(size_t beam_width, uint32_t initial_token) {
    beam_search_t bs = {0};
    bs.beam_width = beam_width;
    bs.num_beams = 1;
    bs.beams = malloc(beam_width * sizeof(beam_t));
    
    bs.beams[0].tokens = malloc(512); /* Max length */
    bs.beams[0].tokens[0] = initial_token;
    bs.beams[0].length = 1;
    bs.beams[0].score = 0.0f;
    
    return bs;
}

/* Beam search step */
uint32_t beam_search_step(float *logits, size_t vocab_size, beam_search_t *beam_search) {
    if (!logits || !beam_search || beam_search->num_beams == 0) return 0;
    
    /* For simplicity, just return best scoring token (not full beam search) */
    /* Full beam search would track multiple hypotheses */
    uint32_t best_token = 0;
    float best_score = logits[0];
    
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > best_score) {
            best_score = logits[i];
            best_token = i;
        }
    }
    
    /* Update best beam */
    if (beam_search->beams[0].length < 100) {
        beam_search->beams[0].tokens[beam_search->beams[0].length] = best_token;
        beam_search->beams[0].length++;
        beam_search->beams[0].score += best_score;
    }
    
    return best_token;
}

/* Get best hypothesis from beam search */
beam_t beam_search_best(const beam_search_t *beam_search) {
    if (!beam_search || beam_search->num_beams == 0) {
        beam_t empty = {0};
        return empty;
    }
    return beam_search->beams[0];
}

/* Free beam search resources */
void beam_search_free(beam_search_t *beam_search) {
    if (!beam_search) return;
    
    for (size_t i = 0; i < beam_search->num_beams; i++) {
        free(beam_search->beams[i].tokens);
    }
    free(beam_search->beams);
}
