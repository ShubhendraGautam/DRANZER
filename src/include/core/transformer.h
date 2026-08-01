#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "core/model_types.h"

/* Per-generation key/value cache and scratch space for incremental
 * autoregressive decoding. One instance belongs to one model and one
 * sequence. Keys and values are stored per layer for every position that
 * has already been consumed. */
typedef struct {
    const neural_model_t *model;
    size_t length;
    size_t capacity;
    size_t num_layers;
    size_t num_heads;
    size_t embedding_dim;
    float **keys;
    float **values;
    float *hidden;
    float *attn_norm;
    float *query;
    float *attn_concat;
    float *attn_raw;
    float *ff_hidden;
    float *ff_raw;
    float *scores;
} model_kv_cache_t;

/* Multi-head self-attention forward pass for layer l. Reads
 * model->cache_hidden[l] as input, writes Q/K/V/probs/concat into that
 * layer's cache entries (needed by backward), and writes the final
 * (post-W_o, pre-residual) result into model->ws_fwd_attn_raw. Causally
 * masked: position i only attends to j <= i. */
void multihead_attention_forward(neural_model_t *model, size_t l, size_t seq_len);

/* Backprop through multihead_attention_forward for layer l.
 * dL_dattn_raw: incoming gradient w.r.t. the attention block's raw output
 *   (post-W_o, pre-residual, pre-dropout) - i.e. dL/d(model->ws_fwd_attn_raw).
 * dL_dhidden_accum: caller-owned buffer that this function ACCUMULATES
 *   into (+=) - the caller must have already seeded it with the
 *   residual-branch contribution to dL/dhidden[l] before calling.
 * Accumulates into layer->W_q_grad/W_k_grad/W_v_grad/W_o_grad - caller
 * must have zeroed them for a fresh gradient (model_zero_gradients does). */
void multihead_attention_backward(neural_model_t *model, size_t l, size_t seq_len,
                                   float *dL_dattn_raw, float *dL_dhidden_accum);

/* Forward pass through the full stack of transformer layers plus the
 * output head. Populates the activation cache as a side effect (needed by
 * model_train_step's backward pass) but this is harmless for standalone
 * inference use (infer/generate) - just some extra writes into memory
 * nothing else reads afterward.
 * @param model: The neural model
 * @param token_ids: Input token IDs
 * @param seq_len: Length of sequence
 * @param output_logits: Output logits for next token prediction (vocab_size)
 * @return MODEL_SUCCESS on success
 */
model_errors_t model_forward(neural_model_t *model,
                              uint32_t *token_ids,
                              size_t seq_len,
                              float *output_logits);

/* Allocate/reset/free an incremental decoding cache. The cache is tied to
 * the model passed at initialization and must be reset before starting a
 * new sequence. */
model_errors_t model_kv_cache_init(model_kv_cache_t *cache, const neural_model_t *model);
void model_kv_cache_reset(model_kv_cache_t *cache);
void model_kv_cache_free(model_kv_cache_t *cache);

/* Consume one token at cache->length, append its per-layer keys/values,
 * and return logits for the following token. This is inference-only:
 * dropout is intentionally disabled and no backward activations are
 * populated. */
model_errors_t model_forward_token(neural_model_t *model, model_kv_cache_t *cache,
                                   uint32_t token_id, float *output_logits);

#endif // TRANSFORMER_H
