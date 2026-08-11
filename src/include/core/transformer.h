#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "core/model_types.h"

/* Per-generation key/value cache and scratch space for incremental
 * autoregressive decoding. One instance belongs to one model and one
 * sequence. Keys and values are stored per layer for the newest `capacity`
 * consumed positions. */
typedef struct {
    const neural_model_t *model;
    size_t length;       /* retained tokens, never greater than capacity */
    size_t capacity;
    size_t start;        /* physical slot containing the oldest token */
    size_t total_tokens; /* absolute tokens consumed since reset */
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
    float *position_embedding;
} model_kv_cache_t;

/* Optional restrictions layered on top of causal self-attention.
 *
 * padding_mask has seq_len entries: zero marks padding, nonzero marks a real
 * token. Padded query rows are kept at zero and padded key/value columns are
 * never attended to.
 *
 * attention_mask is a packed row-major seq_len x seq_len boolean matrix:
 * zero removes an edge and nonzero permits it. It cannot override causality,
 * so entries with key position j > query position i remain masked. Either
 * pointer may be NULL. The arrays only need to remain valid for the call;
 * the effective mask is copied into the model's activation cache for
 * backward. */
typedef struct {
    const uint8_t *padding_mask;
    const uint8_t *attention_mask;
} model_attention_mask_t;

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

/* Embeddings, positional encoding, and every transformer layer, leaving the
 * final hidden states in model->cache_hidden[model->num_layers] as a
 * [seq_len x embedding_dim] row-major block. Stops short of the output head
 * so that callers can project either the last position (model_forward, for
 * inference) or every position (core/lm_head.c, for training) from one pass.
 * Populates the whole activation cache, which is what makes the backward
 * pass possible without further allocation.
 * @return MODEL_SUCCESS, or MODEL_INVALID_INPUT for a null model/tokens or a
 *   seq_len of zero or above model->max_seq_len.
 */
model_errors_t model_forward_hidden(neural_model_t *model,
                                    uint32_t *token_ids,
                                    size_t seq_len);

/* Mask-aware form of model_forward_hidden(). A fully masked attention row is
 * valid and produces a zero attention context, avoiding infinity sentinels
 * under the project's finite-math build. */
model_errors_t model_forward_hidden_masked(
    neural_model_t *model, uint32_t *token_ids, size_t seq_len,
    const model_attention_mask_t *mask);

/* Forward pass through the full stack of transformer layers plus the
 * output head, projecting the LAST position only. Populates the activation
 * cache as a side effect (needed by model_train_step's backward pass) but
 * this is harmless for standalone inference use (infer/generate) - just some
 * extra writes into memory nothing else reads afterward.
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

/* Mask-aware full forward pass. With a padding mask, logits are projected
 * from the last non-padding position; an all-padding input is invalid. */
model_errors_t model_forward_masked(neural_model_t *model,
                                    uint32_t *token_ids,
                                    size_t seq_len,
                                    const model_attention_mask_t *mask,
                                    float *output_logits);

/* Allocate/reset/free an incremental decoding cache. The cache is tied to
 * the model passed at initialization and must be reset before starting a
 * new sequence. */
model_errors_t model_kv_cache_init(model_kv_cache_t *cache, const neural_model_t *model);
model_errors_t model_kv_cache_init_with_capacity(model_kv_cache_t *cache,
                                                  const neural_model_t *model,
                                                  size_t capacity);
void model_kv_cache_reset(model_kv_cache_t *cache);
void model_kv_cache_free(model_kv_cache_t *cache);

/* Consume one token at the next absolute position and return logits for the
 * following token. Once capacity is full, the oldest per-layer key/value row
 * is evicted from a ring buffer. Sinusoidal positions continue absolutely
 * beyond the trained window. This is inference-only: dropout is disabled and
 * no backward activations are populated. */
model_errors_t model_forward_token(neural_model_t *model, model_kv_cache_t *cache,
                                   uint32_t token_id, float *output_logits);

#endif // TRANSFORMER_H
