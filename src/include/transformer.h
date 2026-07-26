#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "model_types.h"

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

#endif // TRANSFORMER_H
