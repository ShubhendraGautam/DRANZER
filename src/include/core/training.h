#ifndef TRAINING_H
#define TRAINING_H

#include "core/transformer.h"

/* Train the model on a sequence: forward pass, cross-entropy loss, full
 * backpropagation through every layer (attention, FFN, layer norms,
 * dropout) and the token embeddings, then an optimizer step (SGD or
 * AdamW, per model->optimizer_type) over every parameter that received a
 * gradient. Additive position embeddings are fixed and never trained; RoPE
 * has no trainable position parameters.
 * @param model: The neural model
 * @param token_ids: Input token IDs (context window)
 * @param target_id: Target next token ID
 * @param seq_len: Length of the context window (must be >= 1 and <= model->max_seq_len)
 * @return MODEL_SUCCESS on success
 */
model_errors_t model_train_step(neural_model_t *model,
                                uint32_t *token_ids,
                                uint32_t target_id,
                                size_t seq_len);

/* Add one example's backward pass to model->grads without changing any
 * parameter, optimizer, scheduler, or metric state. Call
 * model_zero_gradients() once at the beginning of an accumulation cycle.
 *
 * Supervises the LAST position only: one forward pass, one target, one
 * gradient signal. Prefer model_accumulate_gradients_all() below, which gets
 * seq_len signals out of the same pass. This form is kept because it is what
 * the gradient checks and the single-step tests are written against, and
 * because it is the reference the all-position path is verified to agree
 * with (tests/core/test_all_position_loss.c). */
model_errors_t model_accumulate_gradients(neural_model_t *model,
                                          uint32_t *token_ids,
                                          uint32_t target_id,
                                          size_t seq_len,
                                          float *out_loss);

/* Same, but supervising EVERY position: targets[i] is the token that should
 * follow position i, and the loss is the mean cross-entropy over the
 * positions that are not LM_HEAD_IGNORE_TARGET (see core/lm_head.h).
 *
 * The causal mask makes this valid without any change to the model - row i
 * cannot have seen token i+1 - so this is the same forward pass yielding
 * seq_len training signals instead of one. The per-layer backward is shared
 * verbatim with the single-target form above; only the head differs.
 *
 * @param out_loss: mean loss over supervised positions; may be NULL.
 * @param out_supervised: how many positions were counted; may be NULL. Zero
 *   means no gradient was contributed at all.
 */
model_errors_t model_accumulate_gradients_all(neural_model_t *model,
                                              uint32_t *token_ids,
                                              const uint32_t *targets,
                                              size_t seq_len,
                                              float *out_loss,
                                              size_t *out_supervised);

/* Mask-aware all-position training. Padding positions are ignored even if
 * their targets are otherwise valid; the general attention mask is applied
 * in both the cached forward activations and attention backward pass. */
model_errors_t model_accumulate_gradients_all_masked(
    neural_model_t *model, uint32_t *token_ids, const uint32_t *targets,
    size_t seq_len, const model_attention_mask_t *mask,
    float *out_loss, size_t *out_supervised);

/* Average the accumulated gradients over sample_count, perform one
 * optimizer update, and record average_loss as one optimizer-step metric. */
model_errors_t model_apply_accumulated_gradients(neural_model_t *model,
                                                  size_t sample_count,
                                                  float average_loss);

/**
 * Get predicted next token
 * @param model: The neural model
 * @param token_ids: Input token IDs
 * @param seq_len: Length of sequence
 * @return Predicted token ID
 */
uint32_t model_predict_next_token(neural_model_t *model,
                                  uint32_t *token_ids,
                                  size_t seq_len);

#endif // TRAINING_H
