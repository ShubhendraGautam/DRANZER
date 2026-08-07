#ifndef LM_HEAD_H
#define LM_HEAD_H

#include "core/model_types.h"

/* The output head, projecting EVERY sequence position to vocabulary logits,
 * and the cross-entropy over all of them.
 *
 * Why this module exists
 * ----------------------
 * The original training path supervised one position per forward pass: the
 * head read cache_hidden[num_layers][seq_len-1] and the loss had a single
 * target. Every other position's hidden state was computed in full - all of
 * attention, all of the FFN - and discarded.
 *
 * That is sound but wasteful by a factor of seq_len. The causal mask in
 * transformer.c already guarantees position i's final hidden state depends
 * only on tokens 0..i, which is exactly the condition that makes supervising
 * every position valid: row i predicts token i+1 without ever having seen
 * it. So the same forward pass yields seq_len training signals instead of
 * one, for the cost of turning three m=1 matmuls into m=seq_len.
 *
 * Nothing about the model changes. This is a change to the head and the
 * loss; attention, the FFN, layer norm, dropout, and the optimizer are
 * untouched, and the per-layer backward in training.c is shared verbatim
 * between the two paths.
 *
 * Target convention
 * -----------------
 * targets[i] is the token that should follow position i. For a contiguous
 * window the caller passes the input shifted left by one, with the token
 * after the window as the final target. Positions set to
 * LM_HEAD_IGNORE_TARGET contribute neither loss nor gradient, which is what
 * padding and (later) general attention masks will need.
 */

/* Target sentinel: this position is not supervised. */
#define LM_HEAD_IGNORE_TARGET UINT32_MAX

/* Project every position to logits, into model->ws_logits_all as a
 * [seq_len x vocab_size] row-major block. Expects model_forward_hidden() to
 * have already run for this same seq_len. */
model_errors_t lm_head_forward_all(neural_model_t *model, size_t seq_len);

/* Mean cross-entropy over supervised positions, from the logits
 * lm_head_forward_all() just wrote.
 *
 * Writes dL/dlogits into model->ws_grad_logits_all, already divided by the
 * number of supervised positions so the caller's gradient is the mean rather
 * than the sum; ignored positions get an all-zero row. Uses the
 * max-subtracted log-sum-exp form in double, matching core/evaluation.c
 * rather than the softmax-then-log the single-target path used.
 *
 * @param out_supervised: receives how many positions were counted; may be
 *   NULL. Zero means every target was LM_HEAD_IGNORE_TARGET or out of
 *   vocabulary, in which case the loss is 0 and the gradient all zero.
 * @return MODEL_INVALID_INPUT on a null argument or bad seq_len.
 */
model_errors_t lm_head_loss_and_grad_all(neural_model_t *model,
                                         const uint32_t *targets,
                                         size_t seq_len,
                                         float *out_loss,
                                         size_t *out_supervised);

/* Backprop the head: accumulate into output_projection_grad and
 * output_bias_grad, and seed model->ws_dhidden_in with dL/d(final hidden)
 * for every position. Overwrites ws_dhidden_in rather than accumulating into
 * it, so the caller need not pre-zero it.
 *
 * Expects lm_head_loss_and_grad_all() to have filled ws_grad_logits_all. */
model_errors_t lm_head_backward_all(neural_model_t *model, size_t seq_len);

#endif // LM_HEAD_H
