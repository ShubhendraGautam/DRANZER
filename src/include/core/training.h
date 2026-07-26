#ifndef TRAINING_H
#define TRAINING_H

#include "model_types.h"

/* Train the model on a sequence: forward pass, cross-entropy loss, full
 * backpropagation through every layer (attention, FFN, layer norms,
 * dropout) and the token embeddings, then an optimizer step (SGD or
 * AdamW, per model->optimizer_type) over every parameter that received a
 * gradient. position_embeddings are fixed and never trained.
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
