/* Side-effect-free next-token evaluation. */

#include "core/evaluation.h"
#include "core/transformer.h"
#include <math.h>

model_errors_t model_evaluate_step(neural_model_t *model,
                                   uint32_t *token_ids,
                                   uint32_t target_id,
                                   size_t seq_len,
                                   double *out_loss) {
    if (!model || !token_ids || !out_loss || target_id >= model->vocab_size ||
        seq_len == 0 || seq_len > model->max_seq_len) {
        return MODEL_INVALID_INPUT;
    }

    int was_training = model->is_training;
    model->is_training = 0;
    model_errors_t rc = model_forward(model, token_ids, seq_len, model->ws_logits);
    model->is_training = was_training;
    if (rc != MODEL_SUCCESS) return rc;

    float max_logit = model->ws_logits[0];
    for (size_t i = 1; i < model->vocab_size; i++) {
        if (model->ws_logits[i] > max_logit) max_logit = model->ws_logits[i];
    }

    double exp_sum = 0.0;
    for (size_t i = 0; i < model->vocab_size; i++) {
        exp_sum += exp((double)model->ws_logits[i] - (double)max_logit);
    }
    *out_loss = log(exp_sum) + (double)max_logit - (double)model->ws_logits[target_id];
    return MODEL_SUCCESS;
}
