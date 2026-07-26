/* Numerical gradient check for the hand-rolled backprop in the model
 * modules (transformer.c/training.c). There's no autodiff to lean on here,
 * so this is the correctness backstop: compares the analytical gradient
 * SGD recovers (grad = (before - after) / lr) against a central-difference
 * numerical gradient (two forward passes + manual cross-entropy), for a
 * representative parameter from each major weight type, across both
 * layers. Forced to plain SGD with no clipping - see the comment on
 * reset_model() for why.
 */
#include "core/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VOCAB 12
#define EMB 4
#define HEADS 2
#define LAYERS 2
#define MAX_SEQ 5
#define SEQ_LEN 3
#define LR 0.01f
#define EPS 1e-3f

static float forward_loss(neural_model_t *model, uint32_t *tokens, size_t seq_len, uint32_t target) {
    float logits[VOCAB];
    model_forward(model, tokens, seq_len, logits);

    float max_l = logits[0];
    for (size_t i = 1; i < model->vocab_size; i++) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < model->vocab_size; i++) {
        sum += expf(logits[i] - max_l);
    }
    float prob_target = expf(logits[target] - max_l) / sum;
    return -logf(fmaxf(prob_target, 1e-7f));
}

static void reset_model(neural_model_t *model) {
    model_free(model);
    srand(42);
    if (model_new(model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        exit(1);
    }
    /* The (original-new_value)/lr recovery trick below only works for
     * plain SGD (param -= lr*grad); model_new defaults to AdamW, whose
     * update magnitude is normalized to roughly sign(grad), not
     * proportional to it. Force SGD + no clipping so this test checks the
     * actual backward-pass gradient, not the optimizer's transform of it. */
    model->optimizer_type = OPTIMIZER_SGD;
    model->grad_clip_norm = 0.0f;
}

static int check_param(const char *name, float *param, uint32_t *tokens, size_t seq_len, uint32_t target,
                        neural_model_t *model) {
    float original = *param;

    *param = original + EPS;
    float loss_plus = forward_loss(model, tokens, seq_len, target);

    *param = original - EPS;
    float loss_minus = forward_loss(model, tokens, seq_len, target);

    *param = original; /* restore before the real training step */

    float numerical_grad = (loss_plus - loss_minus) / (2.0f * EPS);

    model->learning_rate = LR;
    model_train_step(model, tokens, target, seq_len);
    float new_value = *param;
    float analytical_grad = (original - new_value) / LR;

    float diff = fabsf(analytical_grad - numerical_grad);
    float denom = fmaxf(fabsf(analytical_grad), fabsf(numerical_grad));
    float rel_error = (denom > 1e-6f) ? diff / denom : diff;

    int pass = (rel_error < 0.05f) || (diff < 1e-3f);
    printf("%-26s analytical=%+10.6f numerical=%+10.6f rel_err=%.4f  %s\n",
           name, analytical_grad, numerical_grad, rel_error, pass ? "PASS" : "FAIL");
    return pass;
}

int main(void) {
    uint32_t tokens[SEQ_LEN] = {1, 5, 3};
    uint32_t target = 7;
    neural_model_t model = {0};
    int all_pass = 1;

#define CHECK(label, ptr) \
    reset_model(&model); \
    all_pass &= check_param(label, ptr, tokens, SEQ_LEN, target, &model);

    CHECK("layer0.W_q[3]", &model.layers[0].W_q[3]);
    CHECK("layer0.W_k[0]", &model.layers[0].W_k[0]);
    CHECK("layer0.W_v[5]", &model.layers[0].W_v[5]);
    CHECK("layer0.W_o[2]", &model.layers[0].W_o[2]);
    CHECK("layer0.ln_gamma_attn[1]", &model.layers[0].ln_gamma_attn[1]);
    CHECK("layer0.ln_beta_attn[0]", &model.layers[0].ln_beta_attn[0]);
    CHECK("layer0.ln_gamma_ffn[2]", &model.layers[0].ln_gamma_ffn[2]);
    CHECK("layer0.ln_beta_ffn[3]", &model.layers[0].ln_beta_ffn[3]);
    CHECK("layer0.W_ff1[10]", &model.layers[0].W_ff1[10]);
    CHECK("layer0.b_ff1[4]", &model.layers[0].b_ff1[4]);
    CHECK("layer0.W_ff2[7]", &model.layers[0].W_ff2[7]);
    CHECK("layer0.b_ff2[1]", &model.layers[0].b_ff2[1]);
    CHECK("layer1.W_q[1]", &model.layers[1].W_q[1]);
    CHECK("layer1.W_k[6]", &model.layers[1].W_k[6]);
    CHECK("layer1.W_ff2[3]", &model.layers[1].W_ff2[3]);
    CHECK("token_embeddings[tok1]", &model.token_embeddings[tokens[1] * EMB + 2]);
    CHECK("token_embeddings[tok0]", &model.token_embeddings[tokens[0] * EMB + 0]);
    CHECK("output_projection[5]", &model.output_projection[5]);
    CHECK("output_bias[7]", &model.output_bias[7]);

#undef CHECK
    model_free(&model);

    printf("\n%s\n", all_pass ? "ALL GRADIENT CHECKS PASSED" : "SOME GRADIENT CHECKS FAILED");
    return all_pass ? 0 : 1;
}
