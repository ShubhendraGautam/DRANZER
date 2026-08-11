/* Padding and arbitrary edge masks layered over causal self-attention.
 * Exercises inference selection, sparse/all-empty attention rows, and the
 * cached-mask backward path used by variable-length training batches. */

#include "common/fp_bits.h"
#include "core/lm_head.h"
#include "core/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 19
#define EMBEDDING 8
#define HEADS 2
#define LAYERS 2
#define MAX_SEQUENCE 6

static int close_array(const float *left, const float *right, size_t count,
                       float tolerance) {
    for (size_t i = 0; i < count; i++) {
        if (fabsf(left[i] - right[i]) > tolerance) return 0;
    }
    return 1;
}

static int finite_array(const float *values, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (!dranzer_float_is_finite(values[i])) return 0;
    }
    return 1;
}

int main(void) {
    neural_model_t model = {0};
    if (model_new_seeded(&model, VOCAB, EMBEDDING, HEADS, LAYERS,
                         MAX_SEQUENCE, 901) != MODEL_SUCCESS) {
        fprintf(stderr, "masked-attention fixture allocation failed\n");
        return 1;
    }
    model.dropout_rate = 0.0f;
    int failed = 0;
    uint32_t tokens[MAX_SEQUENCE] = {2, 7, 4, 11, 13, 17};
    float short_logits[VOCAB], masked_logits[VOCAB], changed_logits[VOCAB];

    if (model_forward(&model, tokens, 3, short_logits) != MODEL_SUCCESS) {
        fprintf(stderr, "short reference forward failed\n");
        failed = 1;
        goto cleanup;
    }

    uint8_t padding[MAX_SEQUENCE] = {1, 1, 1, 0, 0, 0};
    model_attention_mask_t padded = {.padding_mask = padding};
    if (model_forward_masked(&model, tokens, MAX_SEQUENCE, &padded,
                             masked_logits) != MODEL_SUCCESS ||
        !close_array(short_logits, masked_logits, VOCAB, 1e-6f)) {
        fprintf(stderr, "right padding changed last-real-token logits\n");
        failed = 1;
    }
    uint32_t changed_padding[MAX_SEQUENCE] = {2, 7, 4, 1, 3, 5};
    if (model_forward_masked(&model, changed_padding, MAX_SEQUENCE, &padded,
                             changed_logits) != MODEL_SUCCESS ||
        !close_array(masked_logits, changed_logits, VOCAB, 1e-6f)) {
        fprintf(stderr, "padding token IDs influenced masked inference\n");
        failed = 1;
    }

    /* An explicit all-ones matrix must reproduce ordinary causality, and its
     * future half must remain zero even though the caller marked it allowed. */
    uint8_t all_edges[MAX_SEQUENCE * MAX_SEQUENCE];
    memset(all_edges, 1, sizeof(all_edges));
    model_attention_mask_t general = {.attention_mask = all_edges};
    if (model_forward(&model, tokens, MAX_SEQUENCE, short_logits) !=
            MODEL_SUCCESS ||
        model_forward_masked(&model, tokens, MAX_SEQUENCE, &general,
                             masked_logits) != MODEL_SUCCESS ||
        !close_array(short_logits, masked_logits, VOCAB, 1e-6f)) {
        fprintf(stderr, "all-allowed general mask changed causal forward\n");
        failed = 1;
    }
    for (size_t h = 0; h < HEADS; h++) {
        const float *probs = model.cache_probs[0] +
                             h * MAX_SEQUENCE * MAX_SEQUENCE;
        for (size_t i = 0; i < MAX_SEQUENCE; i++) {
            for (size_t j = i + 1; j < MAX_SEQUENCE; j++) {
                if (probs[i * MAX_SEQUENCE + j] != 0.0f) failed = 1;
            }
        }
    }

    all_edges[(MAX_SEQUENCE - 1) * MAX_SEQUENCE + 1] = 0;
    if (model_forward_masked(&model, tokens, MAX_SEQUENCE, &general,
                             masked_logits) != MODEL_SUCCESS) {
        fprintf(stderr, "general edge-mask forward failed\n");
        failed = 1;
    }
    for (size_t h = 0; h < HEADS; h++) {
        const float *row = model.cache_probs[0] +
            h * MAX_SEQUENCE * MAX_SEQUENCE +
            (MAX_SEQUENCE - 1) * MAX_SEQUENCE;
        if (row[1] != 0.0f) {
            fprintf(stderr, "blocked past edge retained probability mass\n");
            failed = 1;
        }
    }

    /* A row with no legal key is a supported sparse-mask case. Its attention
     * probabilities/context are exact zero and the residual path stays finite. */
    memset(all_edges, 1, sizeof(all_edges));
    const size_t empty_row = MAX_SEQUENCE - 1;
    memset(&all_edges[empty_row * MAX_SEQUENCE], 0, MAX_SEQUENCE);
    if (model_forward_masked(&model, tokens, MAX_SEQUENCE, &general,
                             masked_logits) != MODEL_SUCCESS ||
        !finite_array(masked_logits, VOCAB)) {
        fprintf(stderr, "fully masked attention row produced invalid logits\n");
        failed = 1;
    }
    for (size_t h = 0; h < HEADS; h++) {
        const float *row = model.cache_probs[0] +
            h * MAX_SEQUENCE * MAX_SEQUENCE + empty_row * MAX_SEQUENCE;
        for (size_t j = 0; j < MAX_SEQUENCE; j++) {
            if (row[j] != 0.0f) {
                fprintf(stderr, "fully masked row gained probability mass\n");
                failed = 1;
                break;
            }
        }
    }

    uint8_t no_tokens[MAX_SEQUENCE] = {0};
    model_attention_mask_t all_padding = {.padding_mask = no_tokens};
    if (model_forward_masked(&model, tokens, MAX_SEQUENCE, &all_padding,
                             masked_logits) != MODEL_INVALID_INPUT) {
        fprintf(stderr, "all-padding inference did not reject missing output position\n");
        failed = 1;
    }

    /* Padded all-position training must equal the real prefix even when the
     * caller supplies valid-looking targets for padding: padding owns the
     * supervision decision and the masked backward must add no extra gradient. */
    uint32_t short_targets[3] = {7, 4, 11};
    uint32_t padded_targets[MAX_SEQUENCE] = {7, 4, 11, 6, 8, 10};
    model_zero_gradients(&model);
    float short_loss = 0.0f;
    size_t short_supervised = 0;
    if (model_accumulate_gradients_all(&model, tokens, short_targets, 3,
                                       &short_loss,
                                       &short_supervised) != MODEL_SUCCESS) {
        fprintf(stderr, "short reference backward failed\n");
        failed = 1;
        goto cleanup;
    }
    float *short_grads = malloc(model.total_param_count * sizeof(*short_grads));
    if (!short_grads) {
        failed = 1;
        goto cleanup;
    }
    memcpy(short_grads, model.grads,
           model.total_param_count * sizeof(*short_grads));

    model_zero_gradients(&model);
    float padded_loss = 0.0f;
    size_t padded_supervised = 0;
    if (model_accumulate_gradients_all_masked(
            &model, tokens, padded_targets, MAX_SEQUENCE, &padded,
            &padded_loss, &padded_supervised) != MODEL_SUCCESS ||
        short_supervised != 3 || padded_supervised != 3 ||
        fabsf(short_loss - padded_loss) > 1e-6f ||
        !close_array(short_grads, model.grads, model.total_param_count,
                     2e-5f)) {
        fprintf(stderr, "padded training changed prefix loss or gradients\n");
        failed = 1;
    }
    free(short_grads);

cleanup:
    model_free(&model);
    printf("%s\n", failed ? "ATTENTION MASK CHECK FAILED"
                           : "ATTENTION MASK CHECK PASSED");
    return failed ? 1 : 0;
}
