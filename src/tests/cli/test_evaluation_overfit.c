#include "byte_pair_encoding.h"
#include "cli/evaluation.h"
#include "cli/tokenizer.h"
#include "core/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    const char *training_text = "abababababababababababababababab";
    const char *held_out_text = "abababababababab";
    char held_out_path[128];
    snprintf(held_out_path, sizeof(held_out_path),
             "/tmp/dranzer_held_out_%ld.txt", (long)getpid());

    FILE *file = fopen(held_out_path, "wb");
    size_t written = file ? fwrite(held_out_text, 1, strlen(held_out_text), file) : 0;
    int close_rc = file ? fclose(file) : -1;
    if (!file || written != strlen(held_out_text) || close_rc != 0) {
        remove(held_out_path);
        fprintf(stderr, "failed to create held-out fixture\n");
        return 1;
    }

    bpe_encoder_t *encoder = tokenizer_create_encoder(256);
    bpe_tokens_t training_tokens = {0};
    neural_model_t model = {0};
    evaluation_report_t before = {0};
    evaluation_report_t after = {0};
    int failed = 0;

    if (!encoder || bpe_encoder_freeze(encoder) != BPE_SUCCESS ||
        bpe_encode(encoder, training_text, strlen(training_text), &training_tokens) != BPE_SUCCESS) {
        fprintf(stderr, "failed to prepare tokenizer fixture\n");
        failed = 1;
        goto cleanup;
    }

    srand(23);
    if (model_new(&model, 256, 8, 2, 1, 8) != MODEL_SUCCESS ||
        evaluate_corpus_file(&model, encoder, held_out_path, 4, &before) !=
            EVALUATION_SUCCESS) {
        fprintf(stderr, "failed to evaluate untrained model\n");
        failed = 1;
        goto cleanup;
    }

    model.learning_rate = 0.01f;
    model.metrics.learning_rate = 0.01f;
    model.metrics.initial_learning_rate = 0.01f;
    model.base_lr = 0.01f;
    model.weight_decay = 0.0f;
    for (size_t epoch = 0; epoch < 30; epoch++) {
        for (size_t i = 0; i + 1 < training_tokens.token_count; i++) {
            size_t context_len = i + 1 < 4 ? i + 1 : 4;
            uint32_t *context = &training_tokens.token_ids[i + 1 - context_len];
            if (model_train_step(&model, context, training_tokens.token_ids[i + 1],
                                 context_len) != MODEL_SUCCESS) {
                fprintf(stderr, "tiny-corpus training failed\n");
                failed = 1;
                goto cleanup;
            }
        }
    }

    size_t param_bytes = model.total_param_count * sizeof(float);
    float *params_before_eval = malloc(param_bytes);
    if (!params_before_eval) {
        failed = 1;
        goto cleanup;
    }
    memcpy(params_before_eval, model.params, param_bytes);
    uint32_t steps_before_eval = model.training_steps;

    if (evaluate_corpus_file(&model, encoder, held_out_path, 4, &after) !=
            EVALUATION_SUCCESS ||
        after.token_count != strlen(held_out_text) ||
        after.corpus_bytes != strlen(held_out_text) ||
        after.corpus_fingerprint != before.corpus_fingerprint ||
        after.prediction_count + 1 != after.token_count ||
        after.average_cross_entropy >= before.average_cross_entropy - 0.5 ||
        fabs(after.perplexity - exp(after.average_cross_entropy)) > 1e-10 ||
        model.training_steps != steps_before_eval ||
        memcmp(params_before_eval, model.params, param_bytes) != 0) {
        fprintf(stderr, "held-out evaluation/overfit regression failed: %.6f -> %.6f\n",
                before.average_cross_entropy, after.average_cross_entropy);
        failed = 1;
    } else {
        printf("held-out cross-entropy: %.6f -> %.6f, perplexity: %.6f\n",
               before.average_cross_entropy, after.average_cross_entropy, after.perplexity);
        printf("\nEVALUATION OVERFIT CHECK PASSED\n");
    }
    free(params_before_eval);

cleanup:
    model_free(&model);
    bpe_tokens_free(&training_tokens);
    tokenizer_free_encoder(encoder);
    remove(held_out_path);
    return failed;
}
