#include "byte_pair_encoding.h"
#include "cli/checkpoint.h"
#include "cli/tokenizer.h"
#include "core/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int models_equal(const neural_model_t *a, const neural_model_t *b) {
    size_t bytes = a->total_param_count * sizeof(float);
    size_t history_bytes = a->metrics.history_size * sizeof(float);
    return a->total_param_count == b->total_param_count &&
           a->architecture_flags == b->architecture_flags &&
           memcmp(a->params, b->params, bytes) == 0 &&
           memcmp(a->grads, b->grads, bytes) == 0 &&
           a->adam_m && b->adam_m && a->adam_v && b->adam_v &&
           memcmp(a->adam_m, b->adam_m, bytes) == 0 &&
           memcmp(a->adam_v, b->adam_v, bytes) == 0 &&
           a->training_steps == b->training_steps && a->adam_t == b->adam_t &&
           a->rng_state == b->rng_state && a->current_loss == b->current_loss &&
           a->learning_rate == b->learning_rate && a->dropout_rate == b->dropout_rate &&
           a->metrics.history_size == b->metrics.history_size &&
           memcmp(a->metrics.loss_history, b->metrics.loss_history, history_bytes) == 0 &&
           a->metrics.best_loss == b->metrics.best_loss &&
           a->metrics.worst_loss == b->metrics.worst_loss &&
           a->metrics.avg_loss == b->metrics.avg_loss &&
           a->metrics.learning_rate == b->metrics.learning_rate &&
           a->metrics.steps_without_improvement == b->metrics.steps_without_improvement;
}

int main(void) {
    char directory[128];
    snprintf(directory, sizeof(directory), "/tmp/dranzer_checkpoint_%ld", (long)getpid());
    neural_model_t original = {0};
    neural_model_t resumed = {0};
    bpe_encoder_t *encoder = tokenizer_create_special_encoder(264);
    bpe_encoder_t *loaded_encoder = NULL;
    checkpoint_run_state_t state = {0};
    checkpoint_run_state_t loaded_state = {0};
    char checkpoint_path[1024];
    uint32_t tokens[] = {1, 2, 3, 4, 5, 6};
    int failed = 0;

    if (!encoder || bpe_train(encoder, "banana banana", 13) != BPE_SUCCESS ||
        bpe_encoder_freeze(encoder) != BPE_SUCCESS ||
        model_new_seeded_architecture(&original, 264, 8, 2, 1, 8, 101,
                                      MODEL_ARCH_TIED_EMBEDDINGS |
                                          MODEL_ARCH_ROPE |
                                          MODEL_ARCH_RMSNORM |
                                          MODEL_ARCH_SWIGLU) != MODEL_SUCCESS) {
        fprintf(stderr, "checkpoint fixture initialization failed\n");
        failed = 1;
        goto cleanup;
    }
    original.dropout_rate = 0.25f;
    original.weight_decay = 0.02f;
    original.warmup_steps = 2;
    original.total_steps = 100;
    original.base_lr = 0.005f;
    original.learning_rate = 0.005f;
    original.metrics.learning_rate = 0.005f;
    original.metrics.initial_learning_rate = 0.005f;
    model_seed_rng(&original, 777);

    for (size_t i = 0; i < 8; i++) {
        if (model_train_step(&original, tokens, 7, 6) != MODEL_SUCCESS) {
            failed = 1;
            goto cleanup;
        }
    }

    state.epoch_index = 1;
    state.step_in_epoch = 37;
    state.target_epochs = 4;
    state.input_fingerprint = UINT64_C(0x123456789abcdef0);
    state.input_bytes = 1234;
    state.train_window = 6;
    state.seed = 777;
    state.batch_size = 1;
    state.gradient_accumulation_steps = 3;
    state.shuffle = 1;
    state.checkpoint_interval = 4;
    state.keep_checkpoints = 2;
    strcpy(state.input_file, "data/train.txt");
    strcpy(state.validation_file, "data/validation.txt");
    strcpy(state.tokenizer_path, "model.tokenizer");
    strcpy(state.model_path, "model.pth");
    strcpy(state.checkpoint_dir, "checkpoints");

    if (checkpoint_save(&original, encoder, &state, directory, 2,
                        checkpoint_path, sizeof(checkpoint_path)) != 0 ||
        checkpoint_load(&resumed, &loaded_encoder, &loaded_state, checkpoint_path) != 0 ||
        memcmp(&state, &loaded_state, sizeof(state)) != 0 ||
        !models_equal(&original, &resumed) || !loaded_encoder ||
        loaded_encoder->vocab_size != encoder->vocab_size ||
        !bpe_encoder_has_special_tokens(loaded_encoder) ||
        !bpe_encoder_is_frozen(loaded_encoder)) {
        fprintf(stderr, "checkpoint round-trip changed training state\n");
        failed = 1;
        goto cleanup;
    }

    for (size_t i = 0; i < 12; i++) {
        if (model_train_step(&original, tokens, 7, 6) != MODEL_SUCCESS ||
            model_train_step(&resumed, tokens, 7, 6) != MODEL_SUCCESS) {
            failed = 1;
            goto cleanup;
        }
    }
    if (!models_equal(&original, &resumed)) {
        fprintf(stderr, "resumed trajectory diverged from uninterrupted training\n");
        failed = 1;
        goto cleanup;
    }

    for (size_t i = 0; i < 3; i++) {
        state.step_in_epoch++;
        if (model_train_step(&original, tokens, 7, 6) != MODEL_SUCCESS ||
            checkpoint_save(&original, encoder, &state, directory, 2,
                            NULL, 0) != 0) {
            failed = 1;
            goto cleanup;
        }
    }
    char *paths[8] = {0};
    size_t count = checkpoint_list(directory, paths, 8);
    for (size_t i = 0; i < count; i++) free(paths[i]);
    char latest[1024];
    if (count != 2 || checkpoint_find_latest(directory, latest, sizeof(latest)) != 0) {
        fprintf(stderr, "checkpoint retention/latest selection failed\n");
        failed = 1;
        goto cleanup;
    }

    printf("checkpoint=%s resumed_steps=%u retained=%zu\n",
           checkpoint_path, resumed.training_steps, count);
    printf("\nEXACT CHECKPOINT RESUME CHECK PASSED\n");

cleanup:
    {
        char *paths[16] = {0};
        size_t count = checkpoint_list(directory, paths, 16);
        for (size_t i = 0; i < count; i++) {
            remove(paths[i]);
            free(paths[i]);
        }
        rmdir(directory);
    }
    model_free(&original);
    model_free(&resumed);
    tokenizer_free_encoder(encoder);
    tokenizer_free_encoder(loaded_encoder);
    return failed;
}
