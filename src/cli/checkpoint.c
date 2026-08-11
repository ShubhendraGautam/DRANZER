#include "cli/checkpoint.h"
#include "common/debug.h"
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECKPOINT_MAGIC "DRNZCKP1"
#define CHECKPOINT_END "DRNZEND1"
#define CHECKPOINT_MAGIC_SIZE 8
/* 4: added architecture_flags after model dimensions. Bumped rather than
 * treating the shorter version-3 layout as if it had default flags.
 * 3: added train_stride to the run scalars. Bumped rather than appended
 * so a version-2 checkpoint is rejected outright instead of resuming with a
 * garbage stride - it was written by a binary that supervised one position
 * per window, so its step_in_epoch cursor counts targets where this one
 * counts windows, and replaying it would silently train on the wrong data. */
#define CHECKPOINT_VERSION UINT32_C(4)

static int write_exact(FILE *file, const void *data, size_t size) {
    return fwrite(data, 1, size, file) == size;
}

static int read_exact(FILE *file, void *data, size_t size) {
    return fread(data, 1, size, file) == size;
}

static int write_string(FILE *file, const char *value) {
    size_t length = strlen(value);
    if (length > UINT32_MAX) return 0;
    uint32_t stored_length = (uint32_t)length;
    return write_exact(file, &stored_length, sizeof(stored_length)) &&
           write_exact(file, value, length);
}

static int read_string(FILE *file, char *value, size_t capacity) {
    uint32_t length = 0;
    if (!read_exact(file, &length, sizeof(length)) || (size_t)length >= capacity) return 0;
    if (!read_exact(file, value, length)) return 0;
    value[length] = '\0';
    return 1;
}

int checkpoint_parse_metadata(const char *filename, checkpoint_metadata_t *out_metadata) {
    if (!filename || !out_metadata) return -1;
    memset(out_metadata, 0, sizeof(*out_metadata));
    int consumed = 0;
    if (sscanf(filename, "checkpoint_epoch_%u_step_%u.ckpt%n",
               &out_metadata->epoch, &out_metadata->training_step, &consumed) != 2 ||
        filename[consumed] != '\0') {
        return -1;
    }
    snprintf(out_metadata->filepath, sizeof(out_metadata->filepath), "%s", filename);
    return 0;
}

static int checkpoint_make_path(char *buffer, size_t size, const char *directory,
                                uint32_t epoch, uint32_t step) {
    int written = snprintf(buffer, size, "%s/checkpoint_epoch_%u_step_%u.ckpt",
                           directory, epoch, step);
    return written >= 0 && (size_t)written < size;
}

static int write_checkpoint(FILE *file, const neural_model_t *model,
                            const bpe_encoder_t *encoder,
                            const checkpoint_run_state_t *state) {
    uint32_t version = CHECKPOINT_VERSION;
    uint64_t dims[6] = {model->vocab_size, model->embedding_dim, model->num_heads,
                        model->num_layers, model->max_seq_len, model->total_param_count};
    uint32_t optimizer = (uint32_t)model->optimizer_type;
    uint32_t is_training = (uint32_t)model->is_training;
    uint32_t has_adam = model->adam_m && model->adam_v;
    uint64_t history_size = model->metrics.history_size;
    uint64_t run_scalars[5] = {state->step_in_epoch, state->input_fingerprint,
                               state->input_bytes, state->train_window,
                               state->train_stride};
    int32_t run_ints[4] = {state->batch_size, state->checkpoint_interval,
                           state->keep_checkpoints, state->use_gpu};
    int32_t batching_ints[2] = {state->gradient_accumulation_steps, state->shuffle};

    int ok = write_exact(file, CHECKPOINT_MAGIC, CHECKPOINT_MAGIC_SIZE) &&
             write_exact(file, &version, sizeof(version)) &&
             write_exact(file, &state->epoch_index, sizeof(state->epoch_index)) &&
             write_exact(file, &state->target_epochs, sizeof(state->target_epochs)) &&
             write_exact(file, &state->seed, sizeof(state->seed)) &&
             write_exact(file, run_scalars, sizeof(run_scalars)) &&
             write_exact(file, run_ints, sizeof(run_ints)) &&
             write_exact(file, batching_ints, sizeof(batching_ints)) &&
             write_string(file, state->input_file) &&
             write_string(file, state->validation_file) &&
             write_string(file, state->tokenizer_path) &&
             write_string(file, state->model_path) &&
             write_string(file, state->checkpoint_dir) &&
             write_exact(file, dims, sizeof(dims)) &&
             write_exact(file, &model->architecture_flags,
                         sizeof(model->architecture_flags)) &&
             write_exact(file, &model->training_steps, sizeof(model->training_steps)) &&
             write_exact(file, &model->current_loss, sizeof(model->current_loss)) &&
             write_exact(file, &optimizer, sizeof(optimizer)) &&
             write_exact(file, &model->learning_rate, sizeof(model->learning_rate)) &&
             write_exact(file, &model->adam_beta1, sizeof(model->adam_beta1)) &&
             write_exact(file, &model->adam_beta2, sizeof(model->adam_beta2)) &&
             write_exact(file, &model->adam_eps, sizeof(model->adam_eps)) &&
             write_exact(file, &model->weight_decay, sizeof(model->weight_decay)) &&
             write_exact(file, &model->grad_clip_norm, sizeof(model->grad_clip_norm)) &&
             write_exact(file, &model->adam_t, sizeof(model->adam_t)) &&
             write_exact(file, &model->warmup_steps, sizeof(model->warmup_steps)) &&
             write_exact(file, &model->total_steps, sizeof(model->total_steps)) &&
             write_exact(file, &model->base_lr, sizeof(model->base_lr)) &&
             write_exact(file, &model->min_lr, sizeof(model->min_lr)) &&
             write_exact(file, &model->dropout_rate, sizeof(model->dropout_rate)) &&
             write_exact(file, &is_training, sizeof(is_training)) &&
             write_exact(file, &model->rng_state, sizeof(model->rng_state)) &&
             write_exact(file, model->params, model->total_param_count * sizeof(float)) &&
             write_exact(file, model->grads, model->total_param_count * sizeof(float)) &&
             write_exact(file, &has_adam, sizeof(has_adam));

    if (ok && has_adam) {
        ok = write_exact(file, model->adam_m, model->total_param_count * sizeof(float)) &&
             write_exact(file, model->adam_v, model->total_param_count * sizeof(float));
    }
    ok = ok && write_exact(file, &history_size, sizeof(history_size)) &&
         write_exact(file, model->metrics.loss_history,
                     model->metrics.history_size * sizeof(float)) &&
         write_exact(file, &model->metrics.best_loss, sizeof(float)) &&
         write_exact(file, &model->metrics.worst_loss, sizeof(float)) &&
         write_exact(file, &model->metrics.avg_loss, sizeof(float)) &&
         write_exact(file, &model->metrics.learning_rate, sizeof(float)) &&
         write_exact(file, &model->metrics.initial_learning_rate, sizeof(float)) &&
         write_exact(file, &model->metrics.steps_without_improvement,
                     sizeof(model->metrics.steps_without_improvement));
    if (ok) ok = bpe_encoder_write(encoder, file) == BPE_SUCCESS;
    return ok && write_exact(file, CHECKPOINT_END, CHECKPOINT_MAGIC_SIZE);
}

static int metadata_compare_desc(const void *left, const void *right) {
    const checkpoint_metadata_t *a = left;
    const checkpoint_metadata_t *b = right;
    if (a->training_step != b->training_step)
        return a->training_step < b->training_step ? 1 : -1;
    if (a->epoch != b->epoch) return a->epoch < b->epoch ? 1 : -1;
    return 0;
}

static void checkpoint_prune(const char *directory, size_t keep_last) {
    if (keep_last == 0) return;
    DIR *dir = opendir(directory);
    if (!dir) return;
    checkpoint_metadata_t *items = NULL;
    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        checkpoint_metadata_t metadata;
        if (checkpoint_parse_metadata(entry->d_name, &metadata) != 0) continue;
        checkpoint_metadata_t *grown = realloc(items, (count + 1) * sizeof(*items));
        if (!grown) break;
        items = grown;
        items[count] = metadata;
        snprintf(items[count].filepath, sizeof(items[count].filepath), "%s/%s",
                 directory, entry->d_name);
        count++;
    }
    closedir(dir);
    qsort(items, count, sizeof(*items), metadata_compare_desc);
    for (size_t i = keep_last; i < count; i++) remove(items[i].filepath);
    free(items);
}

int checkpoint_save(const neural_model_t *model, const bpe_encoder_t *encoder,
                    const checkpoint_run_state_t *run_state,
                    const char *checkpoint_dir, size_t keep_last,
                    char *out_path, size_t out_path_size) {
    if (!model || !encoder || !run_state || !checkpoint_dir || !checkpoint_dir[0]) return -1;
    if (mkdir(checkpoint_dir, 0755) != 0 && errno != EEXIST) return -1;

    char final_path[2048];
    char temp_path[2080];
    if (!checkpoint_make_path(final_path, sizeof(final_path), checkpoint_dir,
                              run_state->epoch_index, model->training_steps)) return -1;
    int temp_written = snprintf(temp_path, sizeof(temp_path), "%s.tmp.%ld",
                                final_path, (long)getpid());
    if (temp_written < 0 || (size_t)temp_written >= sizeof(temp_path)) return -1;

    FILE *file = fopen(temp_path, "wb");
    if (!file) return -1;
    int ok = write_checkpoint(file, model, encoder, run_state);
    if (ok) ok = fflush(file) == 0;
    if (ok) ok = fsync(fileno(file)) == 0;
    if (fclose(file) != 0) ok = 0;
    if (!ok || rename(temp_path, final_path) != 0) {
        remove(temp_path);
        return -1;
    }

    checkpoint_prune(checkpoint_dir, keep_last);
    if (out_path && out_path_size) {
        int written = snprintf(out_path, out_path_size, "%s", final_path);
        if (written < 0 || (size_t)written >= out_path_size) return -1;
    }
    DEBUG_PRINT("Checkpoint saved atomically: %s\n", final_path);
    return 0;
}

static int read_checkpoint(FILE *file, neural_model_t *model, bpe_encoder_t **out_encoder,
                           checkpoint_run_state_t *state) {
    char magic[CHECKPOINT_MAGIC_SIZE];
    uint32_t version = 0;
    uint64_t run_scalars[5];
    int32_t run_ints[4];
    int32_t batching_ints[2];
    uint64_t dims[6];
    uint32_t architecture_flags = 0;
    uint32_t optimizer = 0, is_training = 0, has_adam = 0;
    uint64_t history_size = 0;

    if (!read_exact(file, magic, sizeof(magic)) ||
        memcmp(magic, CHECKPOINT_MAGIC, sizeof(magic)) != 0 ||
        !read_exact(file, &version, sizeof(version)) || version != CHECKPOINT_VERSION ||
        !read_exact(file, &state->epoch_index, sizeof(state->epoch_index)) ||
        !read_exact(file, &state->target_epochs, sizeof(state->target_epochs)) ||
        !read_exact(file, &state->seed, sizeof(state->seed)) ||
        !read_exact(file, run_scalars, sizeof(run_scalars)) ||
        !read_exact(file, run_ints, sizeof(run_ints)) ||
        !read_exact(file, batching_ints, sizeof(batching_ints)) ||
        !read_string(file, state->input_file, sizeof(state->input_file)) ||
        !read_string(file, state->validation_file, sizeof(state->validation_file)) ||
        !read_string(file, state->tokenizer_path, sizeof(state->tokenizer_path)) ||
        !read_string(file, state->model_path, sizeof(state->model_path)) ||
        !read_string(file, state->checkpoint_dir, sizeof(state->checkpoint_dir)) ||
        !read_exact(file, dims, sizeof(dims)) ||
        !read_exact(file, &architecture_flags, sizeof(architecture_flags))) return 0;

    state->step_in_epoch = run_scalars[0];
    state->input_fingerprint = run_scalars[1];
    state->input_bytes = run_scalars[2];
    state->train_window = (size_t)run_scalars[3];
    state->train_stride = (size_t)run_scalars[4];
    state->batch_size = run_ints[0];
    state->gradient_accumulation_steps = batching_ints[0];
    state->shuffle = batching_ints[1];
    state->checkpoint_interval = run_ints[1];
    state->keep_checkpoints = run_ints[2];
    state->use_gpu = run_ints[3];
    if (state->batch_size <= 0 || state->gradient_accumulation_steps <= 0 ||
        state->shuffle < 0 || state->shuffle > 1 ||
        state->use_gpu < 0 || state->use_gpu > 1) return 0;
    if (dims[0] > SIZE_MAX || dims[1] > SIZE_MAX || dims[2] > SIZE_MAX ||
        dims[3] > SIZE_MAX || dims[4] > SIZE_MAX || dims[5] > SIZE_MAX ||
        model_new_seeded_architecture(
            model, (size_t)dims[0], (size_t)dims[1], (size_t)dims[2],
            (size_t)dims[3], (size_t)dims[4], MODEL_DEFAULT_SEED,
            architecture_flags) != MODEL_SUCCESS ||
        model->total_param_count != (size_t)dims[5]) return 0;

    if (!read_exact(file, &model->training_steps, sizeof(model->training_steps)) ||
        !read_exact(file, &model->current_loss, sizeof(model->current_loss)) ||
        !read_exact(file, &optimizer, sizeof(optimizer)) || optimizer > OPTIMIZER_ADAM ||
        !read_exact(file, &model->learning_rate, sizeof(model->learning_rate)) ||
        !read_exact(file, &model->adam_beta1, sizeof(model->adam_beta1)) ||
        !read_exact(file, &model->adam_beta2, sizeof(model->adam_beta2)) ||
        !read_exact(file, &model->adam_eps, sizeof(model->adam_eps)) ||
        !read_exact(file, &model->weight_decay, sizeof(model->weight_decay)) ||
        !read_exact(file, &model->grad_clip_norm, sizeof(model->grad_clip_norm)) ||
        !read_exact(file, &model->adam_t, sizeof(model->adam_t)) ||
        !read_exact(file, &model->warmup_steps, sizeof(model->warmup_steps)) ||
        !read_exact(file, &model->total_steps, sizeof(model->total_steps)) ||
        !read_exact(file, &model->base_lr, sizeof(model->base_lr)) ||
        !read_exact(file, &model->min_lr, sizeof(model->min_lr)) ||
        !read_exact(file, &model->dropout_rate, sizeof(model->dropout_rate)) ||
        !read_exact(file, &is_training, sizeof(is_training)) || is_training > 1 ||
        !read_exact(file, &model->rng_state, sizeof(model->rng_state)) ||
        !read_exact(file, model->params, model->total_param_count * sizeof(float)) ||
        !read_exact(file, model->grads, model->total_param_count * sizeof(float)) ||
        !read_exact(file, &has_adam, sizeof(has_adam)) || has_adam > 1) return 0;
    model->optimizer_type = (optimizer_type_t)optimizer;
    model->is_training = (int)is_training;

    if (has_adam) {
        model->adam_m = malloc(model->total_param_count * sizeof(float));
        model->adam_v = malloc(model->total_param_count * sizeof(float));
        if (!model->adam_m || !model->adam_v ||
            !read_exact(file, model->adam_m, model->total_param_count * sizeof(float)) ||
            !read_exact(file, model->adam_v, model->total_param_count * sizeof(float))) return 0;
    }
    if (!read_exact(file, &history_size, sizeof(history_size)) ||
        history_size > model->metrics.history_capacity ||
        !read_exact(file, model->metrics.loss_history, history_size * sizeof(float)) ||
        !read_exact(file, &model->metrics.best_loss, sizeof(float)) ||
        !read_exact(file, &model->metrics.worst_loss, sizeof(float)) ||
        !read_exact(file, &model->metrics.avg_loss, sizeof(float)) ||
        !read_exact(file, &model->metrics.learning_rate, sizeof(float)) ||
        !read_exact(file, &model->metrics.initial_learning_rate, sizeof(float)) ||
        !read_exact(file, &model->metrics.steps_without_improvement,
                    sizeof(model->metrics.steps_without_improvement))) return 0;
    model->metrics.history_size = (size_t)history_size;

    bpe_encoder_t *encoder = malloc(sizeof(*encoder));
    if (!encoder) return 0;
    memset(encoder, 0, sizeof(*encoder));
    if (bpe_encoder_read(encoder, file) != BPE_SUCCESS ||
        encoder->max_vocab_size != model->vocab_size) {
        bpe_encoder_free(encoder);
        free(encoder);
        return 0;
    }
    char end[CHECKPOINT_MAGIC_SIZE];
    if (!read_exact(file, end, sizeof(end)) || memcmp(end, CHECKPOINT_END, sizeof(end)) != 0 ||
        fgetc(file) != EOF) {
        bpe_encoder_free(encoder);
        free(encoder);
        return 0;
    }
    *out_encoder = encoder;
    return 1;
}

int checkpoint_load(neural_model_t *model, bpe_encoder_t **out_encoder,
                    checkpoint_run_state_t *out_run_state,
                    const char *checkpoint_path) {
    if (!model || !out_encoder || !out_run_state || !checkpoint_path) return -1;
    *out_encoder = NULL;
    memset(out_run_state, 0, sizeof(*out_run_state));
    FILE *file = fopen(checkpoint_path, "rb");
    if (!file) return -1;
    neural_model_t loaded = {0};
    bpe_encoder_t *encoder = NULL;
    int ok = read_checkpoint(file, &loaded, &encoder, out_run_state);
    fclose(file);
    if (!ok) {
        model_free(&loaded);
        if (encoder) { bpe_encoder_free(encoder); free(encoder); }
        return -1;
    }
    *model = loaded;
    *out_encoder = encoder;
    return 0;
}

int checkpoint_find_latest(const char *checkpoint_dir, char *out_path, size_t max_path_len) {
    if (!checkpoint_dir || !out_path || max_path_len == 0) return -1;
    DIR *dir = opendir(checkpoint_dir);
    if (!dir) return -1;
    checkpoint_metadata_t best = {0};
    int found = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        checkpoint_metadata_t current;
        if (checkpoint_parse_metadata(entry->d_name, &current) != 0) continue;
        if (!found || current.training_step > best.training_step ||
            (current.training_step == best.training_step && current.epoch > best.epoch)) {
            best = current;
            found = 1;
        }
    }
    closedir(dir);
    if (!found) return -1;
    int written = snprintf(out_path, max_path_len, "%s/%s", checkpoint_dir, best.filepath);
    return written >= 0 && (size_t)written < max_path_len ? 0 : -1;
}

size_t checkpoint_list(const char *checkpoint_dir, char **out_paths, size_t max_checkpoints) {
    if (!checkpoint_dir || !out_paths || max_checkpoints == 0) return 0;
    DIR *dir = opendir(checkpoint_dir);
    if (!dir) return 0;
    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && count < max_checkpoints) {
        checkpoint_metadata_t metadata;
        if (checkpoint_parse_metadata(entry->d_name, &metadata) != 0) continue;
        size_t needed = strlen(checkpoint_dir) + strlen(entry->d_name) + 2;
        out_paths[count] = malloc(needed);
        if (!out_paths[count]) break;
        snprintf(out_paths[count], needed, "%s/%s", checkpoint_dir, entry->d_name);
        count++;
    }
    closedir(dir);
    return count;
}

int checkpoint_delete(const char *checkpoint_path) {
    return checkpoint_path && remove(checkpoint_path) == 0 ? 0 : -1;
}
