#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "byte_pair_encoding.h"
#include "core/model.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t epoch_index;       /* zero-based epoch being replayed/continued */
    uint64_t step_in_epoch;     /* predictions included in completed optimizer updates */
    uint32_t target_epochs;     /* total epoch target, not additional epochs */
    uint64_t input_fingerprint;
    uint64_t input_bytes;
    size_t train_window;
    unsigned int seed;
    int batch_size;
    int gradient_accumulation_steps;
    int shuffle;
    int checkpoint_interval;
    int keep_checkpoints;
    int use_gpu;
    char input_file[1024];
    char validation_file[1024];
    char tokenizer_path[1024];
    char model_path[1024];
    char checkpoint_dir[1024];
} checkpoint_run_state_t;

typedef struct {
    uint32_t training_step;
    uint32_t epoch;
    char filepath[1024];
} checkpoint_metadata_t;

/* Atomically save the complete resumable state and prune older files.
 * out_path may be NULL; otherwise it receives the final checkpoint path. */
int checkpoint_save(const neural_model_t *model, const bpe_encoder_t *encoder,
                    const checkpoint_run_state_t *run_state,
                    const char *checkpoint_dir, size_t keep_last,
                    char *out_path, size_t out_path_size);

/* Load a complete checkpoint into a zero-initialized model and a newly
 * allocated frozen encoder owned by the caller. */
int checkpoint_load(neural_model_t *model, bpe_encoder_t **out_encoder,
                    checkpoint_run_state_t *out_run_state,
                    const char *checkpoint_path);

int checkpoint_find_latest(const char *checkpoint_dir, char *out_path, size_t max_path_len);
size_t checkpoint_list(const char *checkpoint_dir, char **out_paths, size_t max_checkpoints);
int checkpoint_delete(const char *checkpoint_path);
int checkpoint_parse_metadata(const char *filename, checkpoint_metadata_t *out_metadata);

#endif /* CHECKPOINT_H */
