/*
 * Phase 3: Training checkpoints for resumable training
 * Saves/loads model state at intervals
 */

#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <stddef.h>
#include <stdint.h>
#include "model.h"

typedef struct {
    uint32_t training_step;
    uint32_t epoch;
    float loss;
    float learning_rate;
    char timestamp[64];
    
    /* File path for this checkpoint */
    char filepath[256];
} checkpoint_metadata_t;

/**
 * Save training checkpoint
 * @param model: Neural model to save
 * @param step: Training step number
 * @param epoch: Epoch number
 * @param checkpoint_dir: Directory to save checkpoint
 * @return 0 on success, -1 on error
 */
int checkpoint_save(const neural_model_t *model, uint32_t step, uint32_t epoch, 
                    const char *checkpoint_dir);

/**
 * Load training checkpoint
 * @param model: Neural model to load into
 * @param checkpoint_path: Full path to checkpoint file
 * @return 0 on success, -1 on error
 */
int checkpoint_load(neural_model_t *model, const char *checkpoint_path);

/**
 * Find latest checkpoint in directory
 * @param checkpoint_dir: Directory containing checkpoints
 * @param out_path: Output buffer for latest checkpoint path
 * @param max_path_len: Size of output buffer
 * @return 0 if checkpoint found, -1 if none found
 */
int checkpoint_find_latest(const char *checkpoint_dir, char *out_path, size_t max_path_len);

/**
 * List all checkpoints in directory
 * @param checkpoint_dir: Directory to search
 * @param out_paths: Array to store checkpoint paths (must be pre-allocated)
 * @param max_checkpoints: Max checkpoints to return
 * @return Number of checkpoints found
 */
size_t checkpoint_list(const char *checkpoint_dir, char **out_paths, size_t max_checkpoints);

/**
 * Delete checkpoint file
 * @param checkpoint_path: Path to checkpoint to delete
 * @return 0 on success, -1 on error
 */
int checkpoint_delete(const char *checkpoint_path);

/**
 * Get checkpoint metadata from filename
 * @param filename: Checkpoint filename
 * @param out_metadata: Output metadata
 * @return 0 on success, -1 on parse error
 */
int checkpoint_parse_metadata(const char *filename, checkpoint_metadata_t *out_metadata);

#endif // CHECKPOINT_H
