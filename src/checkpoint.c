/*
 * Phase 3: Training checkpoints implementation
 * Saves/loads model state at intervals during training
 */

#include "include/checkpoint.h"
#include "include/debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

/* Helper: Create checkpoint filename */
static void checkpoint_make_filename(char *buffer, size_t buffer_len,
                                     uint32_t step, uint32_t epoch) {
    snprintf(buffer, buffer_len, "checkpoint_epoch_%u_step_%u.ckpt", epoch, step);
}

/* Save training checkpoint */
int checkpoint_save(const neural_model_t *model, uint32_t step, uint32_t epoch,
                    const char *checkpoint_dir) {
    if (!model || !checkpoint_dir) return -1;
    
    /* Create checkpoint directory if it doesn't exist */
    mkdir(checkpoint_dir, 0755);
    
    char filename[256];
    char filepath[512];
    
    checkpoint_make_filename(filename, sizeof(filename), step, epoch);
    snprintf(filepath, sizeof(filepath), "%s/%s", checkpoint_dir, filename);
    
    FILE *f = fopen(filepath, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not create checkpoint file %s\n", filepath);
        return -1;
    }

    /* Write checkpoint metadata, then delegate the model weights themselves
     * to the same serializer model_save() uses, so the two formats can
     * never drift out of sync again. */
    fwrite(&step, sizeof(uint32_t), 1, f);
    fwrite(&epoch, sizeof(uint32_t), 1, f);
    fwrite(&model->current_loss, sizeof(float), 1, f);
    fwrite(&model->learning_rate, sizeof(float), 1, f);

    model_errors_t rc = model_write_state(model, f);
    fclose(f);
    if (rc != MODEL_SUCCESS) {
        fprintf(stderr, "Error: Could not write model state to %s\n", filepath);
        return -1;
    }

    printf("✓ Checkpoint saved: %s\n", filepath);
    DEBUG_PRINT("Checkpoint saved (step=%u, epoch=%u, loss=%.4f)\n", step, epoch, model->current_loss);
    
    return 0;
}

/* Load training checkpoint */
int checkpoint_load(neural_model_t *model, const char *checkpoint_path) {
    if (!model || !checkpoint_path) return -1;
    
    FILE *f = fopen(checkpoint_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open checkpoint file %s\n", checkpoint_path);
        return -1;
    }
    
    /* Read checkpoint metadata, then delegate the model weights themselves
     * to the same deserializer model_load() uses. */
    uint32_t step, epoch;
    float loss, learning_rate;

    if (fread(&step, sizeof(uint32_t), 1, f) != 1 ||
        fread(&epoch, sizeof(uint32_t), 1, f) != 1 ||
        fread(&loss, sizeof(float), 1, f) != 1 ||
        fread(&learning_rate, sizeof(float), 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "Error: Could not read checkpoint metadata from %s\n", checkpoint_path);
        return -1;
    }

    model_errors_t rc = model_read_state(model, f);
    fclose(f);
    if (rc != MODEL_SUCCESS) {
        fprintf(stderr, "Error: Could not read model state from %s\n", checkpoint_path);
        return -1;
    }

    model->current_loss = loss;
    model->learning_rate = learning_rate;

    printf("✓ Checkpoint loaded: %s\n", checkpoint_path);
    DEBUG_PRINT("Checkpoint loaded (step=%u, epoch=%u, loss=%.4f)\n", step, epoch, loss);
    
    return 0;
}

/* Find latest checkpoint in directory */
int checkpoint_find_latest(const char *checkpoint_dir, char *out_path, size_t max_path_len) {
    if (!checkpoint_dir || !out_path) return -1;
    
    DIR *dir = opendir(checkpoint_dir);
    if (!dir) return -1;
    
    struct dirent *entry;
    uint32_t latest_step = 0;
    char latest_filename[256] = {0};
    
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".ckpt")) {
            uint32_t step = 0;
            sscanf(entry->d_name, "checkpoint_epoch_%*u_step_%u.ckpt", &step);
            
            if (step >= latest_step) {
                latest_step = step;
                strncpy(latest_filename, entry->d_name, sizeof(latest_filename) - 1);
            }
        }
    }
    closedir(dir);
    
    if (latest_filename[0] == '\0') return -1;
    
    snprintf(out_path, max_path_len, "%s/%s", checkpoint_dir, latest_filename);
    return 0;
}

/* List all checkpoints */
size_t checkpoint_list(const char *checkpoint_dir, char **out_paths, size_t max_checkpoints) {
    if (!checkpoint_dir || !out_paths || max_checkpoints == 0) return 0;
    
    DIR *dir = opendir(checkpoint_dir);
    if (!dir) return 0;
    
    struct dirent *entry;
    size_t count = 0;
    
    while ((entry = readdir(dir)) != NULL && count < max_checkpoints) {
        if (strstr(entry->d_name, ".ckpt")) {
            out_paths[count] = malloc(512);
            if (!out_paths[count]) {
                break; /* Return what we found so far rather than crash */
            }
            snprintf(out_paths[count], 512, "%s/%s", checkpoint_dir, entry->d_name);
            count++;
        }
    }
    closedir(dir);
    
    return count;
}

/* Delete checkpoint */
int checkpoint_delete(const char *checkpoint_path) {
    if (!checkpoint_path) return -1;
    
    if (remove(checkpoint_path) == 0) {
        printf("✓ Checkpoint deleted: %s\n", checkpoint_path);
        return 0;
    }
    
    return -1;
}

/* Parse checkpoint metadata from filename */
int checkpoint_parse_metadata(const char *filename, checkpoint_metadata_t *out_metadata) {
    if (!filename || !out_metadata) return -1;
    
    memset(out_metadata, 0, sizeof(checkpoint_metadata_t));
    
    sscanf(filename, "checkpoint_epoch_%u_step_%u.ckpt",
           &out_metadata->epoch, &out_metadata->training_step);
    
    strncpy(out_metadata->filepath, filename, sizeof(out_metadata->filepath) - 1);
    
    return 0;
}
