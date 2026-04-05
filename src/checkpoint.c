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
    
    /* Write checkpoint metadata */
    fwrite(&step, sizeof(uint32_t), 1, f);
    fwrite(&epoch, sizeof(uint32_t), 1, f);
    fwrite(&model->current_loss, sizeof(float), 1, f);
    fwrite(&model->learning_rate, sizeof(float), 1, f);
    
    /* Write model state (same as model_save) */
    size_t vocab_size = model->vocab_size;
    size_t embedding_dim = model->embedding_dim;
    size_t num_heads = model->num_heads;
    
    fwrite(&vocab_size, sizeof(size_t), 1, f);
    fwrite(&embedding_dim, sizeof(size_t), 1, f);
    fwrite(&num_heads, sizeof(size_t), 1, f);
    fwrite(&model->training_steps, sizeof(uint32_t), 1, f);
    
    size_t ffn_dim = embedding_dim * 4;
    
    /* Write weight matrices */
    fwrite(model->token_embeddings, sizeof(float), vocab_size * embedding_dim, f);
    fwrite(model->W_q, sizeof(float), embedding_dim * embedding_dim, f);
    fwrite(model->W_k, sizeof(float), embedding_dim * embedding_dim, f);
    fwrite(model->W_v, sizeof(float), embedding_dim * embedding_dim, f);
    fwrite(model->W_o, sizeof(float), embedding_dim * embedding_dim, f);
    fwrite(model->W_ff1, sizeof(float), embedding_dim * ffn_dim, f);
    fwrite(model->b_ff1, sizeof(float), ffn_dim, f);
    fwrite(model->W_ff2, sizeof(float), ffn_dim * embedding_dim, f);
    fwrite(model->b_ff2, sizeof(float), embedding_dim, f);
    fwrite(model->output_projection, sizeof(float), embedding_dim * vocab_size, f);
    fwrite(model->output_bias, sizeof(float), vocab_size, f);
    
    /* Phase 2: Write layer norm parameters */
    fwrite(model->ln_gamma_attn, sizeof(float), embedding_dim, f);
    fwrite(model->ln_beta_attn, sizeof(float), embedding_dim, f);
    fwrite(model->ln_gamma_ffn, sizeof(float), embedding_dim, f);
    fwrite(model->ln_beta_ffn, sizeof(float), embedding_dim, f);
    
    fclose(f);
    
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
    
    /* Read checkpoint metadata */
    uint32_t step, epoch;
    float loss, learning_rate;
    
    fread(&step, sizeof(uint32_t), 1, f);
    fread(&epoch, sizeof(uint32_t), 1, f);
    fread(&loss, sizeof(float), 1, f);
    fread(&learning_rate, sizeof(float), 1, f);
    
    /* Read model dimensions */
    size_t vocab_size, embedding_dim, num_heads;
    uint32_t training_steps;
    
    fread(&vocab_size, sizeof(size_t), 1, f);
    fread(&embedding_dim, sizeof(size_t), 1, f);
    fread(&num_heads, sizeof(size_t), 1, f);
    fread(&training_steps, sizeof(uint32_t), 1, f);
    
    /* Initialize/reinitialize model if needed */
    if (model->vocab_size != vocab_size || model->embedding_dim != embedding_dim) {
        extern model_errors_t model_new(neural_model_t *model, size_t vocab_size,
                                       size_t embedding_dim, size_t num_heads, size_t max_seq_len);
        model_free(model);
        model_new(model, vocab_size, embedding_dim, num_heads, 512);
    }
    
    size_t ffn_dim = embedding_dim * 4;
    
    /* Read weight matrices */
    fread(model->token_embeddings, sizeof(float), vocab_size * embedding_dim, f);
    fread(model->W_q, sizeof(float), embedding_dim * embedding_dim, f);
    fread(model->W_k, sizeof(float), embedding_dim * embedding_dim, f);
    fread(model->W_v, sizeof(float), embedding_dim * embedding_dim, f);
    fread(model->W_o, sizeof(float), embedding_dim * embedding_dim, f);
    fread(model->W_ff1, sizeof(float), embedding_dim * ffn_dim, f);
    fread(model->b_ff1, sizeof(float), ffn_dim, f);
    fread(model->W_ff2, sizeof(float), ffn_dim * embedding_dim, f);
    fread(model->b_ff2, sizeof(float), embedding_dim, f);
    fread(model->output_projection, sizeof(float), embedding_dim * vocab_size, f);
    fread(model->output_bias, sizeof(float), vocab_size, f);
    
    /* Phase 2: Read layer norm parameters */
    fread(model->ln_gamma_attn, sizeof(float), embedding_dim, f);
    fread(model->ln_beta_attn, sizeof(float), embedding_dim, f);
    fread(model->ln_gamma_ffn, sizeof(float), embedding_dim, f);
    fread(model->ln_beta_ffn, sizeof(float), embedding_dim, f);
    
    /* Restore training state */
    model->training_steps = training_steps;
    model->current_loss = loss;
    model->learning_rate = learning_rate;
    
    fclose(f);
    
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
