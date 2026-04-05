/*
 * Phase 3: Configuration file implementation
 * Handles JSON-format configuration for reproducible training
 */

#include "include/config.h"
#include "include/debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Get default configuration */
void config_get_defaults(config_t *out_config) {
    if (!out_config) return;
    
    out_config->vocab_size = 1000;
    out_config->embedding_dim = 64;
    out_config->num_heads = 4;
    out_config->max_seq_len = 512;
    out_config->learning_rate = 0.001f;
    
    out_config->batch_size = 1;           // Start with single sample
    out_config->num_epochs = 3;
    out_config->checkpoint_interval = 10; // Save checkpoint every 10 steps
    
    strcpy(out_config->model_path, "dranzer.pth");
    strcpy(out_config->checkpoint_dir, "checkpoints");
    strcpy(out_config->config_path, "config.json");
}

/* Load configuration from JSON file */
int config_load(const char *filename, config_t *out_config) {
    if (!filename || !out_config) return -1;
    
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Could not open config file %s\n", filename);
        return -1;
    }
    
    /* For now, use defaults and read basic values from simple text format */
    config_get_defaults(out_config);
    
    /* Try to parse simple key=value format as fallback */
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "vocab_size = %zu", &out_config->vocab_size)) continue;
        if (sscanf(line, "embedding_dim = %zu", &out_config->embedding_dim)) continue;
        if (sscanf(line, "num_heads = %zu", &out_config->num_heads)) continue;
        if (sscanf(line, "learning_rate = %f", &out_config->learning_rate)) continue;
        if (sscanf(line, "batch_size = %zu", &out_config->batch_size)) continue;
        if (sscanf(line, "checkpoint_interval = %zu", &out_config->checkpoint_interval)) continue;
    }
    
    fclose(f);
    DEBUG_PRINT("Configuration loaded from %s\n", filename);
    
    return 0;
}

/* Save configuration to JSON file */
int config_save(const char *filename, const config_t *config) {
    if (!filename || !config) return -1;
    
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Could not open config file %s for writing\n", filename);
        return -1;
    }
    
    fprintf(f, "# Phase 3: Model Configuration\n");
    fprintf(f, "# This file defines hyperparameters for reproducible training\n\n");
    
    fprintf(f, "# Model Architecture\n");
    fprintf(f, "vocab_size = %zu\n", config->vocab_size);
    fprintf(f, "embedding_dim = %zu\n", config->embedding_dim);
    fprintf(f, "num_heads = %zu\n", config->num_heads);
    fprintf(f, "max_seq_len = %zu\n\n", config->max_seq_len);
    
    fprintf(f, "# Training Settings\n");
    fprintf(f, "learning_rate = %.8f\n", config->learning_rate);
    fprintf(f, "batch_size = %zu\n", config->batch_size);
    fprintf(f, "num_epochs = %zu\n", config->num_epochs);
    fprintf(f, "checkpoint_interval = %zu\n\n", config->checkpoint_interval);
    
    fprintf(f, "# Paths\n");
    fprintf(f, "model_path = %s\n", config->model_path);
    fprintf(f, "checkpoint_dir = %s\n", config->checkpoint_dir);
    
    fclose(f);
    DEBUG_PRINT("Configuration saved to %s\n", filename);
    
    return 0;
}

/* Print configuration */
void config_print(const config_t *config) {
    if (!config) return;
    
    printf("\n=== Phase 3: Configuration ===\n");
    printf("Model Architecture:\n");
    printf("  Vocabulary size: %zu\n", config->vocab_size);
    printf("  Embedding dimension: %zu\n", config->embedding_dim);
    printf("  Attention heads: %zu\n", config->num_heads);
    printf("  Max sequence length: %zu\n", config->max_seq_len);
    printf("\nTraining Settings:\n");
    printf("  Learning rate: %.8f\n", config->learning_rate);
    printf("  Batch size: %zu\n", config->batch_size);
    printf("  Epochs: %zu\n", config->num_epochs);
    printf("  Checkpoint interval: %zu steps\n", config->checkpoint_interval);
    printf("\nPaths:\n");
    printf("  Model: %s\n", config->model_path);
    printf("  Checkpoints: %s\n", config->checkpoint_dir);
    printf("==============================\n\n");
}
