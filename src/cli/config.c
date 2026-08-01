/*
 * Phase 3: Configuration file implementation
 * Handles JSON-format configuration for reproducible training
 */

#include "cli/config.h"
#include "common/debug.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Get default configuration */
void config_get_defaults(config_t *out_config) {
    if (!out_config) return;
    
    out_config->vocab_size = 1000;
    out_config->embedding_dim = 64;
    out_config->num_heads = 4;
    out_config->num_layers = 2;
    out_config->max_seq_len = 512;
    out_config->learning_rate = 0.001f;
    
    out_config->batch_size = 1;           // Start with single sample
    out_config->gradient_accumulation_steps = 1;
    out_config->shuffle = 0;
    out_config->num_epochs = 3;
    out_config->checkpoint_interval = 10; // Save checkpoint every 10 steps
    out_config->seed = 1;

    out_config->tokenizer_vocab_size = 0;
    out_config->tokenizer_has_special_tokens = 0;
    out_config->pad_token_id = UINT32_MAX;
    out_config->unk_token_id = UINT32_MAX;
    out_config->bos_token_id = UINT32_MAX;
    out_config->eos_token_id = UINT32_MAX;
    out_config->input_fingerprint = 0;
    out_config->input_bytes = 0;
    out_config->validation_fingerprint = 0;
    out_config->validation_bytes = 0;
    out_config->validation_tokens = 0;
    out_config->validation_cross_entropy = 0.0;
    out_config->validation_perplexity = 0.0;
    
    strcpy(out_config->model_path, "dranzer.pth");
    strcpy(out_config->tokenizer_path, "dranzer.pth.tokenizer");
    out_config->input_path[0] = '\0';
    out_config->validation_path[0] = '\0';
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
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "vocab_size = %zu", &out_config->vocab_size)) continue;
        if (sscanf(line, "embedding_dim = %zu", &out_config->embedding_dim)) continue;
        if (sscanf(line, "num_heads = %zu", &out_config->num_heads)) continue;
        if (sscanf(line, "num_layers = %zu", &out_config->num_layers)) continue;
        if (sscanf(line, "max_seq_len = %zu", &out_config->max_seq_len)) continue;
        if (sscanf(line, "learning_rate = %f", &out_config->learning_rate)) continue;
        if (sscanf(line, "batch_size = %zu", &out_config->batch_size)) continue;
        if (sscanf(line, "gradient_accumulation_steps = %zu",
                   &out_config->gradient_accumulation_steps)) continue;
        if (sscanf(line, "shuffle = %d", &out_config->shuffle)) continue;
        if (sscanf(line, "num_epochs = %zu", &out_config->num_epochs)) continue;
        if (sscanf(line, "checkpoint_interval = %zu", &out_config->checkpoint_interval)) continue;
        if (sscanf(line, "seed = %u", &out_config->seed)) continue;
        if (sscanf(line, "tokenizer_vocab_size = %zu", &out_config->tokenizer_vocab_size)) continue;
        if (sscanf(line, "tokenizer_has_special_tokens = %d",
                   &out_config->tokenizer_has_special_tokens)) continue;
        if (sscanf(line, "pad_token_id = %" SCNu32, &out_config->pad_token_id)) continue;
        if (sscanf(line, "unk_token_id = %" SCNu32, &out_config->unk_token_id)) continue;
        if (sscanf(line, "bos_token_id = %" SCNu32, &out_config->bos_token_id)) continue;
        if (sscanf(line, "eos_token_id = %" SCNu32, &out_config->eos_token_id)) continue;
        if (sscanf(line, "input_fingerprint_fnv1a = %" SCNx64,
                   &out_config->input_fingerprint)) continue;
        if (sscanf(line, "input_bytes = %zu", &out_config->input_bytes)) continue;
        if (sscanf(line, "validation_fingerprint_fnv1a = %" SCNx64,
                   &out_config->validation_fingerprint)) continue;
        if (sscanf(line, "validation_bytes = %zu", &out_config->validation_bytes)) continue;
        if (sscanf(line, "validation_tokens = %zu", &out_config->validation_tokens)) continue;
        if (sscanf(line, "validation_cross_entropy = %lf",
                   &out_config->validation_cross_entropy)) continue;
        if (sscanf(line, "validation_perplexity = %lf",
                   &out_config->validation_perplexity)) continue;
        if (sscanf(line, "model_path = %1023[^\n]", out_config->model_path)) continue;
        if (sscanf(line, "tokenizer_path = %1023[^\n]", out_config->tokenizer_path)) continue;
        if (sscanf(line, "input_path = %1023[^\n]", out_config->input_path)) continue;
        if (sscanf(line, "validation_path = %1023[^\n]", out_config->validation_path)) continue;
        if (sscanf(line, "checkpoint_dir = %1023[^\n]", out_config->checkpoint_dir)) continue;
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
    fprintf(f, "num_layers = %zu\n", config->num_layers);
    fprintf(f, "max_seq_len = %zu\n\n", config->max_seq_len);
    
    fprintf(f, "# Training Settings\n");
    fprintf(f, "learning_rate = %.8f\n", config->learning_rate);
    fprintf(f, "batch_size = %zu\n", config->batch_size);
    fprintf(f, "gradient_accumulation_steps = %zu\n",
            config->gradient_accumulation_steps);
    fprintf(f, "shuffle = %d\n", config->shuffle);
    fprintf(f, "num_epochs = %zu\n", config->num_epochs);
    fprintf(f, "checkpoint_interval = %zu\n", config->checkpoint_interval);
    fprintf(f, "seed = %u\n\n", config->seed);

    fprintf(f, "# Frozen Tokenizer and Corpus Provenance\n");
    fprintf(f, "tokenizer_vocab_size = %zu\n", config->tokenizer_vocab_size);
    fprintf(f, "tokenizer_has_special_tokens = %d\n",
            config->tokenizer_has_special_tokens);
    fprintf(f, "pad_token_id = %" PRIu32 "\n", config->pad_token_id);
    fprintf(f, "unk_token_id = %" PRIu32 "\n", config->unk_token_id);
    fprintf(f, "bos_token_id = %" PRIu32 "\n", config->bos_token_id);
    fprintf(f, "eos_token_id = %" PRIu32 "\n", config->eos_token_id);
    fprintf(f, "input_fingerprint_fnv1a = %016" PRIx64 "\n", config->input_fingerprint);
    fprintf(f, "input_bytes = %zu\n", config->input_bytes);
    if (config->validation_path[0]) {
        fprintf(f, "validation_fingerprint_fnv1a = %016" PRIx64 "\n",
                config->validation_fingerprint);
        fprintf(f, "validation_bytes = %zu\n", config->validation_bytes);
        fprintf(f, "validation_tokens = %zu\n", config->validation_tokens);
        fprintf(f, "validation_cross_entropy = %.12f\n", config->validation_cross_entropy);
        fprintf(f, "validation_perplexity = %.12f\n", config->validation_perplexity);
    }
    fprintf(f, "\n");
    
    fprintf(f, "# Paths\n");
    fprintf(f, "model_path = %s\n", config->model_path);
    fprintf(f, "tokenizer_path = %s\n", config->tokenizer_path);
    fprintf(f, "input_path = %s\n", config->input_path);
    if (config->validation_path[0]) {
        fprintf(f, "validation_path = %s\n", config->validation_path);
    }
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
    printf("  Layers: %zu\n", config->num_layers);
    printf("  Max sequence length: %zu\n", config->max_seq_len);
    printf("\nTraining Settings:\n");
    printf("  Learning rate: %.8f\n", config->learning_rate);
    printf("  Batch size: %zu\n", config->batch_size);
    printf("  Gradient accumulation: %zu\n", config->gradient_accumulation_steps);
    printf("  Minibatch shuffle: %s\n", config->shuffle ? "enabled" : "disabled");
    printf("  Epochs: %zu\n", config->num_epochs);
    printf("  Checkpoint interval: %zu steps\n", config->checkpoint_interval);
    printf("  Seed: %u\n", config->seed);
    printf("Frozen tokenizer/corpus:\n");
    printf("  Tokenizer vocabulary: %zu\n", config->tokenizer_vocab_size);
    printf("  Special tokens: %s (PAD=%" PRIu32 ", UNK=%" PRIu32
           ", BOS=%" PRIu32 ", EOS=%" PRIu32 ")\n",
           config->tokenizer_has_special_tokens ? "enabled" : "legacy/disabled",
           config->pad_token_id, config->unk_token_id,
           config->bos_token_id, config->eos_token_id);
    printf("  Input fingerprint (FNV-1a): %016" PRIx64 "\n", config->input_fingerprint);
    printf("  Input bytes: %zu\n", config->input_bytes);
    if (config->validation_path[0]) {
        printf("  Validation fingerprint (FNV-1a): %016" PRIx64 "\n",
               config->validation_fingerprint);
        printf("  Validation: %zu bytes, %zu tokens, cross-entropy %.6f, perplexity %.6f\n",
               config->validation_bytes, config->validation_tokens,
               config->validation_cross_entropy, config->validation_perplexity);
    }
    printf("\nPaths:\n");
    printf("  Model: %s\n", config->model_path);
    printf("  Tokenizer: %s\n", config->tokenizer_path);
    printf("  Input: %s\n", config->input_path);
    if (config->validation_path[0]) printf("  Validation: %s\n", config->validation_path);
    printf("  Checkpoints: %s\n", config->checkpoint_dir);
    printf("==============================\n\n");
}
