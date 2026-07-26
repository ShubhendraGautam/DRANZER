/*
 * Phase 3: Configuration file handling
 * Supports JSON-format configuration for reproducible training
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    /* Model hyperparameters */
    size_t vocab_size;
    size_t embedding_dim;
    size_t num_heads;
    size_t num_layers;
    size_t max_seq_len;
    float learning_rate;
    
    /* Training hyperparameters */
    size_t batch_size;
    size_t num_epochs;
    size_t checkpoint_interval;  // Save checkpoint every N steps
    
    /* Path settings */
    char model_path[256];
    char checkpoint_dir[256];
    char config_path[256];
    
} config_t;

/**
 * Load configuration from JSON file
 * @param filename: Path to configuration file
 * @param out_config: Output configuration structure
 * @return 0 on success, -1 on error
 */
int config_load(const char *filename, config_t *out_config);

/**
 * Save configuration to JSON file
 * @param filename: Path to save configuration
 * @param config: Configuration to save
 * @return 0 on success, -1 on error
 */
int config_save(const char *filename, const config_t *config);

/**
 * Get default configuration
 * @param out_config: Output default configuration
 */
void config_get_defaults(config_t *out_config);

/**
 * Print configuration to stdout
 */
void config_print(const config_t *config);

#endif // CONFIG_H
