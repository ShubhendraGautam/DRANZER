/*
 * Command-line argument parsing
 * Handles all CLI options for training, inference, and generation modes
 */

#ifndef CLI_H
#define CLI_H

#include <stddef.h>
#include "cli/sampling.h"

typedef enum {
    MODE_TRAIN,
    MODE_INFER,
    MODE_GENERATE,
} cli_mode_t;

typedef struct {
    /* Mode */
    cli_mode_t mode;
    
    /* Input/Output */
    char input_file[512];
    char model_path[512];
    char tokenizer_path[512]; /* empty = derive <model_path>.tokenizer */
    char checkpoint_dir[512];
    
    /* Model architecture hyperparameters. Defaults match this project's
     * original hardcoded main.c values, so omitting these flags entirely
     * reproduces the previous behavior exactly. */
    size_t vocab_size;
    size_t embedding_dim;
    size_t num_heads;
    size_t num_layers;
    size_t max_seq_len;
    size_t train_window;   // sliding-window context length used during training; clamped to max_seq_len

    /* Training hyperparameters */
    int epochs;
    int batch_size;
    float learning_rate;
    int checkpoint_interval;

    /* Optimizer / regularization (research-grade training controls) */
    char optimizer[16];        // "adam" (default) or "sgd"
    float dropout_rate;        // 0 disables dropout (default)
    float grad_clip_norm;      // global grad-norm clip; 0 disables it
    float weight_decay;        // decoupled weight decay (AdamW); 0 disables it
    unsigned int warmup_steps; // linear LR warmup length; 0 + total_steps==0 disables the schedule
    unsigned int total_steps;  // LR schedule horizon (warmup+cosine decay); 0 disables the schedule
    
    /* Inference parameters */
    char prompt[1024];
    int generate_length;
    sampling_strategy_t sampling_strategy;
    int top_k;
    float top_p;
    float temperature;
    unsigned int seed;
    
    /* Flags */
    int use_gpu;
    int debug;
    int help;
    
} cli_args_t;

/**
 * Parse command-line arguments
 * @param argc: Argument count
 * @param argv: Argument values
 * @param out_args: Output parsed arguments
 * @return 0 on success, -1 on error
 */
int cli_parse(int argc, char *argv[], cli_args_t *out_args);

/**
 * Print help message
 */
void cli_print_help(const char *program_name);

/**
 * Print parsed arguments
 */
void cli_print_args(const cli_args_t *args);

/**
 * Get default CLI arguments
 */
void cli_get_defaults(cli_args_t *out_args);

#endif // CLI_H
