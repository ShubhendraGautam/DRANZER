/*
 * Command-line argument parsing
 * Handles all CLI options for training, inference, and generation modes
 */

#ifndef CLI_H
#define CLI_H

#include <stddef.h>
#include "cli/sampling.h"

#define CLI_MAX_STOP_SEQUENCES 8
#define CLI_MAX_STOP_SEQUENCE_LENGTH 256

typedef enum {
    MODE_TRAIN,
    MODE_EVAL,
    MODE_INFER,
    MODE_GENERATE,
} cli_mode_t;

typedef struct {
    /* Mode */
    cli_mode_t mode;
    
    /* Input/Output */
    char input_file[1024];
    char validation_file[1024]; /* optional explicit held-out corpus */
    char model_path[1024];
    char tokenizer_path[1024]; /* empty = derive <model_path>.tokenizer */
    char checkpoint_dir[1024];
    char resume_path[1024];   /* empty = fresh run; "latest" resolves in checkpoint_dir */
    
    /* Model architecture hyperparameters. Defaults match this project's
     * original hardcoded main.c values, so omitting these flags entirely
     * reproduces the previous behavior exactly. */
    size_t vocab_size;
    size_t embedding_dim;
    size_t num_heads;
    size_t num_layers;
    size_t max_seq_len;
    size_t train_window;   // sliding-window context length used during training; clamped to max_seq_len
    /* How far the training window advances between examples. 0 means the
     * whole window, i.e. non-overlapping: every corpus token is supervised
     * exactly once per epoch, which is what all-position supervision
     * (core/lm_head.h) makes the natural default.
     *
     * The loop used to advance by one token per target because only the
     * last position of a window was supervised, so a stride of 1 was the
     * only way to reach every target. Keeping the stride configurable makes
     * the resulting trade measurable rather than assumed: a smaller stride
     * gives early positions of each window more context (the first position
     * of a non-overlapping window has exactly one token of it) at a
     * proportional cost in compute. Whether that buys any held-out quality
     * at this project's model sizes is unmeasured, and should be compared
     * against the seed-variance floor before being believed. */
    size_t train_stride;
    size_t eval_window;    // 0 = loaded model's max_seq_len
    int tie_embeddings;    // share token embeddings with the output projection
    int use_rope;           // rotary Q/K positions instead of additive sinusoids
    int use_rmsnorm;        // RMSNorm without beta instead of LayerNorm
    int use_gelu;           // GELU instead of ReLU in the FFN

    /* Training hyperparameters */
    int epochs;
    int batch_size;
    float learning_rate;
    int checkpoint_interval;
    int keep_checkpoints;
    int gradient_accumulation_steps;
    int shuffle;

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
    float repetition_penalty;
    int minimum_generation_length;
    char stop_sequences[CLI_MAX_STOP_SEQUENCES][CLI_MAX_STOP_SEQUENCE_LENGTH];
    size_t stop_sequence_count;
    unsigned int seed;
    
    /* Flags */
    int use_gpu;
    int input_explicit;
    int epochs_explicit;
    int checkpoint_dir_explicit;
    int debug;
    int help;

    /* Comma-separated option names explicitly supplied by the caller.
     * The immutable run manifest pairs this with all resolved values. */
    char explicit_options[2048];
    
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

/* True when option was supplied explicitly (exact comma-delimited match). */
int cli_option_was_explicit(const cli_args_t *args, const char *option);

#endif // CLI_H
