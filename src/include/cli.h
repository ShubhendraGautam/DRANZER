/*
 * Command-line argument parsing
 * Handles all CLI options for training, inference, and generation modes
 */

#ifndef CLI_H
#define CLI_H

#include <stddef.h>
#include "sampling.h"

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
    char checkpoint_dir[512];
    
    /* Training hyperparameters */
    int epochs;
    int batch_size;
    float learning_rate;
    int checkpoint_interval;
    
    /* Inference parameters */
    char prompt[1024];
    int generate_length;
    sampling_strategy_t sampling_strategy;
    int top_k;
    float top_p;
    float temperature;
    
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
