/*
 * Command-line argument parsing implementation
 */

#include "include/cli.h"
#include "include/debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Get default CLI arguments */
void cli_get_defaults(cli_args_t *out_args) {
    if (!out_args) return;
    
    memset(out_args, 0, sizeof(cli_args_t));
    
    out_args->mode = MODE_TRAIN;
    strcpy(out_args->input_file, "../tests/chunk_aa");
    strcpy(out_args->model_path, "dranzer.pth");
    strcpy(out_args->checkpoint_dir, "checkpoints");
    
    /* Training defaults */
    out_args->epochs = 1;
    out_args->batch_size = 1;
    out_args->learning_rate = 0.001f;
    out_args->checkpoint_interval = 10;
    
    /* Inference defaults */
    out_args->generate_length = 50;
    out_args->sampling_strategy = SAMPLING_GREEDY;
    out_args->top_k = 5;
    out_args->top_p = 0.9f;
    out_args->temperature = 0.8f;
    
    /* Flags */
    out_args->use_gpu = 0;
    out_args->debug = 0;
    out_args->help = 0;
}

/* Print help message */
void cli_print_help(const char *program_name) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Attention in C - Neural Model with Multi-Head Attention     ║\n");
    printf("║  Phase 1: Attention | Phase 2: Stability | Phase 3: Scale    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Usage: %s [MODE] [OPTIONS]\n\n", program_name);
    
    printf("MODES:\n");
    printf("  train       Train the model (default)\n");
    printf("  infer       Run inference on a prompt\n");
    printf("  generate    Generate text from seed\n\n");
    
    printf("TRAINING OPTIONS:\n");
    printf("  --input FILE              Input training file (default: test.txt)\n");
    printf("  --epochs N                Number of epochs (default: 1)\n");
    printf("  --batch-size N            Batch size (default: 1)\n");
    printf("  --learning-rate LR        Learning rate (default: 0.001)\n");
    printf("  --model FILE              Model path (default: dranzer.pth)\n");
    printf("  --checkpoint-dir DIR      Checkpoint directory (default: checkpoints)\n");
    printf("  --checkpoint-interval N   Save checkpoint every N steps (default: 10)\n\n");
    
    printf("INFERENCE/GENERATION OPTIONS:\n");
    printf("  --prompt TEXT             Input prompt for inference\n");
    printf("  --length N                Generation length (default: 50)\n");
    printf("  --sampling STRATEGY       Sampling: greedy, topk, topp (default: greedy)\n");
    printf("  --top-k N                 Top-k value (default: 5)\n");
    printf("  --top-p P                 Top-p value 0.0-1.0 (default: 0.9)\n");
    printf("  --temperature T           Temperature 0.0-2.0 (default: 0.8)\n\n");
    
    printf("GENERAL OPTIONS:\n");
    printf("  --gpu                     Use GPU acceleration if available\n");
    printf("  --debug                   Enable debug output\n");
    printf("  --help                    Show this help message\n\n");
    
    printf("EXAMPLES:\n");
    printf("  # Train on file\n");
    printf("  %s train --input data.txt --epochs 5 --batch-size 32\n\n", program_name);
    
    printf("  # Train with GPU\n");
    printf("  %s train --input data.txt --gpu --epochs 10\n\n", program_name);
    
    printf("  # Generate text from prompt\n");
    printf("  %s generate --prompt \"Hello world\" --length 100 --top-p 0.9\n\n", program_name);
    
    printf("  # Inference with top-k sampling\n");
    printf("  %s infer --prompt \"Once upon a time\" --sampling topk --top-k 10\n\n", program_name);
}

/* Parse command-line arguments */
int cli_parse(int argc, char *argv[], cli_args_t *out_args) {
    if (!out_args) return -1;
    
    /* Get defaults first */
    cli_get_defaults(out_args);
    
    if (argc < 1) return 0;
    
    /* Check for help first */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            out_args->help = 1;
            return 0;
        }
    }
    
    /* Parse mode (first positional argument if not a flag) */
    int arg_start = 1;
    if (arg_start < argc && argv[arg_start][0] != '-') {
        if (strcmp(argv[arg_start], "train") == 0) {
            out_args->mode = MODE_TRAIN;
        } else if (strcmp(argv[arg_start], "infer") == 0) {
            out_args->mode = MODE_INFER;
        } else if (strcmp(argv[arg_start], "generate") == 0) {
            out_args->mode = MODE_GENERATE;
        }
        arg_start++;
    }
    
    /* Parse options */
    for (int i = arg_start; i < argc; i++) {
        const char *arg = argv[i];
        
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            out_args->help = 1;
        }
        else if (strcmp(arg, "--gpu") == 0) {
            out_args->use_gpu = 1;
        }
        else if (strcmp(arg, "--debug") == 0) {
            out_args->debug = 1;
        }
        else if (strcmp(arg, "--input") == 0 && i + 1 < argc) {
            strncpy(out_args->input_file, argv[++i], sizeof(out_args->input_file) - 1);
        }
        else if (strcmp(arg, "--model") == 0 && i + 1 < argc) {
            strncpy(out_args->model_path, argv[++i], sizeof(out_args->model_path) - 1);
        }
        else if (strcmp(arg, "--checkpoint-dir") == 0 && i + 1 < argc) {
            strncpy(out_args->checkpoint_dir, argv[++i], sizeof(out_args->checkpoint_dir) - 1);
        }
        else if (strcmp(arg, "--epochs") == 0 && i + 1 < argc) {
            out_args->epochs = atoi(argv[++i]);
        }
        else if (strcmp(arg, "--batch-size") == 0 && i + 1 < argc) {
            out_args->batch_size = atoi(argv[++i]);
        }
        else if (strcmp(arg, "--learning-rate") == 0 && i + 1 < argc) {
            out_args->learning_rate = atof(argv[++i]);
        }
        else if (strcmp(arg, "--checkpoint-interval") == 0 && i + 1 < argc) {
            out_args->checkpoint_interval = atoi(argv[++i]);
        }
        else if (strcmp(arg, "--prompt") == 0 && i + 1 < argc) {
            strncpy(out_args->prompt, argv[++i], sizeof(out_args->prompt) - 1);
        }
        else if (strcmp(arg, "--length") == 0 && i + 1 < argc) {
            out_args->generate_length = atoi(argv[++i]);
        }
        else if (strcmp(arg, "--sampling") == 0 && i + 1 < argc) {
            const char *strategy = argv[++i];
            if (strcmp(strategy, "greedy") == 0) {
                out_args->sampling_strategy = SAMPLING_GREEDY;
            } else if (strcmp(strategy, "topk") == 0) {
                out_args->sampling_strategy = SAMPLING_TOPK;
            } else if (strcmp(strategy, "topp") == 0) {
                out_args->sampling_strategy = SAMPLING_TOPP;
            }
        }
        else if (strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            out_args->top_k = atoi(argv[++i]);
        }
        else if (strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            out_args->top_p = atof(argv[++i]);
        }
        else if (strcmp(arg, "--temperature") == 0 && i + 1 < argc) {
            out_args->temperature = atof(argv[++i]);
        }
    }
    
    return 0;
}

/* Print parsed arguments */
void cli_print_args(const cli_args_t *args) {
    if (!args) return;
    
    printf("\n=== CLI Configuration ===\n");
    
    switch (args->mode) {
        case MODE_TRAIN:
            printf("Mode: TRAIN\n");
            printf("  Input: %s\n", args->input_file);
            printf("  Model: %s\n", args->model_path);
            printf("  Checkpoints: %s\n", args->checkpoint_dir);
            printf("  Epochs: %d\n", args->epochs);
            printf("  Batch size: %d\n", args->batch_size);
            printf("  Learning rate: %.8f\n", args->learning_rate);
            printf("  Checkpoint interval: %d steps\n", args->checkpoint_interval);
            break;
            
        case MODE_INFER:
            printf("Mode: INFERENCE\n");
            printf("  Model: %s\n", args->model_path);
            printf("  Prompt: %s\n", args->prompt);
            printf("  Sampling: ");
            switch (args->sampling_strategy) {
                case SAMPLING_GREEDY:
                    printf("greedy\n");
                    break;
                case SAMPLING_TOPK:
                    printf("top-k (k=%d)\n", args->top_k);
                    break;
                case SAMPLING_TOPP:
                    printf("top-p (p=%.2f)\n", args->top_p);
                    break;
                case SAMPLING_BEAM:
                    printf("beam search\n");
                    break;
            }
            break;
            
        case MODE_GENERATE:
            printf("Mode: GENERATE\n");
            printf("  Model: %s\n", args->model_path);
            printf("  Prompt: %s\n", args->prompt);
            printf("  Length: %d\n", args->generate_length);
            printf("  Temperature: %.2f\n", args->temperature);
            printf("  Sampling: ");
            switch (args->sampling_strategy) {
                case SAMPLING_GREEDY:
                    printf("greedy\n");
                    break;
                case SAMPLING_TOPK:
                    printf("top-k (k=%d)\n", args->top_k);
                    break;
                case SAMPLING_TOPP:
                    printf("top-p (p=%.2f)\n", args->top_p);
                    break;
                case SAMPLING_BEAM:
                    printf("beam search\n");
                    break;
            }
            break;
    }
    
    if (args->use_gpu) printf("  GPU: YES\n");
    if (args->debug) printf("  Debug: YES\n");
    
    printf("==========================\n\n");
}
