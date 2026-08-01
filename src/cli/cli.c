/*
 * Command-line argument parsing implementation
 */

#include "cli/cli.h"
#include "common/debug.h"
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
    out_args->tokenizer_path[0] = '\0';
    strcpy(out_args->checkpoint_dir, "checkpoints");

    /* Model architecture defaults - match the values this project used as
     * hardcoded main.c #defines before these became CLI flags. */
    out_args->vocab_size = 257;
    out_args->embedding_dim = 16;
    out_args->num_heads = 2;
    out_args->num_layers = 2;
    out_args->max_seq_len = 32;
    out_args->train_window = 16;

    /* Training defaults */
    out_args->epochs = 1;
    out_args->batch_size = 1;
    out_args->learning_rate = 0.001f;
    out_args->checkpoint_interval = 10;

    /* Optimizer / regularization defaults - match model_new's own
     * defaults so a caller that skips these flags gets the same behavior
     * as a caller that constructs a neural_model_t directly. */
    strcpy(out_args->optimizer, "adam");
    out_args->dropout_rate = 0.0f;
    out_args->grad_clip_norm = 1.0f;
    out_args->weight_decay = 0.01f;
    out_args->warmup_steps = 0;
    out_args->total_steps = 0;
    
    /* Inference defaults */
    out_args->generate_length = 50;
    out_args->sampling_strategy = SAMPLING_GREEDY;
    out_args->top_k = 5;
    out_args->top_p = 0.9f;
    out_args->temperature = 0.8f;
    out_args->seed = 1;
    
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
    printf("  --tokenizer FILE          BPE vocabulary sidecar (default: <model>.tokenizer)\n");
    printf("  --checkpoint-dir DIR      Checkpoint directory (default: checkpoints)\n");
    printf("  --checkpoint-interval N   Save checkpoint every N steps (default: 10)\n\n");

    printf("MODEL ARCHITECTURE OPTIONS:\n");
    printf("  --vocab-size N            Vocabulary size (default: 257)\n");
    printf("  --embedding-dim N         Embedding dimension; must divide evenly by --num-heads (default: 16)\n");
    printf("  --num-heads N             Attention heads (default: 2)\n");
    printf("  --num-layers N            Stacked transformer layers (default: 2)\n");
    printf("  --max-seq-len N           Max sequence length the model's workspace is sized for (default: 32)\n");
    printf("  --train-window N          Sliding context window used during training; clamped to --max-seq-len (default: 16)\n\n");

    printf("OPTIMIZER / REGULARIZATION OPTIONS:\n");
    printf("  --optimizer NAME          Optimizer: adam, sgd (default: adam)\n");
    printf("  --dropout RATE            Dropout rate 0.0-1.0 (default: 0.0, disabled)\n");
    printf("  --grad-clip NORM          Global gradient-norm clip; 0 disables (default: 1.0)\n");
    printf("  --weight-decay W          AdamW decoupled weight decay (default: 0.01)\n");
    printf("  --warmup-steps N          Linear LR warmup length (default: 0)\n");
    printf("  --total-steps N           LR schedule horizon; 0 disables warmup+cosine schedule (default: 0)\n\n");
    
    printf("INFERENCE/GENERATION OPTIONS:\n");
    printf("  --prompt TEXT             Input prompt for inference\n");
    printf("  --length N                Generation length (default: 50)\n");
    printf("  --sampling STRATEGY       Sampling: greedy, topk, topp (default: greedy)\n");
    printf("  --top-k N                 Top-k value (default: 5)\n");
    printf("  --top-p P                 Top-p value 0.0-1.0 (default: 0.9)\n");
    printf("  --temperature T           Temperature 0.0-2.0 (default: 0.8)\n");
    printf("  --seed N                  Random seed (default: 1)\n\n");
    
    printf("GENERAL OPTIONS:\n");
    printf("  --gpu                     Offload forward-pass matmuls to a CUDA GPU if usable (see `make bench`\n");
    printf("                            for when this actually helps - a clear win at larger model sizes,\n");
    printf("                            slower than CPU at this project's small defaults)\n");
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
        else if (strcmp(arg, "--tokenizer") == 0 && i + 1 < argc) {
            strncpy(out_args->tokenizer_path, argv[++i], sizeof(out_args->tokenizer_path) - 1);
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
        else if (strcmp(arg, "--vocab-size") == 0 && i + 1 < argc) {
            out_args->vocab_size = (size_t)atoi(argv[++i]);
        }
        else if (strcmp(arg, "--embedding-dim") == 0 && i + 1 < argc) {
            out_args->embedding_dim = (size_t)atoi(argv[++i]);
        }
        else if (strcmp(arg, "--num-heads") == 0 && i + 1 < argc) {
            out_args->num_heads = (size_t)atoi(argv[++i]);
        }
        else if (strcmp(arg, "--num-layers") == 0 && i + 1 < argc) {
            out_args->num_layers = (size_t)atoi(argv[++i]);
        }
        else if (strcmp(arg, "--max-seq-len") == 0 && i + 1 < argc) {
            out_args->max_seq_len = (size_t)atoi(argv[++i]);
        }
        else if (strcmp(arg, "--train-window") == 0 && i + 1 < argc) {
            out_args->train_window = (size_t)atoi(argv[++i]);
        }
        else if (strcmp(arg, "--optimizer") == 0 && i + 1 < argc) {
            strncpy(out_args->optimizer, argv[++i], sizeof(out_args->optimizer) - 1);
        }
        else if (strcmp(arg, "--dropout") == 0 && i + 1 < argc) {
            out_args->dropout_rate = atof(argv[++i]);
        }
        else if (strcmp(arg, "--grad-clip") == 0 && i + 1 < argc) {
            out_args->grad_clip_norm = atof(argv[++i]);
        }
        else if (strcmp(arg, "--weight-decay") == 0 && i + 1 < argc) {
            out_args->weight_decay = atof(argv[++i]);
        }
        else if (strcmp(arg, "--warmup-steps") == 0 && i + 1 < argc) {
            out_args->warmup_steps = (unsigned int)atoi(argv[++i]);
        }
        else if (strcmp(arg, "--total-steps") == 0 && i + 1 < argc) {
            out_args->total_steps = (unsigned int)atoi(argv[++i]);
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
        else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            out_args->seed = (unsigned int)strtoul(argv[++i], NULL, 10);
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
            printf("  Tokenizer: %s\n", args->tokenizer_path[0] ? args->tokenizer_path : "<model>.tokenizer");
            printf("  Checkpoints: %s\n", args->checkpoint_dir);
            printf("  Architecture: vocab=%zu emb=%zu heads=%zu layers=%zu max_seq=%zu train_window=%zu\n",
                   args->vocab_size, args->embedding_dim, args->num_heads, args->num_layers,
                   args->max_seq_len, args->train_window);
            printf("  Epochs: %d\n", args->epochs);
            printf("  Batch size: %d\n", args->batch_size);
            printf("  Learning rate: %.8f\n", args->learning_rate);
            printf("  Checkpoint interval: %d steps\n", args->checkpoint_interval);
            printf("  Optimizer: %s (grad-clip=%.2f, weight-decay=%.4f)\n",
                   args->optimizer, args->grad_clip_norm, args->weight_decay);
            printf("  Dropout: %.2f\n", args->dropout_rate);
            if (args->total_steps > 0) {
                printf("  LR schedule: warmup=%u steps, cosine decay to 0 over %u total steps\n",
                       args->warmup_steps, args->total_steps);
            } else {
                printf("  LR schedule: plateau decay (no warmup/cosine horizon configured)\n");
            }
            break;
            
        case MODE_INFER:
            printf("Mode: INFERENCE\n");
            printf("  Model: %s\n", args->model_path);
            printf("  Tokenizer: %s\n", args->tokenizer_path[0] ? args->tokenizer_path : "<model>.tokenizer");
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
            printf("  Tokenizer: %s\n", args->tokenizer_path[0] ? args->tokenizer_path : "<model>.tokenizer");
            printf("  Prompt: %s\n", args->prompt);
            printf("  Length: %d\n", args->generate_length);
            printf("  Temperature: %.2f\n", args->temperature);
            printf("  Seed: %u\n", args->seed);
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
