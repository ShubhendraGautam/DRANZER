/*
 * Command-line argument parsing implementation
 */

#include "cli/cli.h"
#include "common/debug.h"
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Get default CLI arguments */
void cli_get_defaults(cli_args_t *out_args) {
    if (!out_args) return;
    
    memset(out_args, 0, sizeof(cli_args_t));
    
    out_args->mode = MODE_TRAIN;
    strcpy(out_args->input_file, "../test.txt");
    out_args->validation_file[0] = '\0';
    strcpy(out_args->model_path, "dranzer.pth");
    out_args->tokenizer_path[0] = '\0';
    strcpy(out_args->checkpoint_dir, "checkpoints");
    out_args->resume_path[0] = '\0';

    /* Model architecture defaults - match the values this project used as
     * hardcoded main.c #defines before these became CLI flags. */
    out_args->vocab_size = 260;
    out_args->embedding_dim = 16;
    out_args->num_heads = 2;
    out_args->num_layers = 2;
    out_args->max_seq_len = 32;
    out_args->train_window = 16;
    out_args->train_stride = 0;   /* 0 = advance by a whole window */
    out_args->eval_window = 0;
    out_args->tie_embeddings = 0;
    out_args->use_rope = 0;
    out_args->use_rmsnorm = 0;
    out_args->use_gelu = 0;

    /* Training defaults */
    out_args->epochs = 1;
    out_args->batch_size = 1;
    out_args->learning_rate = 0.001f;
    out_args->checkpoint_interval = 10;
    out_args->keep_checkpoints = 3;
    out_args->gradient_accumulation_steps = 1;
    out_args->shuffle = 0;

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
    out_args->repetition_penalty = 1.0f;
    out_args->minimum_generation_length = 0;
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
    printf("  eval        Evaluate a model on an explicit held-out corpus\n");
    printf("  infer       Run inference on a prompt\n");
    printf("  generate    Generate text from seed\n\n");
    
    printf("TRAINING OPTIONS:\n");
    printf("  --input FILE              Training/evaluation corpus (training default: ../test.txt)\n");
    printf("  --validation FILE         Optional held-out corpus evaluated after each epoch\n");
    printf("  --epochs N                Number of epochs (default: 1)\n");
    printf("  --batch-size N            Batch size (default: 1)\n");
    printf("  --gradient-accumulation N Accumulate N minibatches per optimizer step (default: 1)\n");
    printf("  --shuffle                 Deterministically shuffle examples within each minibatch\n");
    printf("  --learning-rate LR        Learning rate (default: 0.001)\n");
    printf("  --model FILE              Model path (default: dranzer.pth)\n");
    printf("  --tokenizer FILE          BPE vocabulary sidecar (default: <model>.tokenizer)\n");
    printf("  --checkpoint-dir DIR      Checkpoint directory (default: checkpoints)\n");
    printf("  --checkpoint-interval N   Save checkpoint every N steps (default: 10)\n");
    printf("  --keep-checkpoints N      Retain newest N checkpoints; 0 keeps all (default: 3)\n");
    printf("  --resume FILE             Resume exact state from FILE; use 'latest' for checkpoint dir\n\n");

    printf("MODEL ARCHITECTURE OPTIONS:\n");
    printf("  --vocab-size N            Vocabulary size (default: 260; includes 4 controls)\n");
    printf("  --embedding-dim N         Embedding dimension; must divide evenly by --num-heads (default: 16)\n");
    printf("  --num-heads N             Attention heads (default: 2)\n");
    printf("  --num-layers N            Stacked transformer layers (default: 2)\n");
    printf("  --max-seq-len N           Training workspace and retained KV window (default: 32)\n");
    printf("  --train-window N          Sliding context window used during training; clamped to --max-seq-len (default: 16)\n");
    printf("  --train-stride N          Tokens the training window advances between examples; 0 = a whole window (default: 0).\n");
    printf("                            Every position in a window is supervised, so a full-window stride covers each corpus\n");
    printf("                            token exactly once per epoch. A smaller stride overlaps windows, giving early positions\n");
    printf("                            more context at a proportional cost in compute.\n");
    printf("  --tie-embeddings          Share token embeddings with the output projection, reducing parameters\n");
    printf("  --rope                    Rotate attention queries/keys by position instead of adding sinusoidal embeddings\n");
    printf("  --rmsnorm                 Use RMSNorm without beta instead of LayerNorm\n");
    printf("  --gelu                    Use GELU instead of ReLU in feed-forward layers\n\n");

    printf("EVALUATION OPTIONS:\n");
    printf("  --input FILE              Explicit held-out corpus (required by eval mode)\n");
    printf("  --eval-window N           Evaluation context; 0 uses model max sequence length (default: 0)\n\n");

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

    printf("GENERATION CONTROL OPTIONS:\n");
    printf("  --repetition-penalty P    Penalize seen tokens; P >= 1.0 (default: 1.0)\n");
    printf("  --min-length N            Minimum generated tokens before EOS/stop (default: 0)\n");
    printf("  --stop TEXT               Stop on TEXT; repeat up to %d times\n",
           CLI_MAX_STOP_SEQUENCES);
    printf("                            These three options are valid only in generate mode.\n\n");
    
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

    printf("  # Evaluate a held-out corpus\n");
    printf("  %s eval --model dranzer.pth --input validation.txt\n\n", program_name);
}

static int record_explicit(cli_args_t *args, const char *option) {
    size_t used = strlen(args->explicit_options);
    size_t length = strlen(option);
    size_t separator = used ? 1 : 0;
    if (used + separator + length >= sizeof(args->explicit_options)) return -1;
    if (separator) args->explicit_options[used++] = ',';
    memcpy(args->explicit_options + used, option, length + 1);
    return 0;
}

int cli_option_was_explicit(const cli_args_t *args, const char *option) {
    if (!args || !option || !option[0]) return 0;
    size_t wanted = strlen(option);
    const char *cursor = args->explicit_options;
    while (*cursor) {
        const char *end = strchr(cursor, ',');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (length == wanted && memcmp(cursor, option, wanted) == 0) return 1;
        if (!end) break;
        cursor = end + 1;
    }
    return 0;
}

static int option_value(int argc, char *argv[], int *index,
                        const char *option, const char **out_value) {
    if (*index + 1 >= argc) {
        fprintf(stderr, "Error: %s requires a value\n", option);
        return -1;
    }
    *out_value = argv[++(*index)];
    return 0;
}

static int store_text(const char *option, const char *value,
                      char *destination, size_t capacity) {
    size_t length = strlen(value);
    if (length >= capacity) {
        fprintf(stderr, "Error: %s value is too long\n", option);
        return -1;
    }
    memcpy(destination, value, length + 1);
    return 0;
}

static int parse_int_value(const char *option, const char *text,
                           int minimum, int maximum, int *out_value) {
    errno = 0;
    char *end = NULL;
    intmax_t value = strtoimax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value < minimum || value > maximum) {
        fprintf(stderr, "Error: %s has invalid integer value '%s'\n", option, text);
        return -1;
    }
    *out_value = (int)value;
    return 0;
}

static int parse_size_value(const char *option, const char *text,
                            size_t minimum, size_t *out_value) {
    if (text[0] == '-') {
        fprintf(stderr, "Error: %s has invalid size value '%s'\n", option, text);
        return -1;
    }
    errno = 0;
    char *end = NULL;
    uintmax_t value = strtoumax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value > SIZE_MAX || (size_t)value < minimum) {
        fprintf(stderr, "Error: %s has invalid size value '%s'\n", option, text);
        return -1;
    }
    *out_value = (size_t)value;
    return 0;
}

static int parse_unsigned_value(const char *option, const char *text,
                                unsigned int *out_value) {
    if (text[0] == '-') {
        fprintf(stderr, "Error: %s has invalid unsigned value '%s'\n", option, text);
        return -1;
    }
    errno = 0;
    char *end = NULL;
    uintmax_t value = strtoumax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value > UINT_MAX) {
        fprintf(stderr, "Error: %s has invalid unsigned value '%s'\n", option, text);
        return -1;
    }
    *out_value = (unsigned int)value;
    return 0;
}

static int finite_float(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int parse_float_value(const char *option, const char *text,
                             float minimum, float maximum,
                             int minimum_inclusive, int maximum_inclusive,
                             float *out_value) {
    errno = 0;
    char *end = NULL;
    float value = strtof(text, &end);
    int below = minimum_inclusive ? value < minimum : value <= minimum;
    int above = maximum_inclusive ? value > maximum : value >= maximum;
    if (errno == ERANGE || end == text || *end != '\0' || !finite_float(value) ||
        below || above) {
        fprintf(stderr, "Error: %s has invalid numeric value '%s'\n", option, text);
        return -1;
    }
    *out_value = value;
    return 0;
}

/* Parse command-line arguments. Unknown options, missing values, partial
 * numeric parses, overflow, and out-of-range values are all hard errors. */
int cli_parse(int argc, char *argv[], cli_args_t *out_args) {
    if (!out_args) return -1;
    cli_get_defaults(out_args);
    if (argc < 1) return 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            out_args->help = 1;
            return 0;
        }
    }

    int arg_start = 1;
    if (arg_start < argc && argv[arg_start][0] != '-') {
        const char *mode = argv[arg_start++];
        if (strcmp(mode, "train") == 0) out_args->mode = MODE_TRAIN;
        else if (strcmp(mode, "eval") == 0) out_args->mode = MODE_EVAL;
        else if (strcmp(mode, "infer") == 0) out_args->mode = MODE_INFER;
        else if (strcmp(mode, "generate") == 0) out_args->mode = MODE_GENERATE;
        else {
            fprintf(stderr, "Error: unknown mode '%s'\n", mode);
            return -1;
        }
        if (record_explicit(out_args, "mode") != 0) return -1;
    }

    for (int i = arg_start; i < argc; i++) {
        const char *arg = argv[i];
        const char *value = NULL;
        int ok = 0;

        if (strcmp(arg, "--gpu") == 0) {
            out_args->use_gpu = 1;
            ok = 1;
        } else if (strcmp(arg, "--tie-embeddings") == 0) {
            out_args->tie_embeddings = 1;
            ok = 1;
        } else if (strcmp(arg, "--rope") == 0) {
            out_args->use_rope = 1;
            ok = 1;
        } else if (strcmp(arg, "--rmsnorm") == 0) {
            out_args->use_rmsnorm = 1;
            ok = 1;
        } else if (strcmp(arg, "--gelu") == 0) {
            out_args->use_gelu = 1;
            ok = 1;
        } else if (strcmp(arg, "--debug") == 0) {
            out_args->debug = 1;
            ok = 1;
        } else if (strcmp(arg, "--shuffle") == 0) {
            out_args->shuffle = 1;
            ok = 1;
        } else if (strcmp(arg, "--input") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 store_text(arg, value, out_args->input_file,
                            sizeof(out_args->input_file)) == 0;
            out_args->input_explicit = ok;
        } else if (strcmp(arg, "--validation") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 store_text(arg, value, out_args->validation_file,
                            sizeof(out_args->validation_file)) == 0;
        } else if (strcmp(arg, "--model") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 store_text(arg, value, out_args->model_path,
                            sizeof(out_args->model_path)) == 0;
        } else if (strcmp(arg, "--tokenizer") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 store_text(arg, value, out_args->tokenizer_path,
                            sizeof(out_args->tokenizer_path)) == 0;
        } else if (strcmp(arg, "--checkpoint-dir") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 store_text(arg, value, out_args->checkpoint_dir,
                            sizeof(out_args->checkpoint_dir)) == 0;
            out_args->checkpoint_dir_explicit = ok;
        } else if (strcmp(arg, "--resume") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 store_text(arg, value, out_args->resume_path,
                            sizeof(out_args->resume_path)) == 0;
        } else if (strcmp(arg, "--prompt") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 store_text(arg, value, out_args->prompt,
                            sizeof(out_args->prompt)) == 0;
        } else if (strcmp(arg, "--stop") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0;
            if (ok && value[0] == '\0') {
                fprintf(stderr, "Error: --stop must not be empty\n");
                ok = 0;
            }
            if (ok && out_args->stop_sequence_count >= CLI_MAX_STOP_SEQUENCES) {
                fprintf(stderr, "Error: --stop may be supplied at most %d times\n",
                        CLI_MAX_STOP_SEQUENCES);
                ok = 0;
            }
            if (ok) {
                ok = store_text(
                         arg, value,
                         out_args->stop_sequences[out_args->stop_sequence_count],
                         sizeof(out_args->stop_sequences[0])) == 0;
                if (ok) out_args->stop_sequence_count++;
            }
        } else if (strcmp(arg, "--optimizer") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0;
            if (ok && strcmp(value, "adam") != 0 && strcmp(value, "sgd") != 0) {
                fprintf(stderr, "Error: --optimizer must be 'adam' or 'sgd'\n");
                ok = 0;
            }
            if (ok) ok = store_text(arg, value, out_args->optimizer,
                                    sizeof(out_args->optimizer)) == 0;
        } else if (strcmp(arg, "--sampling") == 0) {
            ok = option_value(argc, argv, &i, arg, &value) == 0;
            if (ok && strcmp(value, "greedy") == 0) out_args->sampling_strategy = SAMPLING_GREEDY;
            else if (ok && strcmp(value, "topk") == 0) out_args->sampling_strategy = SAMPLING_TOPK;
            else if (ok && strcmp(value, "topp") == 0) out_args->sampling_strategy = SAMPLING_TOPP;
            else if (ok) {
                fprintf(stderr, "Error: --sampling must be 'greedy', 'topk', or 'topp'\n");
                ok = 0;
            }
        } else if (strcmp(arg, "--epochs") == 0 || strcmp(arg, "--batch-size") == 0 ||
                   strcmp(arg, "--gradient-accumulation") == 0 ||
                   strcmp(arg, "--checkpoint-interval") == 0 ||
                   strcmp(arg, "--keep-checkpoints") == 0 || strcmp(arg, "--length") == 0 ||
                   strcmp(arg, "--min-length") == 0 ||
                   strcmp(arg, "--top-k") == 0) {
            int minimum = (strcmp(arg, "--checkpoint-interval") == 0 ||
                           strcmp(arg, "--keep-checkpoints") == 0 ||
                           strcmp(arg, "--length") == 0 ||
                           strcmp(arg, "--min-length") == 0) ? 0 : 1;
            int parsed = 0;
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 parse_int_value(arg, value, minimum, INT_MAX, &parsed) == 0;
            if (ok && strcmp(arg, "--epochs") == 0) {
                out_args->epochs = parsed;
                out_args->epochs_explicit = 1;
            } else if (ok && strcmp(arg, "--batch-size") == 0) out_args->batch_size = parsed;
            else if (ok && strcmp(arg, "--gradient-accumulation") == 0)
                out_args->gradient_accumulation_steps = parsed;
            else if (ok && strcmp(arg, "--checkpoint-interval") == 0)
                out_args->checkpoint_interval = parsed;
            else if (ok && strcmp(arg, "--keep-checkpoints") == 0)
                out_args->keep_checkpoints = parsed;
            else if (ok && strcmp(arg, "--length") == 0) out_args->generate_length = parsed;
            else if (ok && strcmp(arg, "--min-length") == 0)
                out_args->minimum_generation_length = parsed;
            else if (ok) out_args->top_k = parsed;
        } else if (strcmp(arg, "--vocab-size") == 0 ||
                   strcmp(arg, "--embedding-dim") == 0 ||
                   strcmp(arg, "--num-heads") == 0 || strcmp(arg, "--num-layers") == 0 ||
                   strcmp(arg, "--max-seq-len") == 0 ||
                   strcmp(arg, "--train-window") == 0 ||
                   strcmp(arg, "--train-stride") == 0 ||
                   strcmp(arg, "--eval-window") == 0) {
            /* --train-stride accepts 0, meaning "the whole window". */
            size_t minimum = (strcmp(arg, "--eval-window") == 0 ||
                              strcmp(arg, "--train-stride") == 0) ? 0 : 1;
            if (strcmp(arg, "--vocab-size") == 0) minimum = 256;
            size_t parsed = 0;
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 parse_size_value(arg, value, minimum, &parsed) == 0;
            if (ok && strcmp(arg, "--vocab-size") == 0) out_args->vocab_size = parsed;
            else if (ok && strcmp(arg, "--embedding-dim") == 0) out_args->embedding_dim = parsed;
            else if (ok && strcmp(arg, "--num-heads") == 0) out_args->num_heads = parsed;
            else if (ok && strcmp(arg, "--num-layers") == 0) out_args->num_layers = parsed;
            else if (ok && strcmp(arg, "--max-seq-len") == 0) out_args->max_seq_len = parsed;
            else if (ok && strcmp(arg, "--train-window") == 0) out_args->train_window = parsed;
            else if (ok && strcmp(arg, "--train-stride") == 0) out_args->train_stride = parsed;
            else if (ok) out_args->eval_window = parsed;
        } else if (strcmp(arg, "--warmup-steps") == 0 ||
                   strcmp(arg, "--total-steps") == 0 || strcmp(arg, "--seed") == 0) {
            unsigned int parsed = 0;
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 parse_unsigned_value(arg, value, &parsed) == 0;
            if (ok && strcmp(arg, "--warmup-steps") == 0) out_args->warmup_steps = parsed;
            else if (ok && strcmp(arg, "--total-steps") == 0) out_args->total_steps = parsed;
            else if (ok) out_args->seed = parsed;
        } else if (strcmp(arg, "--learning-rate") == 0 || strcmp(arg, "--dropout") == 0 ||
                   strcmp(arg, "--grad-clip") == 0 || strcmp(arg, "--weight-decay") == 0 ||
                   strcmp(arg, "--top-p") == 0 || strcmp(arg, "--temperature") == 0 ||
                   strcmp(arg, "--repetition-penalty") == 0) {
            float minimum = 0.0f, maximum = FLT_MAX;
            int minimum_inclusive = 1, maximum_inclusive = 1;
            if (strcmp(arg, "--learning-rate") == 0) minimum_inclusive = 0;
            else if (strcmp(arg, "--dropout") == 0) { maximum = 1.0f; maximum_inclusive = 0; }
            else if (strcmp(arg, "--top-p") == 0) { maximum = 1.0f; minimum_inclusive = 0; }
            else if (strcmp(arg, "--temperature") == 0) maximum = 2.0f;
            else if (strcmp(arg, "--repetition-penalty") == 0) minimum = 1.0f;
            float parsed = 0.0f;
            ok = option_value(argc, argv, &i, arg, &value) == 0 &&
                 parse_float_value(arg, value, minimum, maximum,
                                   minimum_inclusive, maximum_inclusive, &parsed) == 0;
            if (ok && strcmp(arg, "--learning-rate") == 0) out_args->learning_rate = parsed;
            else if (ok && strcmp(arg, "--dropout") == 0) out_args->dropout_rate = parsed;
            else if (ok && strcmp(arg, "--grad-clip") == 0) out_args->grad_clip_norm = parsed;
            else if (ok && strcmp(arg, "--weight-decay") == 0) out_args->weight_decay = parsed;
            else if (ok && strcmp(arg, "--top-p") == 0) out_args->top_p = parsed;
            else if (ok && strcmp(arg, "--temperature") == 0) out_args->temperature = parsed;
            else if (ok) out_args->repetition_penalty = parsed;
        } else {
            fprintf(stderr, "Error: unknown option '%s'\n", arg);
            return -1;
        }

        if (!ok || record_explicit(out_args, arg) != 0) return -1;
    }
    if (out_args->mode != MODE_GENERATE &&
        (cli_option_was_explicit(out_args, "--repetition-penalty") ||
         cli_option_was_explicit(out_args, "--min-length") ||
         cli_option_was_explicit(out_args, "--stop"))) {
        fprintf(stderr,
                "Error: --repetition-penalty, --min-length, and --stop "
                "are valid only in generate mode\n");
        return -1;
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
            if (args->validation_file[0]) printf("  Validation: %s\n", args->validation_file);
            printf("  Model: %s\n", args->model_path);
            printf("  Tokenizer: %s\n", args->tokenizer_path[0] ? args->tokenizer_path : "<model>.tokenizer");
            printf("  Checkpoints: %s\n", args->checkpoint_dir);
            printf("  Architecture: vocab=%zu emb=%zu heads=%zu layers=%zu max_seq=%zu train_window=%zu train_stride=%zu tied_embeddings=%s rope=%s rmsnorm=%s gelu=%s\n",
                   args->vocab_size, args->embedding_dim, args->num_heads, args->num_layers,
                   args->max_seq_len, args->train_window,
                   args->train_stride ? args->train_stride : args->train_window,
                   args->tie_embeddings ? "yes" : "no",
                   args->use_rope ? "yes" : "no",
                   args->use_rmsnorm ? "yes" : "no",
                   args->use_gelu ? "yes" : "no");
            printf("  Epochs: %d\n", args->epochs);
            printf("  Batch size: %d\n", args->batch_size);
            printf("  Gradient accumulation: %d minibatch(es) per optimizer step\n",
                   args->gradient_accumulation_steps);
            printf("  Minibatch shuffle: %s\n", args->shuffle ? "deterministic" : "disabled");
            printf("  Learning rate: %.8f\n", args->learning_rate);
            printf("  Checkpoint interval: %d steps\n", args->checkpoint_interval);
            printf("  Keep checkpoints: %d\n", args->keep_checkpoints);
            if (args->resume_path[0]) printf("  Resume: %s\n", args->resume_path);
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

        case MODE_EVAL:
            printf("Mode: EVALUATION\n");
            printf("  Input: %s\n", args->input_file);
            printf("  Model: %s\n", args->model_path);
            printf("  Tokenizer: %s\n", args->tokenizer_path[0] ? args->tokenizer_path : "<model>.tokenizer");
            printf("  Context window: %s\n", args->eval_window ? "explicit" : "model maximum");
            if (args->eval_window) printf("  Eval window: %zu\n", args->eval_window);
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
            printf("  Repetition penalty: %.2f\n", args->repetition_penalty);
            printf("  Minimum length: %d\n", args->minimum_generation_length);
            for (size_t i = 0; i < args->stop_sequence_count; i++)
                printf("  Stop sequence %zu: %s\n", i + 1, args->stop_sequences[i]);
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
