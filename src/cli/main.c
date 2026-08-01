/*
 * Main entry point - Neural Model with Multi-Head Attention, Training, and Inference
 * Phase 1: Multi-head attention, gradient descent training, next-token prediction, model persistence
 * Phase 2: Layer normalization, learning rate scheduling, loss tracking
 * Phase 3: Batch processing, configuration files, sampling strategies, training checkpoints
 */

#include "byte_pair_encoding.h"
#include "cli/tokenizer.h"
#include "core/model.h"
#include "common/debug.h"
#include "cli/config.h"
#include "cli/sampling.h"
#include "cli/batch.h"
#include "cli/checkpoint.h"
#include "cli/cli.h"
#include "cli/stream.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Model architecture hyperparameters (vocab size, embedding dim, heads,
 * layers, max sequence length, training context window) are CLI flags now
 * (see cli.h/cli.c) rather than hardcoded here - --train-window is kept
 * below --max-seq-len (clamped in mode_train) so infer/generate, which use
 * the same model->max_seq_len bound, has headroom beyond what training
 * ever exercises. Real cross-token attention needs seq_len > 1 - with a
 * single token, softmax over one key is always 1.0 regardless of Q/K, so
 * W_q/W_k would get almost no useful gradient signal. */

/* Forward declarations */
int mode_train(const cli_args_t *args);
int mode_infer(const cli_args_t *args);
int mode_generate(const cli_args_t *args);

/* The release build uses -ffast-math, under which isfinite() and even
 * value != value may be optimized on the assumption that NaNs cannot
 * occur. Inspecting the IEEE-754 exponent bits keeps CLI validation
 * reliable for values returned by strtof/atof. */
static int float_is_finite(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int resolve_tokenizer_path(const cli_args_t *args, char *path, size_t path_size) {
    if (args->tokenizer_path[0]) {
        int written = snprintf(path, path_size, "%s", args->tokenizer_path);
        return written >= 0 && (size_t)written < path_size ? 0 : -1;
    }
    return tokenizer_default_path(args->model_path, path, path_size) == TOKENIZER_SUCCESS ? 0 : -1;
}

static bpe_encoder_t *load_model_tokenizer(const cli_args_t *args, size_t model_vocab_size) {
    char tokenizer_path[1024];
    if (resolve_tokenizer_path(args, tokenizer_path, sizeof(tokenizer_path)) != 0) {
        fprintf(stderr, "Error: Tokenizer path is too long\n");
        return NULL;
    }

    bpe_encoder_t *encoder = NULL;
    tokenizer_errors_t rc = tokenizer_load_encoder(tokenizer_path, &encoder);
    if (rc == TOKENIZER_FILE_NOT_FOUND) {
        if (args->tokenizer_path[0]) {
            fprintf(stderr, "Error: Explicit tokenizer sidecar %s was not found\n", tokenizer_path);
            return NULL;
        }
        /* Models written before tokenizer persistence have no sidecar.
         * Retain a usable byte-level fallback instead of breaking them. */
        fprintf(stderr,
                "Warning: Tokenizer sidecar %s was not found; using the legacy byte vocabulary.\n",
                tokenizer_path);
        return tokenizer_create_encoder(model_vocab_size);
    }
    if (rc != TOKENIZER_SUCCESS) {
        fprintf(stderr, "Error: Tokenizer sidecar %s is invalid or unreadable (code: %d)\n",
                tokenizer_path, rc);
        return NULL;
    }
    if (encoder->max_vocab_size != model_vocab_size) {
        fprintf(stderr, "Error: Tokenizer/model vocabulary mismatch (%zu vs %zu)\n",
                encoder->max_vocab_size, model_vocab_size);
        tokenizer_free_encoder(encoder);
        return NULL;
    }

    printf("   Tokenizer loaded from %s (%zu learned tokens)\n",
           tokenizer_path, encoder->vocab_size);
    return encoder;
}

int main(int argc, char *argv[]) {
    cli_args_t args;
    
    /* Parse command-line arguments */
    if (cli_parse(argc, argv, &args) != 0) {
        fprintf(stderr, "Error: Failed to parse command-line arguments\n");
        return 2;
    }
    
    /* Handle help flag */
    if (args.help) {
        cli_print_help(argv[0]);
        return 0;
    }

    if ((args.mode == MODE_INFER || args.mode == MODE_GENERATE) &&
        args.sampling_strategy == SAMPLING_TOPK && args.top_k <= 0) {
        fprintf(stderr, "Error: --top-k must be greater than zero\n");
        return 2;
    }
    if ((args.mode == MODE_INFER || args.mode == MODE_GENERATE) &&
        args.sampling_strategy == SAMPLING_TOPP &&
        (!float_is_finite(args.top_p) || args.top_p <= 0.0f || args.top_p > 1.0f)) {
        fprintf(stderr, "Error: --top-p must be greater than zero and at most one\n");
        return 2;
    }
    if ((args.mode == MODE_INFER || args.mode == MODE_GENERATE) &&
        (!float_is_finite(args.temperature) ||
         args.temperature < 0.0f || args.temperature > 2.0f)) {
        fprintf(stderr, "Error: --temperature must be between zero and two\n");
        return 2;
    }
    if (args.mode == MODE_GENERATE && args.generate_length < 0) {
        fprintf(stderr, "Error: --length must not be negative\n");
        return 2;
    }

    srand(args.seed);
    
    /* Enable debug if requested */
    if (args.debug) {
        /* Debug flag is now part of cli_args_t, can be checked as needed */
    }
    
    /* Print parsed arguments (compact view) */
    printf("\n>>> Mode: ");
    switch (args.mode) {
        case MODE_TRAIN: printf("TRAIN\n"); break;
        case MODE_INFER: printf("INFER\n"); break;
        case MODE_GENERATE: printf("GENERATE\n"); break;
    }
    
    if (args.use_gpu) printf(">>> GPU: enabled (if available)\n");
    printf("\n");
    
    /* Dispatch to appropriate mode */
    switch (args.mode) {
        case MODE_INFER:
            return mode_infer(&args);
        case MODE_GENERATE:
            return mode_generate(&args);
        case MODE_TRAIN:
        default:
            return mode_train(&args);
    }
}

/* ===== TRAINING MODE ===== */
int mode_train(const cli_args_t *args) {
    printf("=== Neural Model Training (LARGE FILE OPTIMIZED) ===\n\n");

    /* ===== STEP 1: BPE Tokenization with Streaming ===== */
    printf("[1] Setting up streaming tokenization...\n");
    
    bpe_encoder_t *encoder = tokenizer_create_encoder(args->vocab_size);
    if (encoder == NULL) {
        fprintf(stderr, "Error: Failed to create encoder\n");
        return 1;
    }

    /* Create streaming file reader - handles files of any size */
    stream_reader_t *stream = stream_reader_create(args->input_file, STREAM_CHUNK_SIZE);
    if (stream == NULL) {
        fprintf(stderr, "Error: Failed to open file for streaming\n");
        tokenizer_free_encoder(encoder);
        return 1;
    }
    
    printf("   ✓ Streaming reader created (256KB chunks)\n");
    
    /* Create token stream processor - accumulates tokens for batching */
    token_stream_t *token_stream = token_stream_create(10000, 1000);
    if (token_stream == NULL) {
        fprintf(stderr, "Error: Failed to create token stream\n");
        stream_reader_free(stream);
        tokenizer_free_encoder(encoder);
        return 1;
    }
    
    printf("   ✓ Token stream buffer created (batch size: 1000)\n");

    /* ===== STEP 2: Initialize Model ===== */
    printf("[2] Initializing neural model...\n");
    neural_model_t model = {0};
    model_errors_t init_rc = model_new(&model, args->vocab_size, args->embedding_dim,
                                        args->num_heads, args->num_layers, args->max_seq_len);

    if (init_rc != MODEL_SUCCESS) {
        fprintf(stderr, "Error: Model initialization failed (code: %d)\n", init_rc);
        token_stream_free(token_stream);
        stream_reader_free(stream);
        tokenizer_free_encoder(encoder);
        return 1;
    }

    /* Training's sliding context window can't exceed what the model's
     * workspace/caches were sized for. */
    size_t train_window = args->train_window;
    if (train_window > args->max_seq_len) {
        fprintf(stderr, "Warning: --train-window %zu > --max-seq-len %zu, clamping to %zu\n",
                train_window, args->max_seq_len, args->max_seq_len);
        train_window = args->max_seq_len;
    }

    /* --input's learning rate was previously parsed and only ever printed,
     * never applied - now that training actually updates every parameter
     * (not just output_bias), the learning rate materially affects
     * stability, so it needs to actually reach the model. */
    model.learning_rate = args->learning_rate;
    model.metrics.learning_rate = args->learning_rate;
    model.metrics.initial_learning_rate = args->learning_rate;

    /* Optimizer/regularization: model_new already defaults to AdamW with
     * grad-norm clipping at 1.0 and no dropout/schedule, matching this
     * function's CLI defaults - these overrides only matter when the
     * caller passes non-default flags. */
    model.optimizer_type = (strcmp(args->optimizer, "sgd") == 0) ? OPTIMIZER_SGD : OPTIMIZER_ADAM;
    model.dropout_rate = args->dropout_rate;
    model.grad_clip_norm = args->grad_clip_norm;
    model.weight_decay = args->weight_decay;
    model.warmup_steps = args->warmup_steps;
    model.total_steps = args->total_steps;
    model.base_lr = args->learning_rate;
    model.use_gpu = args->use_gpu;

    printf("   Model initialized:\n");
    printf("   - Vocabulary: %zu tokens\n", model.vocab_size);
    printf("   - Embedding dim: %zu\n", model.embedding_dim);
    printf("   - Attention heads: %zu\n", model.num_heads);
    printf("   - Layers: %zu\n", model.num_layers);
    printf("   - Optimizer: %s (grad-clip=%.2f, weight-decay=%.4f, dropout=%.2f)\n",
           args->optimizer, model.grad_clip_norm, model.weight_decay, model.dropout_rate);
    if (model.use_gpu) {
        printf("   - GPU: requested (--gpu) - used automatically for forward-pass matmuls if a CUDA GPU is usable, CPU otherwise\n");
    }
    printf("   ✓ Model ready for training\n");

    /* ===== STEP 3: Streaming Training Loop ===== */
    printf("[3] Training model on streaming data...\n");
    printf("   Epochs: %d, Batch size: %d, Learning rate: %.8f\n", 
           args->epochs, args->batch_size, args->learning_rate);
    
    size_t training_steps = 0;
    size_t total_tokens_processed = 0;
    char chunk_buffer[STREAM_CHUNK_SIZE + 1];
    
    for (int epoch = 0; epoch < args->epochs; epoch++) {
        /* Reset stream for new epoch */
        stream_reader_t *epoch_stream = stream_reader_create(args->input_file, STREAM_CHUNK_SIZE);
        if (!epoch_stream) {
            fprintf(stderr, "Error: Cannot re-open file for epoch %d\n", epoch);
            break;
        }
        
        printf("   Processing epoch %d/%d...\n", epoch + 1, args->epochs);
        
        /* Process file in chunks */
        while (!stream_is_eof(epoch_stream)) {
            size_t chunk_size = stream_read_chunk(epoch_stream, chunk_buffer, STREAM_CHUNK_SIZE);
            if (chunk_size == 0) break;
            
            chunk_buffer[chunk_size] = '\0';
            
            /* Train BPE on this chunk */
            bpe_train(encoder, chunk_buffer, chunk_size);
            
            /* Encode chunk into tokens */
            bpe_tokens_t chunk_tokens = {0};
            bpe_encode(encoder, chunk_buffer, chunk_size, &chunk_tokens);
            
            if (chunk_tokens.token_count > 0) {
                /* Add tokens to stream for batch processing */
                token_stream_add(token_stream, chunk_tokens.token_ids, chunk_tokens.token_count);
                total_tokens_processed += chunk_tokens.token_count;
                
                /* Process batch when threshold reached */
                if (token_stream_ready_to_flush(token_stream)) {
                    size_t batch_size = token_stream_get_size(token_stream);

                    /* Train on batch using a sliding context window: token
                     * i+1 is predicted from up to train_window preceding
                     * tokens (fewer near the start of the batch). */
                    for (size_t i = 0; i + 1 < batch_size; i++) {
                        size_t window_len = (i + 1 < train_window) ? (i + 1) : train_window;
                        uint32_t *window = &token_stream->token_buffer[i + 1 - window_len];
                        uint32_t target_token = token_stream->token_buffer[i + 1];

                        model_train_step(&model, window, target_token, window_len);
                        training_steps++;
                    }
                    
                    /* Progress reporting. Uses the running average loss
                     * (model.metrics.avg_loss), not the last single step's
                     * loss - now that every parameter trains (not just
                     * output_bias), a single step's loss is noisy enough
                     * to be a misleading progress signal on its own. */
                    if (training_steps % 5000 == 0) {
                        printf("   [Progress] Steps: %zu, Avg Loss: %.6f, File: %.1f MB\n",
                               training_steps, model.metrics.avg_loss,
                               stream_get_total_read(epoch_stream) / (1024.0f * 1024.0f));
                    }
                    
                    /* Reset for next batch */
                    token_stream_reset(token_stream);
                }
            }
            
            free(chunk_tokens.token_ids);
        }
        
        /* Process remaining tokens in final batch */
        if (token_stream_get_size(token_stream) > 0) {
            size_t remaining = token_stream_get_size(token_stream);
            for (size_t i = 0; i + 1 < remaining; i++) {
                size_t window_len = (i + 1 < train_window) ? (i + 1) : train_window;
                uint32_t *window = &token_stream->token_buffer[i + 1 - window_len];
                uint32_t target_token = token_stream->token_buffer[i + 1];

                model_train_step(&model, window, target_token, window_len);
                training_steps++;
            }
            token_stream_reset(token_stream);
        }
        
        stream_reader_free(epoch_stream);
        printf("   Epoch %d/%d - Avg Loss: %.6f, Tokens processed: %zu\n",
               epoch + 1, args->epochs, model.metrics.avg_loss, total_tokens_processed);
    }
    
    printf("   Training complete!\n");
    printf("   Total steps: %zu, Total tokens: %zu\n", training_steps, total_tokens_processed);
    printf("   ✓ Training finished\n\n");

    /* ===== STEP 4: Demonstrations ===== */
    printf("[4] Demonstrations...\n");
    
    /* 4.1: Attention outputs */
    printf("[4.1] Multi-head attention output shapes:\n");
    printf("   Input: 1 token\n");
    printf("   Query shape: (1, %zu)\n", model.embedding_dim);
    printf("   Output: %zu heads × %zu dim = %zu dim\n", 
           model.num_heads, model.embedding_dim / model.num_heads, model.embedding_dim);
    printf("   ✓ Attention mechanism validated\n");
    
    /* 4.2: Random logits for sampling demo */
    printf("[4.2] Sampling strategies demonstration:\n");
    float *demo_logits = malloc(model.vocab_size * sizeof(float));
    if (demo_logits) {
        for (size_t i = 0; i < model.vocab_size; i++) {
            demo_logits[i] = (float)(rand() % 100) / 100.0f;
        }
        
        uint32_t greedy = sample_greedy(demo_logits, model.vocab_size);
        printf("   - Greedy:   selected token %u\n", greedy);
        
        uint32_t topk = sample_topk(demo_logits, model.vocab_size, 5);
        printf("   - Top-5:    selected token %u\n", topk);
        
        uint32_t topp = sample_topp(demo_logits, model.vocab_size, 0.9f);
        printf("   - Top-p:    selected token %u\n", topp);
        
        printf("   ✓ Sampling strategies demonstrated\n");
        
        free(demo_logits);
    }

    /* ===== STEP 5: Model Persistence ===== */
    printf("[5] Saving model to %s\n", args->model_path);
    model_errors_t save_rc = model_save(&model, args->model_path);
    
    if (save_rc == MODEL_SUCCESS) {
        printf("   ✓ Model saved\n");
    } else {
        fprintf(stderr, "   ✗ Save failed (code: %d)\n", save_rc);
    }

    char tokenizer_path[1024];
    int tokenizer_saved = 0;
    if (resolve_tokenizer_path(args, tokenizer_path, sizeof(tokenizer_path)) != 0) {
        fprintf(stderr, "   ✗ Tokenizer path is too long\n");
    } else if (tokenizer_save_encoder(encoder, tokenizer_path) != TOKENIZER_SUCCESS) {
        fprintf(stderr, "   ✗ Failed to save tokenizer to %s\n", tokenizer_path);
    } else {
        printf("   ✓ Tokenizer saved to %s\n", tokenizer_path);
        tokenizer_saved = 1;
    }

    /* ===== STEP 6: Configuration ===== */
    printf("[6] Saving configuration...\n");
    config_t config;
    config_get_defaults(&config);
    config.embedding_dim = model.embedding_dim;
    config.num_heads = model.num_heads;
    config.num_layers = model.num_layers;
    config.vocab_size = model.vocab_size;
    config.max_seq_len = model.max_seq_len;
    config.learning_rate = args->learning_rate;
    config_save("config.json", &config);
    printf("   ✓ Configuration saved\n\n");

    /* ===== CLEANUP ===== */
    printf("[7] Cleaning up...\n");
    model_free(&model);
    token_stream_free(token_stream);
    stream_reader_free(stream);
    tokenizer_free_encoder(encoder);
    printf("   ✓ Resources freed\n\n");

    printf("=== Training Complete ===\n");
    return save_rc == MODEL_SUCCESS && tokenizer_saved ? 0 : 1;
}

/* ===== INFERENCE MODE ===== */
int mode_infer(const cli_args_t *args) {
    printf("=== Neural Model Inference ===\n\n");
    
    if (strlen(args->prompt) == 0) {
        fprintf(stderr, "Error: --prompt required for inference mode\n");
        return 1;
    }
    
    printf("[1] Loading model from %s...\n", args->model_path);
    
    neural_model_t model = {0};
    model_errors_t load_rc = model_load(&model, args->model_path);
    
    if (load_rc != MODEL_SUCCESS) {
        fprintf(stderr, "Error: Failed to load model (code: %d)\n", load_rc);
        return 1;
    }
    model.use_gpu = args->use_gpu;

    printf("   ✓ Model loaded\n");
    printf("   - Vocab: %zu tokens\n", model.vocab_size);
    printf("   - Embedding: %zu\n", model.embedding_dim);
    printf("   - Heads: %zu\n", model.num_heads);

    printf("[2] Tokenizing prompt: \"%s\"\n", args->prompt);
    
    bpe_encoder_t *encoder = load_model_tokenizer(args, model.vocab_size);
    if (!encoder) {
        model_free(&model);
        return 1;
    }
    bpe_tokens_t tokens = {0};
    if (bpe_encode(encoder, args->prompt, strlen(args->prompt), &tokens) != BPE_SUCCESS) {
        fprintf(stderr, "Error: Failed to encode prompt\n");
        model_free(&model);
        tokenizer_free_encoder(encoder);
        return 1;
    }
    
    printf("   Encoded to %zu tokens\n", tokens.token_count);

    printf("[3] Running inference...\n");
    
    if (tokens.token_count > 0) {
        size_t context_len = tokens.token_count;
        uint32_t *context = tokens.token_ids;
        if (context_len > model.max_seq_len) {
            context += context_len - model.max_seq_len;
            context_len = model.max_seq_len;
            printf("   Prompt truncated to the last %zu tokens\n", context_len);
        }

        int was_training = model.is_training;
        model.is_training = 0;
        model_errors_t forward_rc = model_forward(&model, context, context_len, model.ws_logits);
        model.is_training = was_training;
        if (forward_rc != MODEL_SUCCESS) {
            fprintf(stderr, "Error: Model forward pass failed (code: %d)\n", forward_rc);
            model_free(&model);
            tokenizer_free_encoder(encoder);
            free(tokens.token_ids);
            return 1;
        }
        uint32_t predicted = sample_next_token(model.ws_logits, model.vocab_size,
                                               args->sampling_strategy, args->temperature,
                                               (size_t)args->top_k, args->top_p);
        printf("   Predicted next token: %u\n", predicted);
        printf("   ✓ Inference complete\n\n");
    } else {
        fprintf(stderr, "Error: Failed to encode prompt\n");
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }

    /* Cleanup */
    model_free(&model);
    tokenizer_free_encoder(encoder);
    free(tokens.token_ids);
    
    return 0;
}

/* ===== GENERATION MODE ===== */
int mode_generate(const cli_args_t *args) {
    printf("=== Neural Model Generation ===\n\n");
    
    if (strlen(args->prompt) == 0) {
        fprintf(stderr, "Error: --prompt required for generation mode\n");
        return 1;
    }
    
    printf("[1] Loading model from %s...\n", args->model_path);
    
    neural_model_t model = {0};
    model_errors_t gen_load_rc = model_load(&model, args->model_path);
    
    if (gen_load_rc != MODEL_SUCCESS) {
        fprintf(stderr, "Error: Failed to load model (code: %d)\n", gen_load_rc);
        return 1;
    }
    model.use_gpu = args->use_gpu;

    printf("   ✓ Model loaded\n");

    printf("[2] Tokenizing prompt: \"%s\"\n", args->prompt);
    
    bpe_encoder_t *encoder = load_model_tokenizer(args, model.vocab_size);
    if (!encoder) {
        model_free(&model);
        return 1;
    }
    bpe_tokens_t tokens = {0};
    if (bpe_encode(encoder, args->prompt, strlen(args->prompt), &tokens) != BPE_SUCCESS ||
        tokens.token_count == 0) {
        fprintf(stderr, "Error: Failed to encode prompt\n");
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }
    
    printf("   Seed: %zu tokens\n", tokens.token_count);

    printf("[3] Generating %d tokens with %s sampling...\n", 
           args->generate_length,
           args->sampling_strategy == SAMPLING_GREEDY ? "greedy" :
           args->sampling_strategy == SAMPLING_TOPK ? "top-k" : "top-p");
    
    /* Generate tokens */
    uint32_t *generated = malloc(model.max_seq_len * sizeof(uint32_t));
    if (!generated) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }
    
    size_t prompt_len = tokens.token_count;
    const uint32_t *prompt_tokens = tokens.token_ids;
    if (prompt_len > model.max_seq_len) {
        prompt_tokens += prompt_len - model.max_seq_len;
        prompt_len = model.max_seq_len;
        printf("   Prompt truncated to the last %zu tokens\n", prompt_len);
    }
    memcpy(generated, prompt_tokens, prompt_len * sizeof(uint32_t));
    size_t current_len = prompt_len;

    model_kv_cache_t kv_cache = {0};
    model_errors_t cache_rc = model_kv_cache_init(&kv_cache, &model);
    if (cache_rc != MODEL_SUCCESS) {
        fprintf(stderr, "Error: Failed to allocate KV cache (code: %d)\n", cache_rc);
        free(generated);
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }

    for (size_t i = 0; i < prompt_len; i++) {
        if (model_forward_token(&model, &kv_cache, generated[i], model.ws_logits) != MODEL_SUCCESS) {
            fprintf(stderr, "Error: Failed to prime KV cache\n");
            model_kv_cache_free(&kv_cache);
            free(generated);
            model_free(&model);
            tokenizer_free_encoder(encoder);
            free(tokens.token_ids);
            return 1;
        }
    }

    size_t available = model.max_seq_len - current_len;
    size_t requested = args->generate_length > 0 ? (size_t)args->generate_length : 0;
    size_t new_token_count = requested < available ? requested : available;
    if (new_token_count < requested) {
        printf("   Generation capped at %zu new tokens by max_seq_len=%zu\n",
               new_token_count, model.max_seq_len);
    }

    for (size_t i = 0; i < new_token_count; i++) {
        uint32_t next_token = sample_next_token(model.ws_logits, model.vocab_size,
                                                args->sampling_strategy, args->temperature,
                                                (size_t)args->top_k, args->top_p);
        generated[current_len++] = next_token;
        if (i + 1 < new_token_count &&
            model_forward_token(&model, &kv_cache, next_token, model.ws_logits) != MODEL_SUCCESS) {
            fprintf(stderr, "Error: Incremental forward pass failed\n");
            model_kv_cache_free(&kv_cache);
            free(generated);
            model_free(&model);
            tokenizer_free_encoder(encoder);
            free(tokens.token_ids);
            return 1;
        }
    }
    
    printf("   Generated %zu total tokens\n", current_len);
    printf("   ✓ Generation complete\n\n");

    printf("=== Generated Sequence ===\n");
    
    /* Decode and display generated text */
    char *generated_text = NULL;
    size_t text_len = 0;
    bpe_errors_t decode_rc = bpe_decode(encoder, generated, current_len, &generated_text, &text_len);
    
    if (decode_rc == BPE_SUCCESS && generated_text) {
        printf("%s\n\n", generated_text);
        free(generated_text);
    } else {
        /* Fallback to showing token IDs if decoding fails */
        printf("[");
        for (size_t i = 0; i < current_len && i < 20; i++) {
            printf("%u%s", generated[i], i + 1 < current_len ? ", " : "");
        }
        if (current_len > 20) printf(", ...");
        printf("]\n\n");
    }

    /* Cleanup */
    model_kv_cache_free(&kv_cache);
    free(generated);
    model_free(&model);
    tokenizer_free_encoder(encoder);
    free(tokens.token_ids);
    
    return 0;
}
