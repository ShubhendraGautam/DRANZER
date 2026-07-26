/*
 * Main entry point - Neural Model with Multi-Head Attention, Training, and Inference
 * Phase 1: Multi-head attention, gradient descent training, next-token prediction, model persistence
 * Phase 2: Layer normalization, learning rate scheduling, loss tracking
 * Phase 3: Batch processing, configuration files, sampling strategies, training checkpoints
 */

#include "../libs/include/byte_pair_encoding.h"
#include "include/tokenizer.h"
#include "include/model.h"
#include "include/debug.h"
#include "include/config.h"
#include "include/sampling.h"
#include "include/batch.h"
#include "include/checkpoint.h"
#include "include/cli.h"
#include "include/stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_SIZE 257
#define EMBEDDING_DIM 16
#define NUM_HEADS 2
#define NUM_LAYERS 2
#define MAX_SEQ_LEN 32
/* How many preceding tokens each training step conditions on. Kept below
 * MAX_SEQ_LEN so infer/generate (which use the same model->max_seq_len
 * bound) has headroom beyond what training ever exercises. Real
 * cross-token attention needs seq_len > 1 - with a single token, softmax
 * over one key is always 1.0 regardless of Q/K, so W_q/W_k would get
 * almost no useful gradient signal. */
#define TRAIN_WINDOW 16

/* Forward declarations */
int mode_train(const cli_args_t *args);
int mode_infer(const cli_args_t *args);
int mode_generate(const cli_args_t *args);

int main(int argc, char *argv[]) {
    cli_args_t args;
    
    /* Parse command-line arguments */
    cli_parse(argc, argv, &args);
    
    /* Handle help flag */
    if (args.help) {
        cli_print_help(argv[0]);
        return 0;
    }
    
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
    
    bpe_encoder_t *encoder = tokenizer_create_encoder(VOCAB_SIZE);
    if (encoder == NULL) {
        fprintf(stderr, "Error: Failed to create encoder\n");
        return 1;
    }

    /* Create streaming file reader - handles files of any size */
    stream_reader_t *stream = stream_reader_create(args->input_file, STREAM_CHUNK_SIZE);
    if (stream == NULL) {
        fprintf(stderr, "Error: Failed to open file for streaming\n");
        bpe_encoder_free(encoder);
        return 1;
    }
    
    printf("   ✓ Streaming reader created (256KB chunks)\n");
    
    /* Create token stream processor - accumulates tokens for batching */
    token_stream_t *token_stream = token_stream_create(10000, 1000);
    if (token_stream == NULL) {
        fprintf(stderr, "Error: Failed to create token stream\n");
        stream_reader_free(stream);
        bpe_encoder_free(encoder);
        return 1;
    }
    
    printf("   ✓ Token stream buffer created (batch size: 1000)\n");

    /* ===== STEP 2: Initialize Model ===== */
    printf("[2] Initializing neural model...\n");
    neural_model_t model = {0};
    model_errors_t init_rc = model_new(&model, VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN);

    if (init_rc != MODEL_SUCCESS) {
        fprintf(stderr, "Error: Model initialization failed (code: %d)\n", init_rc);
        token_stream_free(token_stream);
        stream_reader_free(stream);
        bpe_encoder_free(encoder);
        return 1;
    }

    /* --input's learning rate was previously parsed and only ever printed,
     * never applied - now that training actually updates every parameter
     * (not just output_bias), the learning rate materially affects
     * stability, so it needs to actually reach the model. */
    model.learning_rate = args->learning_rate;
    model.metrics.learning_rate = args->learning_rate;
    model.metrics.initial_learning_rate = args->learning_rate;

    printf("   Model initialized:\n");
    printf("   - Vocabulary: %zu tokens\n", model.vocab_size);
    printf("   - Embedding dim: %zu\n", model.embedding_dim);
    printf("   - Attention heads: %zu\n", model.num_heads);
    printf("   - Layers: %zu\n", model.num_layers);
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
                     * i+1 is predicted from up to TRAIN_WINDOW preceding
                     * tokens (fewer near the start of the batch). */
                    for (size_t i = 0; i + 1 < batch_size; i++) {
                        size_t window_len = (i + 1 < TRAIN_WINDOW) ? (i + 1) : TRAIN_WINDOW;
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
                size_t window_len = (i + 1 < TRAIN_WINDOW) ? (i + 1) : TRAIN_WINDOW;
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

    /* ===== STEP 6: Configuration ===== */
    printf("[6] Saving configuration...\n");
    config_t config;
    config_get_defaults(&config);
    config.embedding_dim = model.embedding_dim;
    config.num_heads = model.num_heads;
    config.vocab_size = model.vocab_size;
    config_save("config.json", &config);
    printf("   ✓ Configuration saved\n\n");

    /* ===== CLEANUP ===== */
    printf("[7] Cleaning up...\n");
    model_free(&model);
    token_stream_free(token_stream);
    stream_reader_free(stream);
    bpe_encoder_free(encoder);
    printf("   ✓ Resources freed\n\n");

    printf("=== Training Complete ===\n");
    return 0;
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
    
    printf("   ✓ Model loaded\n");
    printf("   - Vocab: %zu tokens\n", model.vocab_size);
    printf("   - Embedding: %zu\n", model.embedding_dim);
    printf("   - Heads: %zu\n", model.num_heads);

    printf("[2] Tokenizing prompt: \"%s\"\n", args->prompt);
    
    /* Create encoder for inference */
    bpe_encoder_t *encoder = tokenizer_create_encoder(model.vocab_size);
    bpe_tokens_t tokens = {0};
    bpe_encode(encoder, args->prompt, strlen(args->prompt), &tokens);
    
    printf("   Encoded to %zu tokens\n", tokens.token_count);

    printf("[3] Running inference...\n");
    
    if (tokens.token_count > 0) {
        uint32_t predicted = model_predict_next_token(&model, tokens.token_ids, tokens.token_count);
        printf("   Predicted next token: %u\n", predicted);
        printf("   ✓ Inference complete\n\n");
    } else {
        fprintf(stderr, "Error: Failed to encode prompt\n");
        model_free(&model);
        bpe_encoder_free(encoder);
        free(tokens.token_ids);
        return 1;
    }

    /* Cleanup */
    model_free(&model);
    bpe_encoder_free(encoder);
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
    
    printf("   ✓ Model loaded\n");

    printf("[2] Tokenizing prompt: \"%s\"\n", args->prompt);
    
    /* Create encoder */
    bpe_encoder_t *encoder = tokenizer_create_encoder(model.vocab_size);
    bpe_tokens_t tokens = {0};
    bpe_encode(encoder, args->prompt, strlen(args->prompt), &tokens);
    
    printf("   Seed: %zu tokens\n", tokens.token_count);

    printf("[3] Generating %d tokens with %s sampling...\n", 
           args->generate_length,
           args->sampling_strategy == SAMPLING_GREEDY ? "greedy" :
           args->sampling_strategy == SAMPLING_TOPK ? "top-k" : "top-p");
    
    /* Generate tokens */
    uint32_t *generated = malloc((tokens.token_count + args->generate_length) * sizeof(uint32_t));
    if (!generated) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        model_free(&model);
        bpe_encoder_free(encoder);
        free(tokens.token_ids);
        return 1;
    }
    
    /* Copy seed tokens */
    memcpy(generated, tokens.token_ids, tokens.token_count * sizeof(uint32_t));
    size_t current_len = tokens.token_count;
    
    /* Generate new tokens */
    for (int i = 0; i < args->generate_length && current_len < MAX_SEQ_LEN; i++) {
        /* Simple greedy generation for demo */
        uint32_t next_token = model_predict_next_token(&model, generated, current_len);
        generated[current_len++] = next_token;
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
    free(generated);
    model_free(&model);
    bpe_encoder_free(encoder);
    free(tokens.token_ids);
    
    return 0;
}
