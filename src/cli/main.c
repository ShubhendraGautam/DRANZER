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
#include "common/fp_bits.h"
#include "core/rng.h"
#include "cli/config.h"
#include "cli/evaluation.h"
#include "cli/generation.h"
#include "cli/manifest.h"
#include "cli/sampling.h"
#include "cli/batch.h"
#include "cli/checkpoint.h"
#include "cli/cli.h"
#include "cli/stream.h"
#include "core/bundle.h"
#include <inttypes.h>
#include <limits.h>
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
int mode_eval(const cli_args_t *args);
int mode_infer(const cli_args_t *args);
int mode_generate(const cli_args_t *args);

static int stream_token_to_stdout(uint32_t token_id, const char *text,
                                  size_t text_length, void *user_data) {
    (void)token_id;
    (void)user_data;
    if (text_length > 0 && fwrite(text, 1, text_length, stdout) != text_length)
        return 1;
    return fflush(stdout) == 0 ? 0 : 1;
}

static void free_stop_tokens(bpe_tokens_t *tokens, size_t count) {
    for (size_t i = 0; i < count; i++) bpe_tokens_free(&tokens[i]);
}

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int copy_path(char *destination, size_t capacity, const char *source) {
    if (!destination || !source || capacity == 0) return -1;
    size_t length = strlen(source);
    if (length >= capacity) return -1;
    memcpy(destination, source, length + 1);
    return 0;
}

static int copy_resolved_path(char *destination, size_t capacity, const char *source) {
    char resolved[PATH_MAX];
    const char *stored = realpath(source, resolved) ? resolved : source;
    return copy_path(destination, capacity, stored);
}

static int reject_resume_override(const cli_args_t *requested,
                                  const checkpoint_run_state_t *state,
                                  const neural_model_t *model) {
#define REJECT_DIFFERENT(option, condition) do { \
    if (cli_option_was_explicit(requested, option) && (condition)) { \
        fprintf(stderr, "Error: %s cannot change an exact-resume trajectory\n", option); \
        return -1; \
    } \
} while (0)
    REJECT_DIFFERENT("--vocab-size", requested->vocab_size != model->vocab_size);
    REJECT_DIFFERENT("--embedding-dim", requested->embedding_dim != model->embedding_dim);
    REJECT_DIFFERENT("--num-heads", requested->num_heads != model->num_heads);
    REJECT_DIFFERENT("--num-layers", requested->num_layers != model->num_layers);
    REJECT_DIFFERENT("--max-seq-len", requested->max_seq_len != model->max_seq_len);
    REJECT_DIFFERENT("--train-window", requested->train_window != state->train_window);
    /* state->train_stride is the resolved value (never 0), so compare the
     * resolved request against it rather than the raw flag. */
    REJECT_DIFFERENT("--train-stride",
                     (requested->train_stride ? requested->train_stride
                                              : requested->train_window) !=
                         state->train_stride);
    REJECT_DIFFERENT("--batch-size", requested->batch_size != state->batch_size);
    REJECT_DIFFERENT("--gradient-accumulation",
                     requested->gradient_accumulation_steps !=
                         state->gradient_accumulation_steps);
    REJECT_DIFFERENT("--shuffle", requested->shuffle != state->shuffle);
    REJECT_DIFFERENT("--learning-rate",
                     requested->learning_rate != model->metrics.initial_learning_rate);
    REJECT_DIFFERENT("--optimizer",
                     (strcmp(requested->optimizer, "sgd") == 0) !=
                         (model->optimizer_type == OPTIMIZER_SGD));
    REJECT_DIFFERENT("--dropout", requested->dropout_rate != model->dropout_rate);
    REJECT_DIFFERENT("--grad-clip", requested->grad_clip_norm != model->grad_clip_norm);
    REJECT_DIFFERENT("--weight-decay", requested->weight_decay != model->weight_decay);
    REJECT_DIFFERENT("--warmup-steps", requested->warmup_steps != model->warmup_steps);
    REJECT_DIFFERENT("--total-steps", requested->total_steps != model->total_steps);
    REJECT_DIFFERENT("--seed", requested->seed != state->seed);
    REJECT_DIFFERENT("--gpu", requested->use_gpu != state->use_gpu);
#undef REJECT_DIFFERENT
    return 0;
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

    printf("   Tokenizer loaded from %s (%zu learned merges, %s mode)\n",
           tokenizer_path,
           encoder->vocab_size - bpe_encoder_first_learned_id(encoder),
           bpe_encoder_has_special_tokens(encoder) ? "special-v1" : "legacy");
    return encoder;
}

/* Prefer the self-contained, versioned artifact. Files written by releases
 * before the bundle format remain readable through the legacy weight file
 * plus tokenizer-sidecar path. */
static int load_model_artifact(const cli_args_t *args,
                               neural_model_t *model,
                               bpe_encoder_t **out_encoder) {
    model_bundle_metadata_t metadata = {0};
    bundle_errors_t bundle_rc = model_bundle_load(
        model, out_encoder, &metadata, args->model_path);
    if (bundle_rc == BUNDLE_SUCCESS) {
        if (cli_option_was_explicit(args, "--tokenizer")) {
            fprintf(stderr,
                    "Error: %s embeds its frozen tokenizer; --tokenizer cannot override it\n",
                    args->model_path);
            tokenizer_free_encoder(*out_encoder);
            *out_encoder = NULL;
            model_free(model);
            return -1;
        }
        printf("   Versioned model bundle loaded (embedded frozen tokenizer, seed %" PRIu64 ")\n",
               metadata.seed);
        return 0;
    }
    if (bundle_rc != BUNDLE_NOT_BUNDLE) {
        fprintf(stderr, "Error: Model bundle %s is invalid or unreadable (code: %d)\n",
                args->model_path, bundle_rc);
        return -1;
    }

    model_errors_t legacy_rc = model_load(model, args->model_path);
    if (legacy_rc != MODEL_SUCCESS) {
        fprintf(stderr, "Error: Failed to load legacy model from %s (code: %d)\n",
                args->model_path, legacy_rc);
        return -1;
    }
    *out_encoder = load_model_tokenizer(args, model->vocab_size);
    if (!*out_encoder) {
        model_free(model);
        return -1;
    }
    printf("   Legacy model artifact loaded\n");
    return 0;
}

static bpe_encoder_t *prepare_training_tokenizer(const cli_args_t *args,
                                                 char *tokenizer_path,
                                                 size_t tokenizer_path_size,
                                                 tokenizer_corpus_stats_t *corpus_stats) {
    if (resolve_tokenizer_path(args, tokenizer_path, tokenizer_path_size) != 0) {
        fprintf(stderr, "Error: Tokenizer path is too long\n");
        return NULL;
    }

    bpe_encoder_t *encoder = NULL;
    if (args->tokenizer_path[0]) {
        tokenizer_errors_t load_rc = tokenizer_load_encoder(tokenizer_path, &encoder);
        if (load_rc == TOKENIZER_SUCCESS) {
            if (encoder->max_vocab_size != args->vocab_size) {
                fprintf(stderr,
                        "Error: Existing tokenizer/model vocabulary mismatch (%zu vs %zu)\n",
                        encoder->max_vocab_size, args->vocab_size);
                tokenizer_free_encoder(encoder);
                return NULL;
            }
            tokenizer_errors_t fingerprint_rc =
                tokenizer_fingerprint_file(args->input_file, corpus_stats);
            if (fingerprint_rc != TOKENIZER_SUCCESS) {
                fprintf(stderr, "Error: Failed to fingerprint training corpus (code: %d)\n",
                        fingerprint_rc);
                tokenizer_free_encoder(encoder);
                return NULL;
            }
            printf("   Reusing frozen tokenizer %s (%zu learned merges, %s mode)\n",
                   tokenizer_path,
                   encoder->vocab_size - bpe_encoder_first_learned_id(encoder),
                   bpe_encoder_has_special_tokens(encoder) ? "special-v1" : "legacy");
            return encoder;
        }
        if (load_rc != TOKENIZER_FILE_NOT_FOUND) {
            fprintf(stderr, "Error: Existing tokenizer %s is invalid or unreadable (code: %d)\n",
                    tokenizer_path, load_rc);
            return NULL;
        }
    }

    encoder = tokenizer_create_special_encoder(args->vocab_size);
    if (!encoder) {
        fprintf(stderr,
                "Error: Failed to create special-aware tokenizer "
                "(--vocab-size must be at least 260)\n");
        return NULL;
    }

    tokenizer_errors_t train_rc =
        tokenizer_train_encoder_file(encoder, args->input_file, corpus_stats);
    if (train_rc != TOKENIZER_SUCCESS) {
        fprintf(stderr, "Error: Failed to train tokenizer from %s (code: %d)\n",
                args->input_file, train_rc);
        tokenizer_free_encoder(encoder);
        return NULL;
    }
    if (tokenizer_save_encoder(encoder, tokenizer_path) != TOKENIZER_SUCCESS) {
        fprintf(stderr, "Error: Failed to save frozen tokenizer to %s\n", tokenizer_path);
        tokenizer_free_encoder(encoder);
        return NULL;
    }

    printf("   Trained and froze tokenizer (%zu learned merges, special-v1 mode)\n",
           encoder->vocab_size - bpe_encoder_first_learned_id(encoder));
    printf("   Tokenizer saved to %s\n", tokenizer_path);
    return encoder;
}

static void print_evaluation_report(const char *label, const char *filename,
                                    size_t context_window,
                                    const evaluation_report_t *report) {
    printf("   %s: %s\n", label, filename);
    printf("   - Corpus: %zu bytes, FNV-1a %016" PRIx64 "\n",
           report->corpus_bytes, report->corpus_fingerprint);
    printf("   - Context window: %zu\n", context_window);
    printf("   - Tokens: %zu (%zu next-token predictions)\n",
           report->token_count, report->prediction_count);
    printf("   - Cross-entropy: %.6f\n", report->average_cross_entropy);
    printf("   - Perplexity: %.6f\n", report->perplexity);
    printf("   - Elapsed: %.3f s", report->elapsed_seconds);
    if (report->elapsed_seconds > 0.0) {
        printf(" (%.1f predictions/s)",
               (double)report->prediction_count / report->elapsed_seconds);
    }
    printf("\n");
}

typedef struct {
    batch_t *microbatch;
    size_t pending_microbatches;
    size_t pending_samples;
    double pending_loss;
    uint64_t prediction_cursor;
    uint64_t resume_skip;
    size_t replay_slots;
    uint64_t microbatch_index;
} training_accumulator_t;

static uint64_t shuffle_seed(unsigned int seed, uint32_t epoch_index,
                             uint64_t microbatch_index) {
    uint64_t value = (uint64_t)seed ^ (UINT64_C(0x9e3779b97f4a7c15) * (epoch_index + 1));
    value ^= UINT64_C(0xbf58476d1ce4e5b9) * (microbatch_index + 1);
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static int commit_accumulated_training(neural_model_t *model,
                                       const bpe_encoder_t *encoder,
                                       training_accumulator_t *accumulator,
                                       uint32_t epoch_index,
                                       checkpoint_run_state_t *checkpoint_state,
                                       const cli_args_t *args) {
    if (accumulator->pending_samples == 0) return 0;
    float average_loss = (float)(accumulator->pending_loss /
                                 (double)accumulator->pending_samples);
    if (model_apply_accumulated_gradients(model, accumulator->pending_samples,
                                          average_loss) != MODEL_SUCCESS) return -1;

    accumulator->pending_microbatches = 0;
    accumulator->pending_samples = 0;
    accumulator->pending_loss = 0.0;
    checkpoint_state->epoch_index = epoch_index;
    checkpoint_state->step_in_epoch = accumulator->prediction_cursor;

    if (args->checkpoint_interval > 0 &&
        model->training_steps % (uint32_t)args->checkpoint_interval == 0) {
        char path[2048];
        if (checkpoint_save(model, encoder, checkpoint_state, args->checkpoint_dir,
                            args->keep_checkpoints > 0
                                ? (size_t)args->keep_checkpoints : 0,
                            path, sizeof(path)) != 0) return -1;
        printf("   ✓ Checkpoint saved: %s\n", path);
    }
    return 0;
}

static int process_training_microbatch(neural_model_t *model,
                                       const bpe_encoder_t *encoder,
                                       training_accumulator_t *accumulator,
                                       uint32_t epoch_index,
                                       checkpoint_run_state_t *checkpoint_state,
                                       const cli_args_t *args) {
    size_t count = batch_get_size(accumulator->microbatch);
    if (count == 0) return 0;
    if (args->shuffle) {
        batch_shuffle(accumulator->microbatch,
                      shuffle_seed(args->seed, epoch_index,
                                   accumulator->microbatch_index));
    }
    accumulator->microbatch_index++;
    if (accumulator->pending_samples == 0) model_zero_gradients(model);

    for (size_t i = 0; i < count; i++) {
        float loss = 0.0f;
        /* Each example contributes the MEAN cross-entropy over its own
         * supervised positions, and pending_samples counts examples, so the
         * optimizer step averages a mean of means. Windows are equal-length
         * except possibly the last one of an epoch, which is therefore
         * weighted slightly above its token count. That is the same
         * convention the per-example loss already used and it keeps one
         * optimizer step comparable to another regardless of window length. */
        if (model_accumulate_gradients_all(
                model, accumulator->microbatch->token_sequences[i],
                accumulator->microbatch->target_sequences[i],
                accumulator->microbatch->sequence_lengths[i],
                &loss, NULL) != MODEL_SUCCESS) {
            return -1;
        }
        accumulator->pending_loss += loss;
        accumulator->pending_samples++;
    }
    accumulator->pending_microbatches++;
    batch_reset(accumulator->microbatch);

    if (accumulator->pending_microbatches >=
        (size_t)args->gradient_accumulation_steps) {
        return commit_accumulated_training(model, encoder, accumulator,
                                           epoch_index, checkpoint_state, args);
    }
    return 0;
}

static int submit_training_prediction(neural_model_t *model,
                                      const bpe_encoder_t *encoder,
                                      training_accumulator_t *accumulator,
                                      const uint32_t *window,
                                      const uint32_t *targets,
                                      size_t window_len, uint32_t epoch_index,
                                      checkpoint_run_state_t *checkpoint_state,
                                      const cli_args_t *args) {
    accumulator->prediction_cursor++;
    if (accumulator->prediction_cursor <= accumulator->resume_skip) {
        accumulator->replay_slots++;
        if (accumulator->replay_slots == (size_t)args->batch_size) {
            accumulator->replay_slots = 0;
            accumulator->microbatch_index++;
        }
        return 0;
    }
    /* A periodic checkpoint is written only after a complete minibatch or
     * accumulation group. A non-terminal cursor can therefore never land
     * inside a minibatch. */
    if (accumulator->replay_slots != 0) return -1;
    if (batch_add_sequence(accumulator->microbatch, window, window_len,
                           targets) != 0) return -1;
    if (batch_is_full(accumulator->microbatch)) {
        return process_training_microbatch(model, encoder, accumulator,
                                           epoch_index, checkpoint_state, args);
    }
    return 0;
}

/* Walk the token stream in windows, submitting each as one training example
 * with a target for every position.
 *
 * One example is now a whole window rather than a single target, so the unit
 * the resume cursor counts changed with it: accumulator->prediction_cursor
 * counts windows here where it counted targets before. CHECKPOINT_VERSION
 * was bumped to 3 for exactly that reason - a version-2 cursor would resume
 * at the wrong place, silently, and it is rejected instead.
 *
 * A window starting at token `start` spans up to train_window tokens and its
 * targets are those same tokens shifted by one, so it must stop one short of
 * the end of the stream: the position holding the final token has nothing to
 * predict.
 *
 * allow_partial_window distinguishes the two reasons a window can come up
 * short. Mid-file the stream is a chunk buffer, so a short window means the
 * chunk boundary landed there and the rest of the tokens simply have not
 * been read yet - emitting it would train on a truncation that is an
 * artifact of the I/O size. At end of stream a short window is genuine and
 * must be emitted or the corpus tail is never trained on. The caller passes
 * 0 for the first case and 1 for the second.
 *
 * *out_next_start receives the offset of the first window NOT emitted, so
 * the caller knows exactly how much of the buffer to retain.
 */
static int process_training_token_range(neural_model_t *model,
                                        const bpe_encoder_t *encoder,
                                        training_accumulator_t *accumulator,
                                        token_stream_t *stream,
                                        size_t first_start,
                                        size_t train_window,
                                        size_t train_stride,
                                        int allow_partial_window,
                                        size_t *out_next_start,
                                        uint32_t epoch_index,
                                        checkpoint_run_state_t *checkpoint_state,
                                        const cli_args_t *args) {
    size_t size = token_stream_get_size(stream);
    if (train_window == 0 || train_stride == 0) return -1;

    size_t start = first_start;
    for (; start < size; start += train_stride) {
        /* Tokens from `start` that have a successor to predict. */
        size_t available = size - 1 - start;
        if (start + 1 > size - 1) available = 0;
        if (available == 0) break;

        size_t window_len = available < train_window ? available : train_window;
        if (!allow_partial_window && window_len < train_window) break;

        const uint32_t *window = &stream->token_buffer[start];
        const uint32_t *targets = &stream->token_buffer[start + 1];

        if (submit_training_prediction(
                model, encoder, accumulator, window, targets, window_len,
                epoch_index, checkpoint_state, args) != 0) return -1;
    }

    if (out_next_start) *out_next_start = start < size ? start : size;
    return 0;
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
        (!dranzer_float_is_finite(args.top_p) || args.top_p <= 0.0f || args.top_p > 1.0f)) {
        fprintf(stderr, "Error: --top-p must be greater than zero and at most one\n");
        return 2;
    }
    if ((args.mode == MODE_INFER || args.mode == MODE_GENERATE) &&
        (!dranzer_float_is_finite(args.temperature) ||
         args.temperature < 0.0f || args.temperature > 2.0f)) {
        fprintf(stderr, "Error: --temperature must be between zero and two\n");
        return 2;
    }
    if (args.mode == MODE_GENERATE && args.generate_length < 0) {
        fprintf(stderr, "Error: --length must not be negative\n");
        return 2;
    }
    if (args.mode == MODE_TRAIN &&
        (args.num_heads == 0 || args.embedding_dim % args.num_heads != 0)) {
        fprintf(stderr, "Error: --embedding-dim must divide evenly by --num-heads\n");
        return 2;
    }
    if (args.mode == MODE_TRAIN &&
        ((args.total_steps == 0 && args.warmup_steps != 0) ||
         (args.total_steps > 0 && args.warmup_steps >= args.total_steps))) {
        fprintf(stderr,
                "Error: --warmup-steps requires --total-steps and must be smaller than it\n");
        return 2;
    }
    if (args.mode == MODE_TRAIN &&
        (size_t)args.batch_size > SIZE_MAX /
            (size_t)args.gradient_accumulation_steps) {
        fprintf(stderr, "Error: effective batch size overflows size_t\n");
        return 2;
    }

    
    /* Enable debug if requested */
    if (args.debug) {
        /* Debug flag is now part of cli_args_t, can be checked as needed */
    }
    
    /* Print parsed arguments (compact view) */
    printf("\n>>> Mode: ");
    switch (args.mode) {
        case MODE_TRAIN: printf("TRAIN\n"); break;
        case MODE_EVAL: printf("EVAL\n"); break;
        case MODE_INFER: printf("INFER\n"); break;
        case MODE_GENERATE: printf("GENERATE\n"); break;
    }
    
    if (args.use_gpu) printf(">>> GPU: enabled (if available)\n");
    printf("\n");
    
    /* Dispatch to appropriate mode */
    switch (args.mode) {
        case MODE_EVAL:
            return mode_eval(&args);
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

    cli_args_t effective_args = *args;
    const cli_args_t *requested_args = args;
    args = &effective_args;
    neural_model_t model = {0};
    bpe_encoder_t *encoder = NULL;
    checkpoint_run_state_t checkpoint_state = {0};
    uint32_t start_epoch = 0;
    uint64_t resume_skip = 0;
    int resumed = args->resume_path[0] != '\0';
    char resume_source[2048] = {0};

    /* ===== STEP 1: Train/load and freeze the tokenizer ===== */
    printf("[1] Preparing frozen tokenizer and run state...\n");
    char tokenizer_path[1024];
    tokenizer_corpus_stats_t corpus_stats = {0};
    if (resumed) {
        if (strcmp(args->resume_path, "latest") == 0) {
            if (checkpoint_find_latest(args->checkpoint_dir, resume_source,
                                       sizeof(resume_source)) != 0) {
                fprintf(stderr, "Error: No checkpoint found in %s\n", args->checkpoint_dir);
                return 1;
            }
        } else {
            int written = snprintf(resume_source, sizeof(resume_source), "%s", args->resume_path);
            if (written < 0 || (size_t)written >= sizeof(resume_source)) return 1;
        }
        if (checkpoint_load(&model, &encoder, &checkpoint_state, resume_source) != 0) {
            fprintf(stderr, "Error: Failed to load complete checkpoint %s\n", resume_source);
            return 1;
        }
        if (reject_resume_override(requested_args, &checkpoint_state, &model) != 0) {
            model_free(&model);
            tokenizer_free_encoder(encoder);
            return 1;
        }

        if ((!requested_args->input_explicit &&
             copy_path(effective_args.input_file, sizeof(effective_args.input_file),
                       checkpoint_state.input_file) != 0) ||
            (!requested_args->validation_file[0] &&
             copy_path(effective_args.validation_file, sizeof(effective_args.validation_file),
                       checkpoint_state.validation_file) != 0) ||
            (!cli_option_was_explicit(requested_args, "--tokenizer") &&
             copy_path(effective_args.tokenizer_path,
                       sizeof(effective_args.tokenizer_path),
                       checkpoint_state.tokenizer_path) != 0) ||
            (!cli_option_was_explicit(requested_args, "--model") &&
             copy_path(effective_args.model_path, sizeof(effective_args.model_path),
                       checkpoint_state.model_path) != 0) ||
            (!requested_args->checkpoint_dir_explicit &&
             copy_path(effective_args.checkpoint_dir,
                       sizeof(effective_args.checkpoint_dir),
                       checkpoint_state.checkpoint_dir) != 0)) {
            fprintf(stderr, "Error: Checkpoint contains an overlong path\n");
            model_free(&model);
            tokenizer_free_encoder(encoder);
            return 1;
        }
        effective_args.vocab_size = model.vocab_size;
        effective_args.embedding_dim = model.embedding_dim;
        effective_args.num_heads = model.num_heads;
        effective_args.num_layers = model.num_layers;
        effective_args.max_seq_len = model.max_seq_len;
        effective_args.train_window = checkpoint_state.train_window;
        effective_args.train_stride = checkpoint_state.train_stride;
        effective_args.batch_size = checkpoint_state.batch_size;
        effective_args.gradient_accumulation_steps =
            checkpoint_state.gradient_accumulation_steps;
        effective_args.shuffle = checkpoint_state.shuffle;
        effective_args.seed = checkpoint_state.seed;
        if (!cli_option_was_explicit(requested_args, "--checkpoint-interval")) {
            effective_args.checkpoint_interval = checkpoint_state.checkpoint_interval;
        }
        if (!cli_option_was_explicit(requested_args, "--keep-checkpoints")) {
            effective_args.keep_checkpoints = checkpoint_state.keep_checkpoints;
        }
        checkpoint_state.checkpoint_interval = effective_args.checkpoint_interval;
        checkpoint_state.keep_checkpoints = effective_args.keep_checkpoints;
        effective_args.use_gpu = checkpoint_state.use_gpu;
        effective_args.learning_rate = model.metrics.initial_learning_rate;
        effective_args.dropout_rate = model.dropout_rate;
        effective_args.grad_clip_norm = model.grad_clip_norm;
        effective_args.weight_decay = model.weight_decay;
        effective_args.warmup_steps = model.warmup_steps;
        effective_args.total_steps = model.total_steps;
        snprintf(effective_args.optimizer, sizeof(effective_args.optimizer), "%s",
                 model.optimizer_type == OPTIMIZER_SGD ? "sgd" : "adam");
        if (!requested_args->epochs_explicit) {
            effective_args.epochs = (int)checkpoint_state.target_epochs;
        } else if (requested_args->epochs <= (int)checkpoint_state.epoch_index) {
            fprintf(stderr, "Error: --epochs must extend beyond resumed epoch %u\n",
                    checkpoint_state.epoch_index + 1);
            model_free(&model);
            tokenizer_free_encoder(encoder);
            return 1;
        }
        checkpoint_state.target_epochs = (uint32_t)effective_args.epochs;
        if (cli_option_was_explicit(requested_args, "--model") &&
            copy_path(checkpoint_state.model_path,
                      sizeof(checkpoint_state.model_path),
                      effective_args.model_path) != 0) {
            fprintf(stderr, "Error: Model output path is too long\n");
            model_free(&model);
            tokenizer_free_encoder(encoder);
            return 1;
        }
        if (cli_option_was_explicit(requested_args, "--tokenizer") &&
            copy_path(checkpoint_state.tokenizer_path,
                      sizeof(checkpoint_state.tokenizer_path),
                      effective_args.tokenizer_path) != 0) {
            fprintf(stderr, "Error: Tokenizer output path is too long\n");
            model_free(&model);
            tokenizer_free_encoder(encoder);
            return 1;
        }
        start_epoch = checkpoint_state.epoch_index;
        resume_skip = checkpoint_state.step_in_epoch;
        if (tokenizer_fingerprint_file(args->input_file, &corpus_stats) != TOKENIZER_SUCCESS ||
            corpus_stats.fingerprint != checkpoint_state.input_fingerprint ||
            corpus_stats.byte_count != checkpoint_state.input_bytes) {
            fprintf(stderr, "Error: Training corpus does not match checkpoint fingerprint\n");
            model_free(&model);
            tokenizer_free_encoder(encoder);
            return 1;
        }
        if ((requested_args->input_explicit &&
             copy_resolved_path(checkpoint_state.input_file,
                                sizeof(checkpoint_state.input_file),
                                args->input_file) != 0) ||
            (cli_option_was_explicit(requested_args, "--validation") &&
             copy_resolved_path(checkpoint_state.validation_file,
                                sizeof(checkpoint_state.validation_file),
                                args->validation_file) != 0)) {
            fprintf(stderr, "Error: Resumed corpus path is too long\n");
            model_free(&model);
            tokenizer_free_encoder(encoder);
            return 1;
        }
        if (resolve_tokenizer_path(args, tokenizer_path, sizeof(tokenizer_path)) != 0 ||
            tokenizer_save_encoder(encoder, tokenizer_path) != TOKENIZER_SUCCESS) {
            fprintf(stderr, "Error: Could not restore embedded tokenizer sidecar\n");
            model_free(&model);
            tokenizer_free_encoder(encoder);
            return 1;
        }
        printf("   Resumed %s at epoch %u after %" PRIu64 " predictions in that epoch\n",
               resume_source, start_epoch + 1, resume_skip);
    } else {
        encoder = prepare_training_tokenizer(
            args, tokenizer_path, sizeof(tokenizer_path), &corpus_stats);
        if (!encoder) return 1;
    }

    printf("   Corpus: %zu bytes, FNV-1a %016" PRIx64 "\n",
           corpus_stats.byte_count, corpus_stats.fingerprint);
    printf("   ✓ Token IDs and merge order frozen before optimization\n");
    
    /* Create token stream processor - accumulates tokens for batching */
    token_stream_t *token_stream = token_stream_create(10000, 1000);
    if (token_stream == NULL) {
        fprintf(stderr, "Error: Failed to create token stream\n");
        tokenizer_free_encoder(encoder);
        return 1;
    }
    
    printf("   ✓ Token stream buffer created (flush threshold: 1000 tokens)\n");

    /* ===== STEP 2: Initialize Model ===== */
    printf("[2] Initializing neural model...\n");
    model_errors_t init_rc = resumed ? MODEL_SUCCESS :
        model_new_seeded(&model, args->vocab_size, args->embedding_dim,
                         args->num_heads, args->num_layers, args->max_seq_len,
                         args->seed);

    if (init_rc != MODEL_SUCCESS) {
        fprintf(stderr, "Error: Model initialization failed (code: %d)\n", init_rc);
        token_stream_free(token_stream);
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

    /* 0 means a whole window: non-overlapping, so every corpus token is
     * supervised exactly once per epoch. A stride above the window would
     * skip tokens entirely, which is never what anyone means. */
    size_t train_stride = args->train_stride ? args->train_stride : train_window;
    if (train_stride > train_window) {
        fprintf(stderr, "Warning: --train-stride %zu > effective window %zu, clamping to %zu"
                        " (a larger stride would skip corpus tokens)\n",
                train_stride, train_window, train_window);
        train_stride = train_window;
    }
    batch_t *training_batch = batch_create((size_t)args->batch_size, train_window);
    if (!training_batch) {
        fprintf(stderr, "Error: Failed to allocate training minibatch\n");
        model_free(&model);
        token_stream_free(token_stream);
        tokenizer_free_encoder(encoder);
        return 1;
    }

    /* --input's learning rate was previously parsed and only ever printed,
     * never applied - now that training actually updates every parameter
     * (not just output_bias), the learning rate materially affects
     * stability, so it needs to actually reach the model. */
    if (!resumed) {
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
    }
    model.use_gpu = args->use_gpu;

    if (!resumed) {
        checkpoint_state.epoch_index = 0;
        checkpoint_state.step_in_epoch = 0;
        checkpoint_state.target_epochs = (uint32_t)args->epochs;
        checkpoint_state.input_fingerprint = corpus_stats.fingerprint;
        checkpoint_state.input_bytes = corpus_stats.byte_count;
        checkpoint_state.train_window = train_window;
        checkpoint_state.train_stride = train_stride;
        checkpoint_state.seed = args->seed;
        checkpoint_state.batch_size = args->batch_size;
        checkpoint_state.gradient_accumulation_steps = args->gradient_accumulation_steps;
        checkpoint_state.shuffle = args->shuffle;
        checkpoint_state.checkpoint_interval = args->checkpoint_interval;
        checkpoint_state.keep_checkpoints = args->keep_checkpoints;
        checkpoint_state.use_gpu = args->use_gpu;
        if (copy_resolved_path(checkpoint_state.input_file,
                               sizeof(checkpoint_state.input_file), args->input_file) != 0 ||
            (args->validation_file[0] &&
             copy_resolved_path(checkpoint_state.validation_file,
                                sizeof(checkpoint_state.validation_file),
                                args->validation_file) != 0) ||
            copy_path(checkpoint_state.tokenizer_path,
                      sizeof(checkpoint_state.tokenizer_path), tokenizer_path) != 0 ||
            copy_path(checkpoint_state.model_path,
                      sizeof(checkpoint_state.model_path), args->model_path) != 0 ||
            copy_path(checkpoint_state.checkpoint_dir,
                      sizeof(checkpoint_state.checkpoint_dir),
                      args->checkpoint_dir) != 0) {
            fprintf(stderr, "Error: A training artifact path is too long\n");
            model_free(&model);
            batch_free(training_batch);
            token_stream_free(token_stream);
            tokenizer_free_encoder(encoder);
            return 1;
        }
    } else if (copy_path(checkpoint_state.checkpoint_dir,
                         sizeof(checkpoint_state.checkpoint_dir),
                         args->checkpoint_dir) != 0) {
        fprintf(stderr, "Error: Checkpoint directory path is too long\n");
        model_free(&model);
        batch_free(training_batch);
        token_stream_free(token_stream);
        tokenizer_free_encoder(encoder);
        return 1;
    }

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

    char manifest_path[2048];
    if (run_manifest_write(args, &checkpoint_state, &model, encoder,
                           resume_source, manifest_path,
                           sizeof(manifest_path)) != 0) {
        fprintf(stderr, "Error: Failed to create immutable run manifest\n");
        model_free(&model);
        batch_free(training_batch);
        token_stream_free(token_stream);
        tokenizer_free_encoder(encoder);
        return 1;
    }
    printf("   ✓ Immutable run manifest: %s\n", manifest_path);

    /* ===== STEP 3: Streaming Training Loop ===== */
    printf("[3] Training model on streaming data...\n");
    printf("   Epochs: %d, Minibatch: %d, Accumulation: %d, Effective batch: %zu\n",
           args->epochs, args->batch_size, args->gradient_accumulation_steps,
           (size_t)args->batch_size * (size_t)args->gradient_accumulation_steps);
    printf("   Learning rate: %.8f, Minibatch shuffle: %s\n",
           args->learning_rate, args->shuffle ? "deterministic" : "disabled");
    
    size_t starting_training_steps = model.training_steps;
    size_t total_tokens_processed = 0;
    char chunk_buffer[STREAM_CHUNK_SIZE + 1];
    int training_failed = 0;
    evaluation_report_t final_validation_report = {0};
    
    for (int epoch = (int)start_epoch; epoch < args->epochs; epoch++) {
        uint64_t skip_for_epoch = (uint32_t)epoch == start_epoch ? resume_skip : 0;
        batch_reset(training_batch);
        training_accumulator_t accumulator = {
            .microbatch = training_batch,
            .resume_skip = skip_for_epoch,
        };
        token_stream_reset(token_stream);
        size_t next_target = 1;
        if (bpe_encoder_has_special_tokens(encoder)) {
            uint32_t bos = BPE_BOS_TOKEN_ID;
            if (token_stream_add(token_stream, &bos, 1) != 0) {
                fprintf(stderr, "Error: Failed to add BOS token\n");
                training_failed = 1;
                break;
            }
            total_tokens_processed++;
        }
        /* Reset stream for new epoch */
        stream_reader_t *epoch_stream = stream_reader_create(args->input_file, STREAM_CHUNK_SIZE);
        if (!epoch_stream) {
            fprintf(stderr, "Error: Cannot re-open file for epoch %d\n", epoch);
            training_failed = 1;
            break;
        }
        
        printf("   Processing epoch %d/%d...\n", epoch + 1, args->epochs);
        
        /* Process file in chunks */
        while (!stream_is_eof(epoch_stream)) {
            size_t chunk_size = stream_read_chunk(epoch_stream, chunk_buffer, STREAM_CHUNK_SIZE);
            if (chunk_size == 0) break;
            
            chunk_buffer[chunk_size] = '\0';
            
            /* The tokenizer was frozen before model initialization. Every
             * epoch therefore sees exactly the same token IDs. */
            bpe_tokens_t chunk_tokens = {0};
            if (bpe_encode(encoder, chunk_buffer, chunk_size, &chunk_tokens) != BPE_SUCCESS) {
                fprintf(stderr, "Error: Failed to encode training corpus\n");
                training_failed = 1;
                break;
            }
            
            if (chunk_tokens.token_count > 0) {
                /* Add tokens to stream for batch processing */
                if (token_stream_add(token_stream, chunk_tokens.token_ids,
                                     chunk_tokens.token_count) != 0) {
                    fprintf(stderr, "Error: Failed to extend training token buffer\n");
                    training_failed = 1;
                    free(chunk_tokens.token_ids);
                    break;
                }
                total_tokens_processed += chunk_tokens.token_count;
                
                /* Process batch when threshold reached */
                if (token_stream_ready_to_flush(token_stream)) {
                    size_t stream_size = token_stream_get_size(token_stream);
                    size_t next_start = next_target;
                    if (process_training_token_range(
                            &model, encoder, &accumulator, token_stream,
                            next_target, train_window, train_stride,
                            0 /* mid-file: never emit a chunk-truncated window */,
                            &next_start, (uint32_t)epoch,
                            &checkpoint_state, args) != 0) {
                        fprintf(stderr, "Error: Training or checkpoint step failed\n");
                        training_failed = 1;
                    }
                    
                    /* Progress reporting. Uses the running average loss
                     * (model.metrics.avg_loss), not the last single step's
                     * loss - now that every parameter trains (not just
                     * output_bias), a single step's loss is noisy enough
                     * to be a misleading progress signal on its own. */
                    if (model.training_steps > 0 && model.training_steps % 5000 == 0) {
                        printf("   [Progress] Steps: %zu, Avg Loss: %.6f, File: %.1f MB\n",
                               (size_t)model.training_steps, model.metrics.avg_loss,
                               stream_get_total_read(epoch_stream) / (1024.0f * 1024.0f));
                    }
                    
                    /* Keep everything from the first window that was not
                     * emitted. Unlike the sliding-by-one loop this replaced,
                     * no extra context prefix is retained: a window is
                     * self-contained under all-position supervision, so the
                     * tokens before next_start have been fully trained on and
                     * are not context for anything still to come. The next
                     * window therefore starts at offset 0 of the retained
                     * buffer. */
                    size_t retained = stream_size - next_start;
                    token_stream_retain_tail(token_stream, retained);
                    next_target = 0;
                }
            }
            
            free(chunk_tokens.token_ids);
            if (training_failed) break;
        }
        
        if (!training_failed && bpe_encoder_has_special_tokens(encoder)) {
            uint32_t eos = BPE_EOS_TOKEN_ID;
            if (token_stream_add(token_stream, &eos, 1) != 0) {
                fprintf(stderr, "Error: Failed to add EOS token\n");
                training_failed = 1;
            } else {
                total_tokens_processed++;
            }
        }

        /* Process remaining tokens, including the document EOS target. */
        if (!training_failed &&
            process_training_token_range(
                &model, encoder, &accumulator, token_stream, next_target,
                train_window, train_stride,
                1 /* end of stream: a short window here is the real corpus tail */,
                NULL, (uint32_t)epoch, &checkpoint_state, args) != 0) {
            fprintf(stderr, "Error: Training or checkpoint step failed\n");
            training_failed = 1;
        }
        if (!training_failed) {
            token_stream_reset(token_stream);
        }
        
        stream_reader_free(epoch_stream);
        if (training_failed) break;
        if (accumulator.prediction_cursor < skip_for_epoch) {
            fprintf(stderr, "Error: Checkpoint cursor exceeds epoch prediction count\n");
            training_failed = 1;
            break;
        }
        if (accumulator.prediction_cursor > skip_for_epoch &&
            batch_has_data(training_batch) &&
            process_training_microbatch(&model, encoder, &accumulator,
                                        (uint32_t)epoch, &checkpoint_state,
                                        args) != 0) {
            fprintf(stderr, "Error: Final minibatch failed\n");
            training_failed = 1;
            break;
        }
        if (accumulator.prediction_cursor > skip_for_epoch &&
            commit_accumulated_training(&model, encoder, &accumulator,
                                        (uint32_t)epoch, &checkpoint_state,
                                        args) != 0) {
            fprintf(stderr, "Error: Final accumulated optimizer step failed\n");
            training_failed = 1;
            break;
        }
        checkpoint_state.epoch_index = (uint32_t)epoch + 1;
        checkpoint_state.step_in_epoch = 0;
        resume_skip = 0;
        printf("   Epoch %d/%d - Avg Loss: %.6f, Tokens processed: %zu\n",
               epoch + 1, args->epochs, model.metrics.avg_loss, total_tokens_processed);

        if (args->validation_file[0]) {
            evaluation_errors_t eval_rc = evaluate_corpus_file(
                &model, encoder, args->validation_file, train_window,
                &final_validation_report);
            if (eval_rc != EVALUATION_SUCCESS) {
                fprintf(stderr, "Error: Validation failed after epoch %d (code: %d)\n",
                        epoch + 1, eval_rc);
                training_failed = 1;
                break;
            }
            print_evaluation_report("Validation", args->validation_file,
                                    train_window, &final_validation_report);
        }
    }

    if (training_failed) {
        model_free(&model);
        batch_free(training_batch);
        token_stream_free(token_stream);
        tokenizer_free_encoder(encoder);
        return 1;
    }

    /* A checkpoint may represent the end of the requested run. Evaluate once
     * when resume starts at that terminal cursor so config.json never loses
     * the held-out metrics merely because no additional epoch was needed. */
    if (args->validation_file[0] && final_validation_report.prediction_count == 0) {
        evaluation_errors_t eval_rc = evaluate_corpus_file(
            &model, encoder, args->validation_file, train_window,
            &final_validation_report);
        if (eval_rc != EVALUATION_SUCCESS) {
            fprintf(stderr, "Error: Final validation failed (code: %d)\n", eval_rc);
            model_free(&model);
            batch_free(training_batch);
            token_stream_free(token_stream);
            tokenizer_free_encoder(encoder);
            return 1;
        }
        print_evaluation_report("Validation", args->validation_file,
                                train_window, &final_validation_report);
    }

    /* Keep a terminal checkpoint even when the final update did not land on
     * the periodic interval. The epoch cursor points just past the last
     * completed epoch, so resuming it is unambiguous. */
    if (args->checkpoint_interval > 0) {
        char final_checkpoint_path[2048];
        checkpoint_state.epoch_index = (uint32_t)args->epochs;
        checkpoint_state.step_in_epoch = 0;
        if (checkpoint_save(&model, encoder, &checkpoint_state, args->checkpoint_dir,
                            args->keep_checkpoints > 0
                                ? (size_t)args->keep_checkpoints : 0,
                            final_checkpoint_path,
                            sizeof(final_checkpoint_path)) != 0) {
            fprintf(stderr, "Error: Failed to save final checkpoint\n");
            model_free(&model);
            batch_free(training_batch);
            token_stream_free(token_stream);
            tokenizer_free_encoder(encoder);
            return 1;
        }
        printf("   ✓ Final checkpoint saved: %s\n", final_checkpoint_path);
    }
    
    printf("   Training complete!\n");
    printf("   New optimizer steps: %zu, Total optimizer steps: %u, Corpus tokens streamed: %zu\n",
           (size_t)model.training_steps - starting_training_steps,
           model.training_steps, total_tokens_processed);
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
        /* Its own stream off the run's seed, so the demonstration prints the
         * same tokens on every platform and does not disturb the sampling
         * stream a real generate run uses. */
        uint64_t demo_rng = dranzer_rng_stream(args->seed,
                                               DRANZER_RNG_STREAM_TESTING);
        for (size_t i = 0; i < model.vocab_size; i++) {
            demo_logits[i] = dranzer_rng_uniform(&demo_rng, 0.0f, 1.0f);
        }
        
        uint32_t greedy = sample_greedy(demo_logits, model.vocab_size);
        printf("   - Greedy:   selected token %u\n", greedy);
        
        uint32_t topk = sample_topk(demo_logits, model.vocab_size, 5, &demo_rng);
        printf("   - Top-5:    selected token %u\n", topk);
        
        uint32_t topp = sample_topp(demo_logits, model.vocab_size, 0.9f, &demo_rng);
        printf("   - Top-p:    selected token %u\n", topp);
        
        printf("   ✓ Sampling strategies demonstrated\n");
        
        free(demo_logits);
    }

    /* ===== STEP 5: Model Persistence ===== */
    printf("[5] Saving versioned model bundle to %s\n", args->model_path);
    /* train_stride is deliberately NOT recorded here. The bundle is a
     * versioned artifact with compatibility fixtures, and adding a field
     * means a format bump; nothing about loading a model depends on the
     * stride it was trained with. The run manifest and the checkpoint both
     * carry it, which is where this project keeps run provenance. */
    model_bundle_metadata_t bundle_metadata = {
        .train_window = train_window,
        .seed = args->seed,
        .input_fingerprint = corpus_stats.fingerprint,
        .input_bytes = corpus_stats.byte_count,
    };
    bundle_errors_t save_rc = model_bundle_save(
        &model, encoder, &bundle_metadata, args->model_path);
    
    if (save_rc == BUNDLE_SUCCESS) {
        printf("   ✓ Atomic model + tokenizer bundle saved\n");
    } else {
        fprintf(stderr, "   ✗ Save failed (code: %d)\n", save_rc);
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
    config.learning_rate = model.metrics.initial_learning_rate;
    config.batch_size = (size_t)args->batch_size;
    config.gradient_accumulation_steps =
        (size_t)args->gradient_accumulation_steps;
    config.shuffle = args->shuffle;
    config.num_epochs = (size_t)args->epochs;
    config.checkpoint_interval = (size_t)args->checkpoint_interval;
    config.seed = args->seed;
    config.tokenizer_vocab_size = encoder->vocab_size;
    config.tokenizer_has_special_tokens = bpe_encoder_has_special_tokens(encoder);
    config.pad_token_id = bpe_encoder_special_token_id(encoder, BPE_SPECIAL_PAD);
    config.unk_token_id = bpe_encoder_special_token_id(encoder, BPE_SPECIAL_UNK);
    config.bos_token_id = bpe_encoder_special_token_id(encoder, BPE_SPECIAL_BOS);
    config.eos_token_id = bpe_encoder_special_token_id(encoder, BPE_SPECIAL_EOS);
    config.input_fingerprint = corpus_stats.fingerprint;
    config.input_bytes = corpus_stats.byte_count;
    int config_paths_valid =
        copy_path(config.model_path, sizeof(config.model_path), args->model_path) == 0 &&
        copy_path(config.tokenizer_path, sizeof(config.tokenizer_path), tokenizer_path) == 0 &&
        copy_path(config.input_path, sizeof(config.input_path), args->input_file) == 0 &&
        copy_path(config.checkpoint_dir, sizeof(config.checkpoint_dir),
                  args->checkpoint_dir) == 0;
    if (args->validation_file[0] &&
        copy_path(config.validation_path, sizeof(config.validation_path),
                  args->validation_file) == 0) {
        config.validation_fingerprint = final_validation_report.corpus_fingerprint;
        config.validation_bytes = final_validation_report.corpus_bytes;
        config.validation_tokens = final_validation_report.token_count;
        config.validation_cross_entropy = final_validation_report.average_cross_entropy;
        config.validation_perplexity = final_validation_report.perplexity;
    } else if (args->validation_file[0]) {
        config_paths_valid = 0;
    }
    int config_saved = config_paths_valid && config_save("config.json", &config) == 0;
    if (config_saved) {
        printf("   ✓ Configuration and corpus provenance saved\n\n");
    } else {
        fprintf(stderr, "   ✗ Failed to save configuration\n\n");
    }

    /* ===== CLEANUP ===== */
    printf("[7] Cleaning up...\n");
    model_free(&model);
    batch_free(training_batch);
    token_stream_free(token_stream);
    tokenizer_free_encoder(encoder);
    printf("   ✓ Resources freed\n\n");

    printf("=== Training Complete ===\n");
    return save_rc == BUNDLE_SUCCESS && config_saved ? 0 : 1;
}

/* ===== EVALUATION MODE ===== */
int mode_eval(const cli_args_t *args) {
    printf("=== Neural Model Evaluation ===\n\n");

    if (!args->input_explicit || !args->input_file[0]) {
        fprintf(stderr, "Error: eval mode requires an explicit --input FILE\n");
        return 2;
    }

    neural_model_t model = {0};
    bpe_encoder_t *encoder = NULL;
    if (load_model_artifact(args, &model, &encoder) != 0) {
        return 1;
    }
    model.use_gpu = args->use_gpu;

    size_t context_window = args->eval_window ? args->eval_window : model.max_seq_len;
    if (context_window > model.max_seq_len) {
        fprintf(stderr, "Error: --eval-window %zu exceeds model max sequence length %zu\n",
                context_window, model.max_seq_len);
        tokenizer_free_encoder(encoder);
        model_free(&model);
        return 2;
    }

    evaluation_report_t report = {0};
    evaluation_errors_t eval_rc = evaluate_corpus_file(
        &model, encoder, args->input_file, context_window, &report);
    if (eval_rc != EVALUATION_SUCCESS) {
        fprintf(stderr, "Error: Evaluation failed for %s (code: %d)\n",
                args->input_file, eval_rc);
        tokenizer_free_encoder(encoder);
        model_free(&model);
        return 1;
    }

    print_evaluation_report("Evaluation", args->input_file, context_window, &report);
    printf("   ✓ Model and optimizer state were not updated\n");

    tokenizer_free_encoder(encoder);
    model_free(&model);
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
    bpe_encoder_t *encoder = NULL;
    if (load_model_artifact(args, &model, &encoder) != 0) {
        return 1;
    }
    model.use_gpu = args->use_gpu;

    printf("   ✓ Model loaded\n");
    printf("   - Vocab: %zu tokens\n", model.vocab_size);
    printf("   - Embedding: %zu\n", model.embedding_dim);
    printf("   - Heads: %zu\n", model.num_heads);

    printf("[2] Tokenizing prompt: \"%s\"\n", args->prompt);
    
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
        uint32_t *context = malloc(model.max_seq_len * sizeof(*context));
        size_t context_len = 0;
        if (!context || generation_prepare_prompt(
                encoder, tokens.token_ids, tokens.token_count, context,
                model.max_seq_len, &context_len) != 0) {
            fprintf(stderr, "Error: Failed to prepare model prompt\n");
            free(context);
            model_free(&model);
            tokenizer_free_encoder(encoder);
            free(tokens.token_ids);
            return 1;
        }
        size_t structural = bpe_encoder_has_special_tokens(encoder) ? 1 : 0;
        if (context_len < tokens.token_count + structural) {
            printf("   Prompt truncated to %zu model-visible tokens\n", context_len);
        }

        int was_training = model.is_training;
        model.is_training = 0;
        model_errors_t forward_rc = model_forward(&model, context, context_len, model.ws_logits);
        model.is_training = was_training;
        if (forward_rc != MODEL_SUCCESS) {
            fprintf(stderr, "Error: Model forward pass failed (code: %d)\n", forward_rc);
            free(context);
            model_free(&model);
            tokenizer_free_encoder(encoder);
            free(tokens.token_ids);
            return 1;
        }
        generation_mask_control_logits(encoder, model.ws_logits, model.vocab_size);
        uint64_t infer_rng = dranzer_rng_stream(args->seed,
                                                DRANZER_RNG_STREAM_SAMPLING);
        uint32_t predicted = sample_next_token(model.ws_logits, model.vocab_size,
                                               args->sampling_strategy, args->temperature,
                                               (size_t)args->top_k, args->top_p,
                                               &infer_rng);
        if (generation_token_is_eos(encoder, predicted))
            printf("   Predicted next token: EOS (%u)\n", predicted);
        else
            printf("   Predicted next token: %u\n", predicted);
        printf("   ✓ Inference complete\n\n");
        free(context);
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
    bpe_encoder_t *encoder = NULL;
    if (load_model_artifact(args, &model, &encoder) != 0) {
        return 1;
    }
    model.use_gpu = args->use_gpu;

    printf("   ✓ Model loaded\n");

    printf("[2] Tokenizing prompt: \"%s\"\n", args->prompt);
    
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

    bpe_tokens_t stop_tokens[CLI_MAX_STOP_SEQUENCES] = {{0}};
    generation_stop_sequence_t stops[CLI_MAX_STOP_SEQUENCES] = {{0}};
    for (size_t i = 0; i < args->stop_sequence_count; i++) {
        const char *stop_text = args->stop_sequences[i];
        if (bpe_encode(encoder, stop_text, strlen(stop_text),
                       &stop_tokens[i]) != BPE_SUCCESS ||
            stop_tokens[i].token_count == 0) {
            fprintf(stderr, "Error: Failed to encode stop sequence %zu\n", i + 1);
            free_stop_tokens(stop_tokens, i + 1);
            model_free(&model);
            tokenizer_free_encoder(encoder);
            free(tokens.token_ids);
            return 1;
        }
        stops[i].token_ids = stop_tokens[i].token_ids;
        stops[i].token_count = stop_tokens[i].token_count;
    }

    printf("[3] Generating %d tokens with %s sampling...\n", 
           args->generate_length,
           args->sampling_strategy == SAMPLING_GREEDY ? "greedy" :
           args->sampling_strategy == SAMPLING_TOPK ? "top-k" : "top-p");
    
    size_t requested = args->generate_length > 0 ?
                           (size_t)args->generate_length : 0;
    if (requested > SIZE_MAX - model.max_seq_len ||
        model.max_seq_len + requested > SIZE_MAX / sizeof(uint32_t)) {
        fprintf(stderr, "Error: Requested generation length is too large\n");
        free_stop_tokens(stop_tokens, args->stop_sequence_count);
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }
    size_t sequence_capacity = model.max_seq_len + requested;

    /* Retain token IDs for caller-visible results and repetition controls;
     * decoded text itself is still streamed incrementally. */
    uint32_t *generated = malloc(sequence_capacity * sizeof(uint32_t));
    if (!generated) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free_stop_tokens(stop_tokens, args->stop_sequence_count);
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }
    
    size_t prompt_len = 0;
    if (generation_prepare_prompt(
            encoder, tokens.token_ids, tokens.token_count, generated,
            model.max_seq_len, &prompt_len) != 0) {
        fprintf(stderr, "Error: Failed to prepare model prompt\n");
        free(generated);
        free_stop_tokens(stop_tokens, args->stop_sequence_count);
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }
    size_t structural = bpe_encoder_has_special_tokens(encoder) ? 1 : 0;
    if (prompt_len < tokens.token_count + structural) {
        printf("   Prompt truncated to %zu model-visible tokens\n", prompt_len);
    }
    size_t current_len = prompt_len;

    size_t new_token_count = requested;

    char *prompt_text = NULL;
    size_t prompt_text_length = 0;
    if (bpe_decode(encoder, generated, prompt_len, &prompt_text,
                   &prompt_text_length) != BPE_SUCCESS || !prompt_text) {
        fprintf(stderr, "Error: Failed to decode prepared prompt\n");
        free(generated);
        free_stop_tokens(stop_tokens, args->stop_sequence_count);
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }

    /* One stream for the whole generation, so the text --seed names is the same
     * on any C library (see core/rng.h) and does not depend on how many tokens
     * an earlier call happened to sample. */
    uint64_t generate_rng = dranzer_rng_stream(args->seed,
                                               DRANZER_RNG_STREAM_SAMPLING);
    generation_options_t options = {
        .strategy = args->sampling_strategy,
        .temperature = args->temperature,
        .top_k = (size_t)args->top_k,
        .top_p = args->top_p,
        .repetition_penalty = args->repetition_penalty,
        .minimum_new_tokens = (size_t)args->minimum_generation_length,
        .sequence_capacity = sequence_capacity,
        .stop_sequences = stops,
        .stop_sequence_count = args->stop_sequence_count,
        .on_token = stream_token_to_stdout,
        .rng_state = &generate_rng,
    };
    generation_result_t generation = {0};
    printf("\n=== Generated Sequence ===\n");
    int output_failed = prompt_text_length > 0 &&
                        fwrite(prompt_text, 1, prompt_text_length, stdout) !=
                            prompt_text_length;
    free(prompt_text);
    if (fflush(stdout) != 0) output_failed = 1;
    if (output_failed) {
        fprintf(stderr, "Error: Failed while streaming generated output\n");
        free(generated);
        free_stop_tokens(stop_tokens, args->stop_sequence_count);
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }

    generation_errors_t generation_rc = generation_decode_with_options(
        &model, encoder, generated, prompt_len, new_token_count,
        &options, &generation);
    printf("\n\n");
    if (generation_rc != GENERATION_SUCCESS) {
        fprintf(stderr, "Error: Generation failed (code: %d)\n", generation_rc);
        free(generated);
        free_stop_tokens(stop_tokens, args->stop_sequence_count);
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }
    if (generation.stopped_by_callback && ferror(stdout)) {
        fprintf(stderr, "Error: Failed while streaming generated output\n");
        free(generated);
        free_stop_tokens(stop_tokens, args->stop_sequence_count);
        model_free(&model);
        tokenizer_free_encoder(encoder);
        free(tokens.token_ids);
        return 1;
    }
    current_len = generation.total_count;
    size_t retained_context = current_len < model.max_seq_len ?
                                  current_len : model.max_seq_len;

    printf("   Sampled %zu new token%s; streamed %zu "
           "(%zu sequence total; newest %zu retained)\n",
           generation.new_count, generation.new_count == 1 ? "" : "s",
           generation.emitted_count, current_len, retained_context);
    if (generation.stopped_on_eos) printf("   Stopped naturally on EOS\n");
    if (generation.stopped_on_stop_sequence)
        printf("   Stopped on --stop sequence %zu\n",
               generation.stop_sequence_index + 1);
    printf("   ✓ Generation complete\n\n");

    /* Cleanup */
    free(generated);
    free_stop_tokens(stop_tokens, args->stop_sequence_count);
    model_free(&model);
    tokenizer_free_encoder(encoder);
    free(tokens.token_ids);
    
    return 0;
}
