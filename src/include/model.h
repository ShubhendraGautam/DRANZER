#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    MODEL_SUCCESS = 0,
    MODEL_INVALID_INPUT,
    MODEL_ALLOCATION_FAILURE,
    MODEL_IO_ERROR,
} model_errors_t;

/* Learning metrics tracking for Phase 2 stability improvements */
typedef struct {
    float *loss_history;      // Array of losses per training step
    size_t history_size;      // Current history length
    size_t history_capacity;  // Max history capacity

    float best_loss;          // Minimum loss achieved
    float worst_loss;         // Maximum loss achieved
    float avg_loss;           // Running average loss

    float learning_rate;      // Current learning rate (for scheduling)
    float initial_learning_rate;
    uint32_t steps_without_improvement; // For early stopping
} learning_metrics_t;

/* One transformer block: self-attention + FFN, each with its own residual
 * connection and layer norm. neural_model_t holds an array of num_layers
 * of these, stacked so each layer's output feeds the next layer's input. */
typedef struct {
    // Attention projections (embedding_dim x embedding_dim)
    float *W_q, *W_k, *W_v, *W_o;
    float *W_q_grad, *W_k_grad, *W_v_grad, *W_o_grad;

    // Post-attention layer norm (embedding_dim)
    float *ln_gamma_attn, *ln_beta_attn;
    float *ln_gamma_attn_grad, *ln_beta_attn_grad;

    // Feedforward (embedding_dim x ffn_dim, ffn_dim x embedding_dim; ffn_dim = embedding_dim*4)
    float *W_ff1, *b_ff1, *W_ff2, *b_ff2;
    float *W_ff1_grad, *b_ff1_grad, *W_ff2_grad, *b_ff2_grad;

    // Post-FFN layer norm (embedding_dim)
    float *ln_gamma_ffn, *ln_beta_ffn;
    float *ln_gamma_ffn_grad, *ln_beta_ffn_grad;
} transformer_layer_t;

typedef struct {
    // Hyperparameters
    size_t vocab_size;
    size_t embedding_dim;
    size_t num_heads;
    size_t num_layers;
    float learning_rate;

    // Embeddings
    float *token_embeddings;      // vocab_size x embedding_dim (trained)
    float *token_embeddings_grad;
    float *position_embeddings;   // max_seq_len x embedding_dim (fixed sinusoidal, not trained)

    // Stacked transformer blocks
    transformer_layer_t *layers;  // num_layers entries

    // Output head (next token prediction, reads only the last sequence position)
    float *output_projection; // embedding_dim x vocab_size
    float *output_bias;       // vocab_size
    float *output_projection_grad;
    float *output_bias_grad;

    // Training state
    uint32_t training_steps;
    float current_loss;

    // Phase 2: Learning metrics
    learning_metrics_t metrics;

    // Max sequence length this model's workspace/caches are sized for.
    size_t max_seq_len;

    // --- Forward-pass activation cache, used by model_train_step to make
    // backpropagation possible without any additional heap allocation.
    // Each array has num_layers entries (cache_hidden has num_layers+1:
    // cache_hidden[l] is the input to layer l; cache_hidden[num_layers] is
    // the final output fed to the output projection). Every entry is
    // individually malloc'd once in model_new (not per training step) and
    // sized for model->max_seq_len, the worst case. model_forward (used
    // standalone by inference) also writes through these, but only
    // model_train_step reads them back for backward - so this struct is
    // not thread-safe/reentrant, consistent with the existing workspace.
    float **cache_hidden;       // [num_layers+1], each max_seq_len*embedding_dim
    float **cache_Q, **cache_K, **cache_V;   // [num_layers], each max_seq_len*embedding_dim
    float **cache_probs;        // [num_layers], each num_heads*max_seq_len*max_seq_len (post-softmax)
    float **cache_attn_concat;  // [num_layers], each max_seq_len*embedding_dim (pre-W_o, heads concatenated)
    float **cache_attn_ln_out;  // [num_layers], each max_seq_len*embedding_dim (post attn-residual-LN, i.e. FFN input)
    float **cache_attn_xhat;    // [num_layers], each max_seq_len*embedding_dim (attn-LN normalized values)
    float **cache_attn_std;     // [num_layers], each max_seq_len (attn-LN std-dev per position)
    float **cache_ff_hidden;    // [num_layers], each max_seq_len*ffn_dim (post-ReLU)
    float **cache_ffn_xhat;     // [num_layers], each max_seq_len*embedding_dim (FFN-LN normalized values)
    float **cache_ffn_std;      // [num_layers], each max_seq_len (FFN-LN std-dev per position)

    // --- Reusable scratch for forward and backward passes. Single instance
    // each, overwritten every layer iteration - never needs to persist
    // across layers, unlike the cache_* arrays above. Sized for max_seq_len.
    float *ws_fwd_attn_raw, *ws_fwd_ff_raw;              // forward: pre-residual sums, max_seq_len*embedding_dim
    float *ws_dhidden_in, *ws_dhidden_out;               // backward: dL/dhidden ping-pong, max_seq_len*embedding_dim
    float *ws_d_s2, *ws_d_x1_total, *ws_d_s1;            // backward: intermediate gradients, max_seq_len*embedding_dim
    float *ws_d_ff_hidden;                               // backward: dL/dff_hidden, max_seq_len*ffn_dim
    float *ws_d_attn_concat;                             // backward: dL/dattn_concat (= dL/dcontext), max_seq_len*embedding_dim
    float *ws_d_scores;                                  // backward: per-head dL/dprobs -> dL/dscores, max_seq_len*max_seq_len
    float *ws_d_Q, *ws_d_K, *ws_d_V;                     // backward: dL/dQ,dK,dV accumulators, max_seq_len*embedding_dim

    // Logits scratch, shared by training and inference.
    float *ws_logits, *ws_grad_logits;                   // vocab_size each

} neural_model_t;

/**
 * Initialize a new neural model with random weights
 */
model_errors_t model_new(neural_model_t *model,
                         size_t vocab_size,
                         size_t embedding_dim,
                         size_t num_heads,
                         size_t num_layers,
                         size_t max_seq_len);

/**
 * Forward pass through the model. Populates the activation cache as a side
 * effect (needed by model_train_step's backward pass) but this is harmless
 * for standalone inference use (infer/generate) - just some extra writes
 * into memory nothing else reads afterward.
 * @param model: The neural model
 * @param token_ids: Input token IDs
 * @param seq_len: Length of sequence
 * @param output_logits: Output logits for next token prediction (vocab_size)
 * @return MODEL_SUCCESS on success
 */
model_errors_t model_forward(neural_model_t *model,
                             uint32_t *token_ids,
                             size_t seq_len,
                             float *output_logits);

/**
 * Train the model on a sequence: forward pass, cross-entropy loss, full
 * backpropagation through every layer (attention, FFN, layer norms) and the
 * token embeddings, then a plain SGD update of every parameter that
 * received a gradient. position_embeddings are fixed and never trained.
 * @param model: The neural model
 * @param token_ids: Input token IDs (context window)
 * @param target_id: Target next token ID
 * @param seq_len: Length of the context window (must be >= 1 and <= model->max_seq_len)
 * @return MODEL_SUCCESS on success
 */
model_errors_t model_train_step(neural_model_t *model,
                                uint32_t *token_ids,
                                uint32_t target_id,
                                size_t seq_len);

/**
 * Get predicted next token
 * @param model: The neural model
 * @param token_ids: Input token IDs
 * @param seq_len: Length of sequence
 * @return Predicted token ID
 */
uint32_t model_predict_next_token(neural_model_t *model,
                                  uint32_t *token_ids,
                                  size_t seq_len);

/**
 * Save model weights to file
 */
model_errors_t model_save(neural_model_t *model, const char *filename);

/**
 * Load model weights from file
 */
model_errors_t model_load(neural_model_t *model, const char *filename);

/**
 * Write model dimensions + all weights (not gradients, not training
 * metrics) to an already-open file stream. Shared by model_save and
 * checkpoint_save so the on-disk weight format has one source of truth.
 */
model_errors_t model_write_state(const neural_model_t *model, FILE *f);

/**
 * Read model dimensions + all weights from an already-open file stream,
 * reinitializing *model (via model_new) if its current dimensions don't
 * match what's in the stream. Shared by model_load and checkpoint_load.
 */
model_errors_t model_read_state(neural_model_t *model, FILE *f);

/**
 * Free all model resources
 */
void model_free(neural_model_t *model);

/**
 * Phase 2: Layer normalization function
 * Normalizes activations and applies learned scale/shift parameters
 */
void layer_normalize(float *input, float *output, size_t size,
                     float *gamma, float *beta, float epsilon);

/**
 * Phase 2: Update learning rate based on metrics (learning rate scheduling)
 */
void update_learning_rate(neural_model_t *model);

/**
 * Phase 2: Get learning metrics (loss history, stats)
 */
void model_get_metrics(neural_model_t *model, learning_metrics_t *out_metrics);

/**
 * Phase 2: Print training metrics and statistics
 */
void model_print_metrics(neural_model_t *model);

#endif // MODEL_H
