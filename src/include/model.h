#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>

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

typedef struct {
    // Hyperparameters
    size_t vocab_size;
    size_t embedding_dim;
    size_t num_heads;
    size_t num_layers;
    float learning_rate;
    
    // Embeddings
    float *token_embeddings;    // vocab_size x embedding_dim
    float *position_embeddings; // max_seq_len x embedding_dim
    
    // Attention parameters (per head)
    float *W_q;  // embedding_dim x embedding_dim
    float *W_k;  // embedding_dim x embedding_dim
    float *W_v;  // embedding_dim x embedding_dim
    float *W_o;  // embedding_dim x embedding_dim (output projection)
    
    // Layer normalization parameters (Phase 2)
    float *ln_gamma_attn;      // embedding_dim (scale for attention output norm)
    float *ln_beta_attn;       // embedding_dim (shift for attention output norm)
    float *ln_gamma_ffn;       // embedding_dim (scale for FFN output norm)
    float *ln_beta_ffn;        // embedding_dim (shift for FFN output norm)
    
    // Feedforward parameters
    float *W_ff1;  // embedding_dim x (embedding_dim * 4)
    float *b_ff1;  // embedding_dim * 4
    float *W_ff2;  // (embedding_dim * 4) x embedding_dim
    float *b_ff2;  // embedding_dim
    
    // Output head (next token prediction)
    float *output_projection; // embedding_dim x vocab_size
    float *output_bias;       // vocab_size
    
    // Gradients for training
    float *token_embeddings_grad;
    float *W_q_grad;
    float *W_k_grad;
    float *W_v_grad;
    float *W_o_grad;
    float *ln_gamma_attn_grad;
    float *ln_beta_attn_grad;
    float *ln_gamma_ffn_grad;
    float *ln_beta_ffn_grad;
    float *W_ff1_grad;
    float *b_ff1_grad;
    float *W_ff2_grad;
    float *b_ff2_grad;
    float *output_projection_grad;
    float *output_bias_grad;
    
    // Training state
    uint32_t training_steps;
    float current_loss;
    
    // Phase 2: Learning metrics
    learning_metrics_t metrics;
    
    // Optimization: Workspace memory pooling for reuse (avoid repeated malloc/free)
    float *workspace;           // Single backing allocation for all forward/train temporaries
    size_t workspace_size;      // Total allocated workspace size (floats)
    size_t max_seq_len;         // Max sequence length for workspace sizing

    // Named sub-regions of `workspace`, computed once in model_new().
    // Reused on every forward/train/predict call instead of malloc+free,
    // which matters on low-end hardware where heap allocation is slower
    // and more prone to fragmentation under sustained training loops.
    // Do NOT free these individually - they are freed together with
    // `workspace`. Because they're shared, model_forward/model_train_step/
    // model_predict_next_token are not reentrant or thread-safe.
    float *ws_Q, *ws_K, *ws_V, *ws_scores, *ws_temp;   // attention temporaries
    float *ws_embeddings, *ws_attn_output, *ws_ff_hidden, *ws_ff_output; // forward pass temporaries
    float *ws_logits, *ws_grad_logits;                 // training/inference temporaries

} neural_model_t;

/**
 * Initialize a new neural model with random weights
 */
model_errors_t model_new(neural_model_t *model,
                         size_t vocab_size,
                         size_t embedding_dim,
                         size_t num_heads,
                         size_t max_seq_len);

/**
 * Forward pass through the model
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
 * Train the model on a sequence (simplified single-sample training)
 * @param model: The neural model
 * @param token_ids: Input token IDs
 * @param target_id: Target next token ID
 * @param seq_len: Length of sequence
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
