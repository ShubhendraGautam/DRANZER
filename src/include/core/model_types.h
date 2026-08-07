#ifndef MODEL_TYPES_H
#define MODEL_TYPES_H

#include <stddef.h>
#include <stdint.h>

/* Shared data types for the model_* modules (tensor_ops, transformer,
 * optimizer, training, serialization, metrics, model). Kept in its own
 * header so every module can include just the type definitions without
 * pulling in another module's function declarations. */

typedef enum {
    MODEL_SUCCESS = 0,
    MODEL_INVALID_INPUT,
    MODEL_ALLOCATION_FAILURE,
    MODEL_IO_ERROR,
} model_errors_t;

typedef enum {
    OPTIMIZER_SGD = 0,
    OPTIMIZER_ADAM,
} optimizer_type_t;

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
    uint32_t steps_without_improvement; // For early stopping / plateau decay
} learning_metrics_t;

/* One transformer block: self-attention + FFN, each with its own residual
 * connection and layer norm. neural_model_t holds an array of num_layers
 * of these, stacked so each layer's output feeds the next layer's input.
 *
 * Every float* below is a VIEW into neural_model_t's flat params/grads
 * buffers (set up once in model_new), not an independently owned
 * allocation - model_free frees the flat buffers, not these pointers. This
 * is what lets the optimizer/serialization code operate generically over
 * "all trainable parameters" without a per-field update call for every one
 * of these arrays, in every layer, every time a new one is added. */
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

    // --- Optimizer configuration. Defaults are set by model_new; the
    // caller may override any of these fields directly before training
    // starts, the same way it already overrides learning_rate. ---
    optimizer_type_t optimizer_type;
    float adam_beta1, adam_beta2, adam_eps;
    float weight_decay;      // decoupled weight decay (AdamW-style); 0 disables it
    float grad_clip_norm;    // global L2-norm gradient clipping threshold; 0 disables it
    uint32_t adam_t;         // Adam bias-correction timestep, advanced once per step

    // --- LR schedule (linear warmup -> cosine decay). total_steps == 0
    // disables this schedule entirely (falls back to plateau decay driven
    // by learning_metrics_t.steps_without_improvement). ---
    uint32_t warmup_steps;
    uint32_t total_steps;
    float base_lr, min_lr;

    // --- Dropout (applied to each sub-layer's output before it is added
    // to the residual branch, as in the original Transformer). Disabled
    // whenever dropout_rate == 0 or is_training == 0 (e.g. at inference). ---
    float dropout_rate;
    int is_training;
    uint64_t rng_state;      // model-owned dropout RNG, persisted by training checkpoints

    // --- GPU offload (forward pass only - see transformer.c). Off by
    // default; opt in with --gpu (main.c) or by setting this directly.
    // Has no effect at all if no CUDA GPU is usable (gpu_matmul_available()
    // is checked at every dispatch site, not just once), so this is safe
    // to leave on unconditionally on a machine that might or might not
    // have an NVIDIA GPU. ---
    int use_gpu;
    int use_scalar_matmul; /* force the portable reference CPU path */

    // --- Flat parameter storage. Every trainable weight (token
    // embeddings, output head, every layer's weights) is a contiguous
    // slice of these two buffers; the named pointers throughout this
    // struct and transformer_layer_t are just views into them. This is
    // what makes model_zero_gradients, the optimizer update, and weight
    // serialization each a single loop/fwrite over total_param_count
    // instead of one line per named field. ---
    float *params;
    float *grads;
    float *adam_m, *adam_v;   // Adam moment buffers, same layout as params/grads - lazily allocated (NULL until the first Adam step; see optimizer.c), so plain-SGD deployments never pay for them
    size_t total_param_count;

    // Embeddings
    float *token_embeddings, *token_embeddings_grad;   // views into params/grads
    float *position_embeddings;   // max_seq_len x embedding_dim (fixed sinusoidal, separately owned, not trained)

    // Stacked transformer blocks
    transformer_layer_t *layers;  // num_layers entries

    // Output head (next token prediction, reads only the last sequence position)
    float *output_projection, *output_bias;             // views into params
    float *output_projection_grad, *output_bias_grad;    // views into grads

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
    float **cache_attn_dropout_mask;  // [num_layers], each max_seq_len*embedding_dim (1.0 keep / 0.0 drop)
    float **cache_ffn_dropout_mask;   // [num_layers], each max_seq_len*embedding_dim (1.0 keep / 0.0 drop)

    // --- Reusable scratch for forward and backward passes. Single instance
    // each, overwritten every layer iteration - never needs to persist
    // across layers, unlike the cache_* arrays above. Sized for max_seq_len.
    float *ws_fwd_attn_raw, *ws_fwd_ff_raw;              // forward: pre-residual sums, max_seq_len*embedding_dim
    float *ws_dhidden_in, *ws_dhidden_out;               // backward: dL/dhidden ping-pong, max_seq_len*embedding_dim
    float *ws_d_s2, *ws_d_x1_total, *ws_d_s1;            // backward: intermediate gradients, max_seq_len*embedding_dim
    float *ws_d_ff_hidden;                               // backward: dL/dff_hidden, max_seq_len*ffn_dim
    float *ws_d_attn_concat;                             // backward: dL/dattn_concat (= dL/dcontext), max_seq_len*embedding_dim
    float *ws_d_scores;                                  // backward: per-head dL/dprobs -> dL/dscores, num_heads*max_seq_len*max_seq_len (one disjoint slab per head, so the head loop can run in parallel under OMP=1)
    float *ws_d_Q, *ws_d_K, *ws_d_V;                     // backward: dL/dQ,dK,dV accumulators, max_seq_len*embedding_dim
    float *ws_d_attn_dropout;                            // backward: dropout-backward'd copy of ws_d_s1 for the attention-path branch, max_seq_len*embedding_dim
    float *ws_d_ffn_dropout;                             // backward: dropout-backward'd copy of ws_d_s2 for the FFN-path branch, max_seq_len*embedding_dim

    // Logits scratch, shared by training and inference.
    float *ws_logits, *ws_grad_logits;                   // vocab_size each

    // All-position logits and their gradient, for the training head in
    // core/lm_head.c: max_seq_len*vocab_size each, row-major by position.
    // Separate from ws_logits above rather than an enlargement of it,
    // because inference and generation pass ws_logits as a plain
    // vocab_size buffer and must keep reading row 0.
    //
    // This is the largest workspace in the model at wide vocabularies -
    // max_seq_len*vocab_size*4 bytes twice over - and unlike the cache_*
    // arrays it is dead weight for an inference-only process. Sized eagerly
    // anyway, on the same "allocate once in model_new, never per step"
    // policy as everything else here; a lazy allocation is the obvious
    // change if that ever costs more than it saves.
    float *ws_logits_all, *ws_grad_logits_all;

} neural_model_t;

#endif // MODEL_TYPES_H
