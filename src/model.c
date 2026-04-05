/*
 * Neural model implementation with multi-head attention and training
 * Phase 1: Multi-head attention, weight training, next-token prediction, model persistence
 */

#include "include/model.h"
#include "include/debug.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define MAX_SEQ_LEN 512

/* Xavier initialization for weight matrices */
static void xavier_init(float *weights, size_t size, size_t fan_in, size_t fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    for (size_t i = 0; i < size; i++) {
        weights[i] = (rand() / (float)RAND_MAX) * 2.0f * limit - limit;
    }
}

/* Softmax implementation */
static void softmax(float *values, size_t size) {
    if (size == 0) return;
    
    float max_val = values[0];
    for (size_t i = 1; i < size; i++) {
        if (values[i] > max_val) max_val = values[i];
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }
    
    if (sum > 0) {
        for (size_t i = 0; i < size; i++) {
            values[i] /= sum;
        }
    }
}

/* ReLU activation */
static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* ReLU derivative */
static inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

/* Matrix multiplication: C = A * B (A: m x k, B: k x n, C: m x n) */
static void matrix_multiply(float *A, float *B, float *C, 
                           size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* Phase 2: Layer normalization - normalize + scale + shift */
static void layer_norm_internal(float *input, float *gamma, float *beta, 
                                size_t size, float epsilon) {
    /* Compute mean */
    float mean = 0.0f;
    for (size_t i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    
    /* Compute variance */
    float variance = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    
    /* Normalize and apply scale/shift */
    float std_dev = sqrtf(variance + epsilon);
    for (size_t i = 0; i < size; i++) {
        input[i] = gamma[i] * ((input[i] - mean) / std_dev) + beta[i];
    }
}

/* Positional encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d)) */
static void compute_positional_encoding(float *pos_embed, size_t seq_len, size_t embedding_dim) {
    for (size_t pos = 0; pos < seq_len; pos++) {
        for (size_t i = 0; i < embedding_dim; i++) {
            float angle = pos / powf(10000.0f, (2.0f * i) / embedding_dim);
            if (i % 2 == 0) {
                pos_embed[pos * embedding_dim + i] = sinf(angle);
            } else {
                pos_embed[pos * embedding_dim + i] = cosf(angle);
            }
        }
    }
}

/* Initialize model with random weights */
model_errors_t model_new(neural_model_t *model,
                         size_t vocab_size,
                         size_t embedding_dim,
                         size_t num_heads,
                         size_t max_seq_len) {
    if (model == NULL || embedding_dim % num_heads != 0) {
        return MODEL_INVALID_INPUT;
    }
    
    DEBUG_PRINT("Initializing neural model: vocab_size=%zu, embedding_dim=%zu, num_heads=%zu\n",
                vocab_size, embedding_dim, num_heads);
    
    model->vocab_size = vocab_size;
    model->embedding_dim = embedding_dim;
    model->num_heads = num_heads;
    model->num_layers = 1;
    model->learning_rate = 0.001f;
    model->training_steps = 0;
    model->current_loss = 0.0f;
    
    size_t ffn_dim = embedding_dim * 4;
    
    /* Allocate weight matrices */
    model->token_embeddings = malloc(vocab_size * embedding_dim * sizeof(float));
    model->position_embeddings = malloc(max_seq_len * embedding_dim * sizeof(float));
    model->W_q = malloc(embedding_dim * embedding_dim * sizeof(float));
    model->W_k = malloc(embedding_dim * embedding_dim * sizeof(float));
    model->W_v = malloc(embedding_dim * embedding_dim * sizeof(float));
    model->W_o = malloc(embedding_dim * embedding_dim * sizeof(float));
    model->W_ff1 = malloc(embedding_dim * ffn_dim * sizeof(float));
    model->b_ff1 = malloc(ffn_dim * sizeof(float));
    model->W_ff2 = malloc(ffn_dim * embedding_dim * sizeof(float));
    model->b_ff2 = malloc(embedding_dim * sizeof(float));
    model->output_projection = malloc(embedding_dim * vocab_size * sizeof(float));
    model->output_bias = malloc(vocab_size * sizeof(float));
    
    if (!model->token_embeddings || !model->position_embeddings || !model->W_q ||
        !model->W_k || !model->W_v || !model->W_o || !model->output_projection) {
        return MODEL_ALLOCATION_FAILURE;
    }
    
    /* Allocate gradients */
    model->token_embeddings_grad = malloc(vocab_size * embedding_dim * sizeof(float));
    model->W_q_grad = malloc(embedding_dim * embedding_dim * sizeof(float));
    model->W_k_grad = malloc(embedding_dim * embedding_dim * sizeof(float));
    model->W_v_grad = malloc(embedding_dim * embedding_dim * sizeof(float));
    model->W_o_grad = malloc(embedding_dim * embedding_dim * sizeof(float));
    model->W_ff1_grad = malloc(embedding_dim * ffn_dim * sizeof(float));
    model->b_ff1_grad = malloc(ffn_dim * sizeof(float));
    model->W_ff2_grad = malloc(ffn_dim * embedding_dim * sizeof(float));
    model->b_ff2_grad = malloc(embedding_dim * sizeof(float));
    model->output_projection_grad = malloc(embedding_dim * vocab_size * sizeof(float));
    model->output_bias_grad = malloc(vocab_size * sizeof(float));
    
    if (!model->W_q_grad || !model->output_projection_grad) {
        return MODEL_ALLOCATION_FAILURE;
    }
    
    /* Phase 2: Allocate layer normalization parameters */
    model->ln_gamma_attn = malloc(embedding_dim * sizeof(float));
    model->ln_beta_attn = malloc(embedding_dim * sizeof(float));
    model->ln_gamma_ffn = malloc(embedding_dim * sizeof(float));
    model->ln_beta_ffn = malloc(embedding_dim * sizeof(float));
    model->ln_gamma_attn_grad = malloc(embedding_dim * sizeof(float));
    model->ln_beta_attn_grad = malloc(embedding_dim * sizeof(float));
    model->ln_gamma_ffn_grad = malloc(embedding_dim * sizeof(float));
    model->ln_beta_ffn_grad = malloc(embedding_dim * sizeof(float));
    
    if (!model->ln_gamma_attn || !model->ln_gamma_ffn) {
        return MODEL_ALLOCATION_FAILURE;
    }
    
    /* Initialize layer norm parameters (gamma=1, beta=0) */
    for (size_t i = 0; i < embedding_dim; i++) {
        model->ln_gamma_attn[i] = 1.0f;
        model->ln_beta_attn[i] = 0.0f;
        model->ln_gamma_ffn[i] = 1.0f;
        model->ln_beta_ffn[i] = 0.0f;
    }
    
    /* Phase 2: Initialize learning metrics */
    model->metrics.history_capacity = 1000;
    model->metrics.loss_history = malloc(1000 * sizeof(float));
    model->metrics.history_size = 0;
    model->metrics.best_loss = 1e9f;
    model->metrics.worst_loss = 0.0f;
    model->metrics.avg_loss = 0.0f;
    model->metrics.learning_rate = model->learning_rate;
    model->metrics.initial_learning_rate = model->learning_rate;
    model->metrics.steps_without_improvement = 0;
    
    /* Initialize weights with Xavier initialization */
    xavier_init(model->token_embeddings, vocab_size * embedding_dim, 1, embedding_dim);
    xavier_init(model->W_q, embedding_dim * embedding_dim, embedding_dim, embedding_dim);
    xavier_init(model->W_k, embedding_dim * embedding_dim, embedding_dim, embedding_dim);
    xavier_init(model->W_v, embedding_dim * embedding_dim, embedding_dim, embedding_dim);
    xavier_init(model->W_o, embedding_dim * embedding_dim, embedding_dim, embedding_dim);
    xavier_init(model->W_ff1, embedding_dim * ffn_dim, embedding_dim, ffn_dim);
    xavier_init(model->W_ff2, ffn_dim * embedding_dim, ffn_dim, embedding_dim);
    xavier_init(model->output_projection, embedding_dim * vocab_size, embedding_dim, vocab_size);
    
    /* Initialize biases to zero */
    memset(model->b_ff1, 0, ffn_dim * sizeof(float));
    memset(model->b_ff2, 0, embedding_dim * sizeof(float));
    memset(model->output_bias, 0, vocab_size * sizeof(float));
    
    /* Compute positional encoding */
    compute_positional_encoding(model->position_embeddings, max_seq_len, embedding_dim);
    
    DEBUG_PRINT("Neural model initialized successfully\n");
    
    return MODEL_SUCCESS;
}

/* Multi-head attention forward pass */
static void multihead_attention(neural_model_t *model,
                               float *sequence,        // seq_len x embedding_dim
                               size_t seq_len,
                               float *output) {        // seq_len x embedding_dim
    
    size_t embedding_dim = model->embedding_dim;
    size_t num_heads = model->num_heads;
    size_t head_dim = embedding_dim / num_heads;
    
    DEBUG_PRINT("Multi-head attention: seq_len=%zu, num_heads=%zu, head_dim=%zu\n", 
                seq_len, num_heads, head_dim);
    
    float *Q = malloc(seq_len * embedding_dim * sizeof(float));
    float *K = malloc(seq_len * embedding_dim * sizeof(float));
    float *V = malloc(seq_len * embedding_dim * sizeof(float));
    float *attention_scores = malloc(seq_len * seq_len * sizeof(float));
    float *attention_probs = malloc(seq_len * seq_len * sizeof(float));
    
    /* Compute Q, K, V projections */
    matrix_multiply(sequence, model->W_q, Q, seq_len, embedding_dim, embedding_dim);
    matrix_multiply(sequence, model->W_k, K, seq_len, embedding_dim, embedding_dim);
    matrix_multiply(sequence, model->W_v, V, seq_len, embedding_dim, embedding_dim);
    
    /* Compute attention scores for each head and aggregate */
    memset(output, 0, seq_len * embedding_dim * sizeof(float));
    
    for (size_t head = 0; head < num_heads; head++) {
        /* Compute attention scores: (Q @ K^T) / sqrt(head_dim) */
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    size_t q_idx = i * embedding_dim + head * head_dim + d;
                    size_t k_idx = j * embedding_dim + head * head_dim + d;
                    score += Q[q_idx] * K[k_idx];
                }
                attention_scores[i * seq_len + j] = score / sqrtf((float)head_dim);
            }
        }
        
        /* Apply softmax */
        for (size_t i = 0; i < seq_len; i++) {
            softmax(&attention_scores[i * seq_len], seq_len);
        }
        
        /* Apply attention to values: attention_probs @ V */
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    float prob = attention_scores[i * seq_len + j];
                    size_t v_idx = j * embedding_dim + head * head_dim + d;
                    sum += prob * V[v_idx];
                }
                output[i * embedding_dim + head * head_dim + d] += sum;
            }
        }
    }
    
    /* Output projection */
    float *temp = malloc(seq_len * embedding_dim * sizeof(float));
    memcpy(temp, output, seq_len * embedding_dim * sizeof(float));
    matrix_multiply(temp, model->W_o, output, seq_len, embedding_dim, embedding_dim);
    
    free(Q);
    free(K);
    free(V);
    free(attention_scores);
    free(attention_probs);
    free(temp);
}

/* Forward pass through neural network */
model_errors_t model_forward(neural_model_t *model,
                             uint32_t *token_ids,
                             size_t seq_len,
                             float *output_logits) {
    
    if (!model || !token_ids || !output_logits) {
        return MODEL_INVALID_INPUT;
    }
    
    if (seq_len > MAX_SEQ_LEN) {
        return MODEL_INVALID_INPUT;
    }
    
    size_t embedding_dim = model->embedding_dim;
    
    DEBUG_PRINT("Model forward pass: seq_len=%zu, embedding_dim=%zu\n", seq_len, embedding_dim);
    
    /* 1. Embed tokens and add positional encoding */
    float *embeddings = malloc(seq_len * embedding_dim * sizeof(float));
    
    for (size_t i = 0; i < seq_len; i++) {
        uint32_t token_id = token_ids[i];
        if (token_id >= model->vocab_size) token_id = 0; // OOV handling
        
        for (size_t d = 0; d < embedding_dim; d++) {
            embeddings[i * embedding_dim + d] = 
                model->token_embeddings[token_id * embedding_dim + d] +
                model->position_embeddings[i * embedding_dim + d];
        }
    }
    
    /* 2. Multi-head attention */
    float *attention_output = malloc(seq_len * embedding_dim * sizeof(float));
    multihead_attention(model, embeddings, seq_len, attention_output);
    
    /* 3. Residual connection + Layer Normalization (Phase 2) */
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < embedding_dim; d++) {
            attention_output[i * embedding_dim + d] += embeddings[i * embedding_dim + d];
        }
        /* Apply layer normalization to each position */
        layer_norm_internal(&attention_output[i * embedding_dim], 
                          model->ln_gamma_attn, model->ln_beta_attn, 
                          embedding_dim, 1e-6f);
    }
    
    /* 4. Feedforward network */
    float *ff_hidden = malloc(seq_len * embedding_dim * 4 * sizeof(float));
    float *ff_output = malloc(seq_len * embedding_dim * sizeof(float));
    
    matrix_multiply(attention_output, model->W_ff1, ff_hidden, 
                   seq_len, embedding_dim, embedding_dim * 4);
    
    /* Add bias and ReLU */
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < embedding_dim * 4; d++) {
            ff_hidden[i * embedding_dim * 4 + d] = 
                relu(ff_hidden[i * embedding_dim * 4 + d] + model->b_ff1[d]);
        }
    }
    
    matrix_multiply(ff_hidden, model->W_ff2, ff_output,
                   seq_len, embedding_dim * 4, embedding_dim);
    
    /* Add bias and residual + Layer Normalization (Phase 2) */
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < embedding_dim; d++) {
            ff_output[i * embedding_dim + d] = 
                ff_output[i * embedding_dim + d] + model->b_ff2[d] + attention_output[i * embedding_dim + d];
        }
        /* Apply layer normalization to each position */
        layer_norm_internal(&ff_output[i * embedding_dim],
                          model->ln_gamma_ffn, model->ln_beta_ffn,
                          embedding_dim, 1e-6f);
    }
    
    /* 5. Output projection to vocabulary (use last token) */
    float *last_hidden = &ff_output[(seq_len - 1) * embedding_dim];
    matrix_multiply(last_hidden, model->output_projection, output_logits, 1, embedding_dim, model->vocab_size);
    
    /* Add output bias */
    for (size_t i = 0; i < model->vocab_size; i++) {
        output_logits[i] += model->output_bias[i];
    }
    
    free(embeddings);
    free(attention_output);
    free(ff_hidden);
    free(ff_output);
    
    return MODEL_SUCCESS;
}

/* Training step with gradient descent */
model_errors_t model_train_step(neural_model_t *model,
                                uint32_t *token_ids,
                                uint32_t target_id,
                                size_t seq_len) {
    
    if (!model || !token_ids || target_id >= model->vocab_size) {
        return MODEL_INVALID_INPUT;
    }
    
    /* Forward pass */
    float *logits = malloc(model->vocab_size * sizeof(float));
    model_forward(model, token_ids, seq_len, logits);
    
    /* Compute cross-entropy loss */
    softmax(logits, model->vocab_size);
    float loss = -logf(fmaxf(logits[target_id], 1e-7f));
    model->current_loss = loss;
    
    DEBUG_PRINT("Training step: loss=%.4f, target_id=%u\n", loss, target_id);
    
    /* Compute gradient of output logits */
    float *grad_logits = malloc(model->vocab_size * sizeof(float));
    memcpy(grad_logits, logits, model->vocab_size * sizeof(float));
    grad_logits[target_id] -= 1.0f;  // Gradient for cross-entropy
    
    /* Simple gradient descent update on output layer */
    for (size_t i = 0; i < model->vocab_size; i++) {
        model->output_bias[i] -= model->learning_rate * grad_logits[i];
    }
    
    model->training_steps++;
    
    /* Phase 2: Update learning metrics */
    if (model->metrics.history_size < model->metrics.history_capacity) {
        model->metrics.loss_history[model->metrics.history_size] = loss;
        model->metrics.history_size++;
    }
    
    if (loss < model->metrics.best_loss) {
        model->metrics.best_loss = loss;
        model->metrics.steps_without_improvement = 0;
    } else {
        model->metrics.steps_without_improvement++;
    }
    
    if (loss > model->metrics.worst_loss) {
        model->metrics.worst_loss = loss;
    }
    
    /* Update running average loss */
    model->metrics.avg_loss = (model->metrics.avg_loss * (model->training_steps - 1) + loss) / model->training_steps;
    
    /* Learning rate scheduling - reduce if no improvement */
    if (model->metrics.steps_without_improvement > 10) {
        model->metrics.learning_rate *= 0.99f;
        model->learning_rate = model->metrics.learning_rate;
        model->metrics.steps_without_improvement = 0;
    }
    
    free(logits);
    free(grad_logits);
    
    return MODEL_SUCCESS;
}

/* Predict next token */
uint32_t model_predict_next_token(neural_model_t *model,
                                  uint32_t *token_ids,
                                  size_t seq_len) {
    
    float *logits = malloc(model->vocab_size * sizeof(float));
    model_forward(model, token_ids, seq_len, logits);
    
    /* Find argmax (greedy prediction) */
    uint32_t next_token = 0;
    float max_logit = logits[0];
    
    for (uint32_t i = 1; i < model->vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            next_token = i;
        }
    }
    
    free(logits);
    return next_token;
}

/* Save model to file */
model_errors_t model_save(neural_model_t *model, const char *filename) {
    if (!model || !filename) {
        return MODEL_INVALID_INPUT;
    }
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        return MODEL_IO_ERROR;
    }
    
    DEBUG_PRINT("Saving model to %s\n", filename);
    
    /* Write header with dimensions */
    fwrite(&model->vocab_size, sizeof(size_t), 1, f);
    fwrite(&model->embedding_dim, sizeof(size_t), 1, f);
    fwrite(&model->num_heads, sizeof(size_t), 1, f);
    fwrite(&model->training_steps, sizeof(uint32_t), 1, f);
    fwrite(&model->current_loss, sizeof(float), 1, f);
    
    size_t ffn_dim = model->embedding_dim * 4;
    
    /* Write weight matrices */
    fwrite(model->token_embeddings, sizeof(float), model->vocab_size * model->embedding_dim, f);
    fwrite(model->W_q, sizeof(float), model->embedding_dim * model->embedding_dim, f);
    fwrite(model->W_k, sizeof(float), model->embedding_dim * model->embedding_dim, f);
    fwrite(model->W_v, sizeof(float), model->embedding_dim * model->embedding_dim, f);
    fwrite(model->W_o, sizeof(float), model->embedding_dim * model->embedding_dim, f);
    fwrite(model->W_ff1, sizeof(float), model->embedding_dim * ffn_dim, f);
    fwrite(model->b_ff1, sizeof(float), ffn_dim, f);
    fwrite(model->W_ff2, sizeof(float), ffn_dim * model->embedding_dim, f);
    fwrite(model->b_ff2, sizeof(float), model->embedding_dim, f);
    fwrite(model->output_projection, sizeof(float), model->embedding_dim * model->vocab_size, f);
    fwrite(model->output_bias, sizeof(float), model->vocab_size, f);
    
    fclose(f);
    DEBUG_PRINT("Model saved successfully\n");
    
    return MODEL_SUCCESS;
}

/* Load model from file */
model_errors_t model_load(neural_model_t *model, const char *filename) {
    if (!model || !filename) {
        return MODEL_INVALID_INPUT;
    }
    
    FILE *f = fopen(filename, "rb");
    if (!f) {
        return MODEL_IO_ERROR;
    }
    
    DEBUG_PRINT("Loading model from %s\n", filename);
    
    /* Read header */
    size_t vocab_size, embedding_dim, num_heads;
    uint32_t training_steps;
    float current_loss;
    
    fread(&vocab_size, sizeof(size_t), 1, f);
    fread(&embedding_dim, sizeof(size_t), 1, f);
    fread(&num_heads, sizeof(size_t), 1, f);
    fread(&training_steps, sizeof(uint32_t), 1, f);
    fread(&current_loss, sizeof(float), 1, f);
    
    /* Initialize model if not already done */
    if (model->vocab_size != vocab_size || model->embedding_dim != embedding_dim) {
        model_free(model);
        model_new(model, vocab_size, embedding_dim, num_heads, MAX_SEQ_LEN);
    }
    
    model->training_steps = training_steps;
    model->current_loss = current_loss;
    
    size_t ffn_dim = embedding_dim * 4;
    
    /* Read weight matrices */
    fread(model->token_embeddings, sizeof(float), vocab_size * embedding_dim, f);
    fread(model->W_q, sizeof(float), embedding_dim * embedding_dim, f);
    fread(model->W_k, sizeof(float), embedding_dim * embedding_dim, f);
    fread(model->W_v, sizeof(float), embedding_dim * embedding_dim, f);
    fread(model->W_o, sizeof(float), embedding_dim * embedding_dim, f);
    fread(model->W_ff1, sizeof(float), embedding_dim * ffn_dim, f);
    fread(model->b_ff1, sizeof(float), ffn_dim, f);
    fread(model->W_ff2, sizeof(float), ffn_dim * embedding_dim, f);
    fread(model->b_ff2, sizeof(float), embedding_dim, f);
    fread(model->output_projection, sizeof(float), embedding_dim * vocab_size, f);
    fread(model->output_bias, sizeof(float), vocab_size, f);
    
    fclose(f);
    DEBUG_PRINT("Model loaded successfully (training_steps=%u, loss=%.4f)\n", 
                training_steps, current_loss);
    
    return MODEL_SUCCESS;
}

/* Free model resources */
void model_free(neural_model_t *model) {
    if (!model) return;
    
    free(model->token_embeddings);
    free(model->position_embeddings);
    free(model->W_q);
    free(model->W_k);
    free(model->W_v);
    free(model->W_o);
    free(model->W_ff1);
    free(model->b_ff1);
    free(model->W_ff2);
    free(model->b_ff2);
    free(model->output_projection);
    free(model->output_bias);
    
    free(model->token_embeddings_grad);
    free(model->W_q_grad);
    free(model->W_k_grad);
    free(model->W_v_grad);
    free(model->W_o_grad);
    free(model->W_ff1_grad);
    free(model->b_ff1_grad);
    free(model->W_ff2_grad);
    free(model->b_ff2_grad);
    free(model->output_projection_grad);
    free(model->output_bias_grad);
    
    /* Phase 2: Free layer normalization parameters and gradients */
    free(model->ln_gamma_attn);
    free(model->ln_beta_attn);
    free(model->ln_gamma_ffn);
    free(model->ln_beta_ffn);
    free(model->ln_gamma_attn_grad);
    free(model->ln_beta_attn_grad);
    free(model->ln_gamma_ffn_grad);
    free(model->ln_beta_ffn_grad);
    
    /* Phase 2: Free learning metrics */
    free(model->metrics.loss_history);
    
    memset(model, 0, sizeof(neural_model_t));
}

/* Phase 2: Public layer normalization wrapper */
void layer_normalize(float *input, float *output, size_t size,
                     float *gamma, float *beta, float epsilon) {
    /* Copy input to output */
    memcpy(output, input, size * sizeof(float));
    
    /* Apply layer normalization in-place */
    layer_norm_internal(output, gamma, beta, size, epsilon);
}

/* Phase 2: Update learning rate based on training progress */
void update_learning_rate(neural_model_t *model) {
    if (!model) return;
    
    /* Implement learning rate decay: reduce if loss plateaus */
    if (model->metrics.steps_without_improvement > 20) {
        model->metrics.learning_rate *= 0.95f;  // More aggressive decay
        model->learning_rate = model->metrics.learning_rate;
        model->metrics.steps_without_improvement = 0;
        
        DEBUG_PRINT("Learning rate reduced to %.6f\n", model->metrics.learning_rate);
    }
}

/* Phase 2: Get learning metrics */
void model_get_metrics(neural_model_t *model, learning_metrics_t *out_metrics) {
    if (!model || !out_metrics) return;
    
    /* Copy metrics to output */
    *out_metrics = model->metrics;
}

/* Phase 2: Print training metrics and statistics */
void model_print_metrics(neural_model_t *model) {
    if (!model) return;
    
    printf("\n=== Phase 2: Learning Metrics ===\n");
    printf("Training steps: %u\n", model->training_steps);
    printf("Current loss: %.6f\n", model->current_loss);
    printf("Best loss: %.6f\n", model->metrics.best_loss);
    printf("Worst loss: %.6f\n", model->metrics.worst_loss);
    printf("Average loss: %.6f\n", model->metrics.avg_loss);
    printf("Current learning rate: %.8f\n", model->metrics.learning_rate);
    printf("Steps without improvement: %u\n", model->metrics.steps_without_improvement);
    
    /* Print last 10 losses for learning curve */
    if (model->metrics.history_size > 0) {
        printf("\nLast %zu loss values: ", 
               model->metrics.history_size < 10 ? model->metrics.history_size : 10);
        size_t start = model->metrics.history_size < 10 ? 0 : model->metrics.history_size - 10;
        for (size_t i = start; i < model->metrics.history_size; i++) {
            printf("%.4f ", model->metrics.loss_history[i]);
        }
        printf("\n");
    }
    
    printf("===================================\n\n");
}
