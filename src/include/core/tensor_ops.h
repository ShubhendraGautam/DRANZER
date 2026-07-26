#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stddef.h>

/* Low-level numeric primitives shared across the model: matmul, softmax,
 * layer norm, ReLU, dropout, positional encoding. None of these know about
 * neural_model_t - they operate purely on caller-supplied buffers, so they
 * can be unit-tested and reused independently of the transformer/training
 * modules built on top of them. */

/* Xavier/Glorot uniform initialization for a weight matrix of `size`
 * elements with the given fan-in/fan-out. */
void xavier_init(float *weights, size_t size, size_t fan_in, size_t fan_out);

/* Softmax over `size` elements, in place. Numerically stable (max-subtracted). */
void softmax(float *values, size_t size);

/* Backprop through one row of softmax. `probs` is the cached forward
 * output (read-only). `dL_dprobs` is overwritten in place with dL/dscores:
 * safe because the row's dot product is computed from the original values
 * before anything in the row is overwritten. */
void softmax_backward(float *probs, float *dL_dprobs, size_t size);

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* Derivative of ReLU. Safe to call with either the pre- or post-activation
 * value: post > 0 iff pre > 0, so the cached post-ReLU activations (which
 * is all the model caches) give the same result. */
static inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

/* Matrix multiplication with cache blocking for improved locality.
 * C (m x n) = A (m x k) @ B (k x n). Zeroes C first.
 *
 * Parallelized over (ii, jj) row/col blocks when built with OMP=1: each
 * block owns a disjoint region of C, so threads never write the same
 * output and there is no cross-thread reduction - results are bit-identical
 * to a serial build. collapse(2) matters here: with only e.g. 2-8 row
 * blocks for typical seq_len-sized m, parallelizing ii alone starves most
 * threads; collapsing ii and jj together gives up to
 * ceil(m/BLOCK_SIZE)*ceil(n/BLOCK_SIZE) independent tasks instead. */
void matrix_multiply(float *restrict A, float *restrict B, float *restrict C,
                      size_t m, size_t k, size_t n);

/* Backprop through C = A @ B (A: m x k, B: k x n, C: m x n).
 * Accumulates (+=) into dA; caller must zero dA first for a fresh gradient.
 * Parallel over i: each iteration owns a disjoint row of dA. */
void matmul_backward_input(float *restrict dC, float *restrict B, float *restrict dA,
                            size_t m, size_t k, size_t n);

/* Backprop through C = A @ B (A: m x k, B: k x n, C: m x n).
 * Accumulates (+=) into dB; caller must zero dB first for a fresh gradient.
 * Parallel over l: each iteration owns a disjoint row of dB. */
void matmul_backward_weight(float *restrict A, float *restrict dC, float *restrict dB,
                             size_t m, size_t k, size_t n);

/* In-place layer normalization + scale/shift, single row of `size`
 * elements. Used by the public layer_normalize() wrapper only; the
 * training path uses layer_norm_forward_cached below since it needs to
 * keep xhat/std around for backward. */
void layer_norm_internal(float *input, float *gamma, float *beta,
                          size_t size, float epsilon);

/* Public layer normalization wrapper: normalizes `input` into `output`,
 * leaving `input` untouched. */
void layer_normalize(float *input, float *output, size_t size,
                      float *gamma, float *beta, float epsilon);

/* Forward layer norm over `seq_len` rows of `size` elements, caching xhat
 * and std_dev per row (needed by layer_norm_backward). */
void layer_norm_forward_cached(float *restrict pre_ln_sum, float *restrict xhat_out, float *restrict std_out,
                                float *restrict ln_out, float *restrict gamma, float *restrict beta,
                                size_t seq_len, size_t size, float epsilon);

/* Backprop through y = gamma*xhat+beta (LayerNorm), one row of `size`
 * elements at a time, seq_len rows. gamma/beta are shared across rows, so
 * their gradients accumulate (+=) across all seq_len positions - caller
 * must zero gamma_grad/beta_grad first for a fresh gradient. dL_dout and
 * dL_dinput may safely be the same buffer (each row is fully read before
 * it is written). */
void layer_norm_backward(float *dL_dout, float *restrict xhat, float *restrict std,
                          float *restrict gamma, float *restrict gamma_grad, float *restrict beta_grad,
                          float *dL_dinput, size_t seq_len, size_t size);

/* Positional encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1)
 * = cos(...). Returns 0 on success, -1 on allocation failure (caller must
 * treat pos_embed as uninitialized in that case - it is not touched). */
int compute_positional_encoding(float *pos_embed, size_t seq_len, size_t embedding_dim);

/* Inverted dropout, one row (seq_len positions x `size` elements each).
 * When is_training is false or rate <= 0, this is a no-op identity (mask
 * left as all-ones so dropout_backward stays a correct no-op too). When
 * active, each element is independently kept with probability (1-rate)
 * and scaled by 1/(1-rate) so the expected activation magnitude is
 * unchanged - this is what lets inference skip dropout entirely without
 * rescaling. `mask_out` caches the 1.0/0.0 keep-mask for backward. */
void dropout_forward(float *x, float *mask_out, size_t total_size, float rate, int is_training);

/* Backprop through dropout_forward: dL_dx *= mask / (1 - rate), in place.
 * Must be called with the same `rate` used in the matching forward call. */
void dropout_backward(float *dL_dx, const float *mask, size_t total_size, float rate);

#endif // TENSOR_OPS_H
