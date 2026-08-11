#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stddef.h>
#include <stdint.h>

/* The matmul kernels and their selection policy live in their own module.
 * Included here so every existing user of tensor_ops.h keeps seeing
 * matrix_multiply() and friends unchanged. */
#include "core/matmul.h"

/* Low-level numeric primitives shared across the model: softmax, layer norm,
 * ReLU, dropout, positional encoding. None of these know about
 * neural_model_t - they operate purely on caller-supplied buffers, so they
 * can be unit-tested and reused independently of the transformer/training
 * modules built on top of them. */

/* Xavier/Glorot uniform initialization for a weight matrix of `size`
 * elements with the given fan-in/fan-out.
 *
 * Draws from the caller's stream (core/rng.h), advancing `*rng_state` once per
 * element. It used to draw from rand(), which made "same seed" mean "same seed
 * on the same C library" and left initial weights unreproducible off the
 * machine that produced them. The stream is a parameter rather than a global
 * so that initializing a tensor cannot be perturbed by an unrelated caller's
 * draws. */
void xavier_init(float *weights, size_t size, size_t fan_in, size_t fan_out,
                 uint64_t *rng_state);

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

/* RMSNorm variants: y = gamma*x/sqrt(mean(x^2)+epsilon). There is no beta
 * parameter. The cached normalized rows and RMS feed the matching backward. */
void rms_normalize(const float *input, float *output, size_t size,
                   const float *gamma, float epsilon);
void rms_norm_forward_cached(const float *restrict input,
                             float *restrict xhat_out,
                             float *restrict rms_out,
                             float *restrict output,
                             const float *restrict gamma,
                             size_t seq_len, size_t size, float epsilon);
void rms_norm_backward(const float *dL_dout,
                       const float *restrict xhat,
                       const float *restrict rms,
                       const float *restrict gamma,
                       float *restrict gamma_grad,
                       float *dL_dinput, size_t seq_len, size_t size);

/* Positional encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1)
 * = cos(...). Returns 0 on success, -1 on allocation failure (caller must
 * treat pos_embed as uninitialized in that case - it is not touched). */
int compute_positional_encoding(float *pos_embed, size_t seq_len, size_t embedding_dim);

/* Compute one sinusoidal row for an absolute position. */
int compute_positional_encoding_at(float *pos_embed, size_t position,
                                   size_t embedding_dim);

/* Inverted dropout, one row (seq_len positions x `size` elements each).
 * When is_training is false or rate <= 0, this is a no-op identity (mask
 * left as all-ones so dropout_backward stays a correct no-op too). When
 * active, each element is independently kept with probability (1-rate)
 * and scaled by 1/(1-rate) so the expected activation magnitude is
 * unchanged - this is what lets inference skip dropout entirely without
 * rescaling. `mask_out` caches the 1.0/0.0 keep-mask for backward.
 *
 * `rng_state` is required and must not be NULL. It is an explicit stream
 * (core/rng.h) so that a checkpoint storing eight bytes resumes the exact
 * future mask sequence. There used to be a second, rand()-based overload that
 * this one fell back to when handed NULL; it was unreproducible across C
 * libraries, and a fallback that silently degrades a reproducibility guarantee
 * is worse than a missing argument. */
void dropout_forward(float *x, float *mask_out, size_t total_size, float rate,
                     int is_training, uint64_t *rng_state);

/* Backprop through dropout_forward: dL_dx *= mask / (1 - rate), in place.
 * Must be called with the same `rate` used in the matching forward call. */
void dropout_backward(float *dL_dx, const float *mask, size_t total_size, float rate);

#endif // TENSOR_OPS_H
