/*
 * Low-level numeric primitives: softmax, layer norm, dropout, positional
 * encoding. No knowledge of neural_model_t - see tensor_ops.h. The matmul
 * kernels live in their own module (core/matmul.c) because they are the one
 * primitive with several interchangeable implementations to choose between.
 */

#include "core/tensor_ops.h"
#include "core/rng.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

float gelu(float x) {
    const float inverse_sqrt_two = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inverse_sqrt_two));
}

float gelu_derivative(float x) {
    const float inverse_sqrt_two = 0.70710678118654752440f;
    const float inverse_sqrt_two_pi = 0.39894228040143267794f;
    return 0.5f * (1.0f + erff(x * inverse_sqrt_two)) +
           x * inverse_sqrt_two_pi * expf(-0.5f * x * x);
}

void xavier_init(float *weights, size_t size, size_t fan_in, size_t fan_out,
                 uint64_t *rng_state) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    for (size_t i = 0; i < size; i++) {
        weights[i] = dranzer_rng_uniform(rng_state, -limit, limit);
    }
}

void softmax(float *values, size_t size) {
    if (size == 0) return;

    float max_val = values[0];
    for (size_t i = 1; i < size; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float exp_val = expf(values[i] - max_val);
        values[i] = exp_val;
        sum += exp_val;
    }

    if (sum > 0) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < size; i++) {
            values[i] *= inv_sum;
        }
    }
}

void softmax_backward(float *probs, float *dL_dprobs, size_t size) {
    float dot = 0.0f;
    for (size_t j = 0; j < size; j++) {
        dot += dL_dprobs[j] * probs[j];
    }
    for (size_t j = 0; j < size; j++) {
        dL_dprobs[j] = probs[j] * (dL_dprobs[j] - dot);
    }
}

void layer_norm_internal(float *input, float *gamma, float *beta,
                          size_t size, float epsilon) {
    float mean = 0.0f;
    for (size_t i = 0; i < size; i++) mean += input[i];
    mean /= size;

    float variance = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;

    float std_dev = sqrtf(variance + epsilon);
    for (size_t i = 0; i < size; i++) {
        input[i] = gamma[i] * ((input[i] - mean) / std_dev) + beta[i];
    }
}

void layer_normalize(float *input, float *output, size_t size,
                      float *gamma, float *beta, float epsilon) {
    memcpy(output, input, size * sizeof(float));
    layer_norm_internal(output, gamma, beta, size, epsilon);
}

void layer_norm_forward_cached(float *restrict pre_ln_sum, float *restrict xhat_out, float *restrict std_out,
                                float *restrict ln_out, float *restrict gamma, float *restrict beta,
                                size_t seq_len, size_t size, float epsilon) {
    for (size_t i = 0; i < seq_len; i++) {
        float *row_in = &pre_ln_sum[i * size];
        float *row_xhat = &xhat_out[i * size];
        float *row_out = &ln_out[i * size];

        float mean = 0.0f;
        for (size_t d = 0; d < size; d++) mean += row_in[d];
        mean /= (float)size;

        float variance = 0.0f;
        for (size_t d = 0; d < size; d++) {
            float diff = row_in[d] - mean;
            variance += diff * diff;
        }
        variance /= (float)size;

        float std_dev = sqrtf(variance + epsilon);
        std_out[i] = std_dev;

        for (size_t d = 0; d < size; d++) {
            float xhat = (row_in[d] - mean) / std_dev;
            row_xhat[d] = xhat;
            row_out[d] = gamma[d] * xhat + beta[d];
        }
    }
}

void layer_norm_backward(float *dL_dout, float *restrict xhat, float *restrict std,
                          float *restrict gamma, float *restrict gamma_grad, float *restrict beta_grad,
                          float *dL_dinput, size_t seq_len, size_t size) {
    for (size_t i = 0; i < seq_len; i++) {
        float *row_dout = &dL_dout[i * size];
        float *row_xhat = &xhat[i * size];
        float *row_din = &dL_dinput[i * size];
        float std_dev = std[i];

        float dxhat[size]; /* VLA: `size` is embedding_dim, always small here */
        float mean_dxhat = 0.0f;
        float mean_dxhat_xhat = 0.0f;

        for (size_t d = 0; d < size; d++) {
            gamma_grad[d] += row_dout[d] * row_xhat[d];
            beta_grad[d] += row_dout[d];

            dxhat[d] = row_dout[d] * gamma[d];
            mean_dxhat += dxhat[d];
            mean_dxhat_xhat += dxhat[d] * row_xhat[d];
        }
        mean_dxhat /= (float)size;
        mean_dxhat_xhat /= (float)size;

        for (size_t d = 0; d < size; d++) {
            row_din[d] = (dxhat[d] - mean_dxhat - row_xhat[d] * mean_dxhat_xhat) / std_dev;
        }
    }
}

void rms_normalize(const float *input, float *output, size_t size,
                   const float *gamma, float epsilon) {
    float mean_square = 0.0f;
    for (size_t d = 0; d < size; d++) mean_square += input[d] * input[d];
    float rms = sqrtf(mean_square / (float)size + epsilon);
    for (size_t d = 0; d < size; d++) output[d] = gamma[d] * input[d] / rms;
}

void rms_norm_forward_cached(const float *restrict input,
                             float *restrict xhat_out,
                             float *restrict rms_out,
                             float *restrict output,
                             const float *restrict gamma,
                             size_t seq_len, size_t size, float epsilon) {
    for (size_t row = 0; row < seq_len; row++) {
        const float *row_input = &input[row * size];
        float *row_xhat = &xhat_out[row * size];
        float *row_output = &output[row * size];
        float mean_square = 0.0f;
        for (size_t d = 0; d < size; d++)
            mean_square += row_input[d] * row_input[d];
        float rms = sqrtf(mean_square / (float)size + epsilon);
        rms_out[row] = rms;
        for (size_t d = 0; d < size; d++) {
            row_xhat[d] = row_input[d] / rms;
            row_output[d] = gamma[d] * row_xhat[d];
        }
    }
}

void rms_norm_backward(const float *dL_dout,
                       const float *restrict xhat,
                       const float *restrict rms,
                       const float *restrict gamma,
                       float *restrict gamma_grad,
                       float *dL_dinput, size_t seq_len, size_t size) {
    for (size_t row = 0; row < seq_len; row++) {
        const float *row_dout = &dL_dout[row * size];
        const float *row_xhat = &xhat[row * size];
        float *row_dinput = &dL_dinput[row * size];
        float mean_scaled_dot = 0.0f;
        for (size_t d = 0; d < size; d++) {
            gamma_grad[d] += row_dout[d] * row_xhat[d];
            mean_scaled_dot += row_dout[d] * gamma[d] * row_xhat[d];
        }
        mean_scaled_dot /= (float)size;
        for (size_t d = 0; d < size; d++) {
            row_dinput[d] =
                (row_dout[d] * gamma[d] - row_xhat[d] * mean_scaled_dot) /
                rms[row];
        }
    }
}

int compute_positional_encoding(float *pos_embed, size_t seq_len, size_t embedding_dim) {
    float *dim_scales = malloc(embedding_dim * sizeof(float));
    if (dim_scales == NULL) return -1;

    for (size_t i = 0; i < embedding_dim; i++) {
        dim_scales[i] = 1.0f / powf(10000.0f, (2.0f * i) / embedding_dim);
    }

    for (size_t pos = 0; pos < seq_len; pos++) {
        for (size_t i = 0; i < embedding_dim; i++) {
            float angle = pos * dim_scales[i];
            if (i % 2 == 0) {
                pos_embed[pos * embedding_dim + i] = sinf(angle);
            } else {
                pos_embed[pos * embedding_dim + i] = cosf(angle);
            }
        }
    }

    free(dim_scales);
    return 0;
}

int compute_positional_encoding_at(float *pos_embed, size_t position,
                                   size_t embedding_dim) {
    if (!pos_embed || embedding_dim == 0) return -1;
    for (size_t i = 0; i < embedding_dim; i++) {
        float scale = 1.0f / powf(10000.0f,
                                  (2.0f * (float)i) / (float)embedding_dim);
        float angle = (float)position * scale;
        pos_embed[i] = (i % 2 == 0) ? sinf(angle) : cosf(angle);
    }
    return 0;
}

void dropout_forward(float *x, float *mask_out, size_t total_size, float rate,
                     int is_training, uint64_t *rng_state) {
    if (!is_training || rate <= 0.0f) {
        for (size_t i = 0; i < total_size; i++) mask_out[i] = 1.0f;
        return;
    }
    if (!rng_state) return; /* documented as required; nothing sane to do */

    float keep_prob = 1.0f - rate;
    float inv_keep = 1.0f / keep_prob;
    for (size_t i = 0; i < total_size; i++) {
        float keep = dranzer_rng_unit(rng_state) < (double)keep_prob ? 1.0f : 0.0f;
        mask_out[i] = keep;
        x[i] = x[i] * keep * inv_keep;
    }
}

void dropout_backward(float *dL_dx, const float *mask, size_t total_size, float rate) {
    if (rate <= 0.0f) return; /* mask is all-ones and inv_keep would be 1.0 anyway, but skip the pass */
    float inv_keep = 1.0f / (1.0f - rate);
    for (size_t i = 0; i < total_size; i++) {
        dL_dx[i] = dL_dx[i] * mask[i] * inv_keep;
    }
}
