/*
 * Forward and backward matmul dispatch policy, shared by transformer.c,
 * training.c, and lm_head.c. See matmul_dispatch.h for why the forward and
 * backward rules differ and where the backward thresholds came from.
 */

#include "core/matmul_dispatch.h"
#include "core/matmul.h"
#include "backends/gpu/gpu_matmul.h"
#include <stdint.h>

void model_dispatch_matmul(neural_model_t *model, float *A, float *B, float *C,
                           size_t m, size_t k, size_t n) {
    if (model->use_scalar_matmul) {
        matrix_multiply_scalar(A, B, C, m, k, n);
        return;
    }
    /* gpu_matmul_available() is checked at every call (not cached once) so a
     * model can be constructed and used identically regardless of whether the
     * machine has an NVIDIA GPU - the flag is simply a no-op without one. */
    if (model->use_gpu && gpu_matmul_available() && gpu_matmul(A, B, C, m, k, n) == 0) {
        return;
    }
    matrix_multiply(A, B, C, m, k, n);
}

/* Guards the multiplication against overflow rather than trusting model
 * dimensions to stay small. */
int model_gpu_backward_worthwhile(size_t m, size_t k, size_t n, size_t min_work) {
    if (m == 0 || k == 0 || n == 0) return 0;
    if (k > SIZE_MAX / m) return 1; /* astronomically large: certainly worth it */
    size_t mk = m * k;
    if (n > SIZE_MAX / mk) return 1;
    return mk * n >= min_work;
}

void model_dispatch_backward_input(neural_model_t *model, float *dC, float *B,
                                   float *dA, size_t m, size_t k, size_t n) {
    if (model->use_gpu &&
        model_gpu_backward_worthwhile(m, k, n, GPU_BACKWARD_INPUT_MIN_WORK) &&
        gpu_matmul_available() &&
        gpu_matmul_backward_input(dC, B, dA, m, k, n) == 0) {
        return;
    }
    matmul_backward_input(dC, B, dA, m, k, n);
}

void model_dispatch_backward_weight(neural_model_t *model, float *A, float *dC,
                                    float *dB, size_t m, size_t k, size_t n) {
    if (model->use_gpu &&
        model_gpu_backward_worthwhile(m, k, n, GPU_BACKWARD_WEIGHT_MIN_WORK) &&
        gpu_matmul_available() &&
        gpu_matmul_backward_weight(A, dC, dB, m, k, n) == 0) {
        return;
    }
    matmul_backward_weight(A, dC, dB, m, k, n);
}
