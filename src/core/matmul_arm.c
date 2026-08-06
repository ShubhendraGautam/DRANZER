/*
 * NEON matrix-multiplication kernel for AArch64.
 *
 * Structurally identical to block_avx2() in core/matmul_x86.c and to
 * kernel_tiled_mr4() in core/matmul.c: cache blocked, four rows of A in
 * flight, vectorized along j so each C element keeps accumulating over l in
 * increasing order. Four columns per vector instead of eight.
 *
 * No target attribute is needed here. Advanced SIMD is mandatory in ARMv8-A,
 * so on aarch64 it is part of the baseline the compiler already targets - the
 * runtime check in cpu_features.c reports it unconditionally for the same
 * reason. 32-bit ARM, where NEON is optional, is not covered.
 */

#include "core/matmul_simd.h"

#ifdef DRANZER_HAVE_NEON

#include <arm_neon.h>
#include <string.h>

static void block_neon(const float *restrict A, const float *restrict B,
                       float *restrict C, size_t k, size_t n,
                       size_t i_start, size_t i_limit,
                       size_t j_start, size_t j_limit,
                       size_t l_start, size_t l_limit) {
    size_t i = i_start;

    for (; i + 4 <= i_limit; i += 4) {
        float *c0 = &C[i * n];
        float *c1 = c0 + n;
        float *c2 = c1 + n;
        float *c3 = c2 + n;
        for (size_t l = l_start; l < l_limit; l++) {
            const float a0 = A[i * k + l];
            const float a1 = A[(i + 1) * k + l];
            const float a2 = A[(i + 2) * k + l];
            const float a3 = A[(i + 3) * k + l];
            const float *row_b = &B[l * n];

            size_t j = j_start;
            for (; j + 4 <= j_limit; j += 4) {
                /* vfmaq_n_f32(acc, vector, scalar) is a fused multiply-add
                 * against a scalar operand, so no broadcast register is
                 * needed - the scalar rides in the instruction encoding. */
                const float32x4_t b = vld1q_f32(row_b + j);
                vst1q_f32(c0 + j, vfmaq_n_f32(vld1q_f32(c0 + j), b, a0));
                vst1q_f32(c1 + j, vfmaq_n_f32(vld1q_f32(c1 + j), b, a1));
                vst1q_f32(c2 + j, vfmaq_n_f32(vld1q_f32(c2 + j), b, a2));
                vst1q_f32(c3 + j, vfmaq_n_f32(vld1q_f32(c3 + j), b, a3));
            }
            for (; j < j_limit; j++) {
                const float b = row_b[j];
                c0[j] += a0 * b;
                c1[j] += a1 * b;
                c2[j] += a2 * b;
                c3[j] += a3 * b;
            }
        }
    }

    /* Fewer than four rows left - the whole of a single-token decode. */
    for (; i < i_limit; i++) {
        float *row_out = &C[i * n];
        for (size_t l = l_start; l < l_limit; l++) {
            const float a = A[i * k + l];
            const float *row_b = &B[l * n];

            size_t j = j_start;
            for (; j + 4 <= j_limit; j += 4) {
                vst1q_f32(row_out + j,
                          vfmaq_n_f32(vld1q_f32(row_out + j),
                                      vld1q_f32(row_b + j), a));
            }
            for (; j < j_limit; j++) {
                row_out[j] += a * row_b[j];
            }
        }
    }
}

void matmul_kernel_neon_mr4(const float *restrict A, const float *restrict B,
                            float *restrict C,
                            size_t m, size_t k, size_t n, size_t tile) {
    memset(C, 0, m * n * sizeof(float));

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (size_t ii = 0; ii < m; ii += tile) {
        for (size_t jj = 0; jj < n; jj += tile) {
            size_t i_limit = (ii + tile > m) ? m : ii + tile;
            size_t j_limit = (jj + tile > n) ? n : jj + tile;
            for (size_t ll = 0; ll < k; ll += tile) {
                size_t l_limit = (ll + tile > k) ? k : ll + tile;
                block_neon(A, B, C, k, n, ii, i_limit, jj, j_limit, ll, l_limit);
            }
        }
    }
}

#else /* !DRANZER_HAVE_NEON */

/* Compiled on every architecture; see the matching note in core/matmul_x86.c. */
typedef int dranzer_matmul_arm_not_supported;

#endif /* DRANZER_HAVE_NEON */
