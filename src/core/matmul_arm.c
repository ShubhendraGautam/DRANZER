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

    /* Loop order j/l with register-resident accumulators, matching
     * block_avx2() in core/matmul_x86.c. That arrangement replaced an l/j one
     * that read and wrote C on every step of k, which measured 1.5-2x slower
     * on x86 and produced a compiler disagreement that took a disassembly to
     * explain (docs/matmul.md).
     *
     * Applied here on the structural argument alone: no AArch64 hardware was
     * available to measure it. What is verified is that it still agrees with
     * the scalar reference, which `test_matmul_kernels.c` checks wherever this
     * kernel can run. The performance claim is x86's, not this kernel's - and
     * `matmul_select()` still declines to select it for exactly that reason. */
    for (; i + 4 <= i_limit; i += 4) {
        float *c0 = &C[i * n];
        float *c1 = c0 + n;
        float *c2 = c1 + n;
        float *c3 = c2 + n;
        const float *a0r = &A[i * k];
        const float *a1r = a0r + k;
        const float *a2r = a1r + k;
        const float *a3r = a2r + k;

        size_t j = j_start;
        for (; j + 4 <= j_limit; j += 4) {
            float32x4_t acc0 = vld1q_f32(c0 + j);
            float32x4_t acc1 = vld1q_f32(c1 + j);
            float32x4_t acc2 = vld1q_f32(c2 + j);
            float32x4_t acc3 = vld1q_f32(c3 + j);

            for (size_t l = l_start; l < l_limit; l++) {
                /* vfmaq_n_f32(acc, vector, scalar) is a fused multiply-add
                 * against a scalar operand, so no broadcast register is
                 * needed - the scalar rides in the instruction encoding. */
                const float32x4_t b = vld1q_f32(&B[l * n + j]);
                acc0 = vfmaq_n_f32(acc0, b, a0r[l]);
                acc1 = vfmaq_n_f32(acc1, b, a1r[l]);
                acc2 = vfmaq_n_f32(acc2, b, a2r[l]);
                acc3 = vfmaq_n_f32(acc3, b, a3r[l]);
            }

            vst1q_f32(c0 + j, acc0);
            vst1q_f32(c1 + j, acc1);
            vst1q_f32(c2 + j, acc2);
            vst1q_f32(c3 + j, acc3);
        }
        for (; j < j_limit; j++) {
            float s0 = c0[j], s1 = c1[j], s2 = c2[j], s3 = c3[j];
            for (size_t l = l_start; l < l_limit; l++) {
                const float b = B[l * n + j];
                s0 += a0r[l] * b;
                s1 += a1r[l] * b;
                s2 += a2r[l] * b;
                s3 += a3r[l] * b;
            }
            c0[j] = s0; c1[j] = s1; c2[j] = s2; c3[j] = s3;
        }
    }

    /* Fewer than four rows left - the whole of a single-token decode. */
    for (; i < i_limit; i++) {
        float *row_out = &C[i * n];
        const float *a_row = &A[i * k];

        size_t j = j_start;
        for (; j + 4 <= j_limit; j += 4) {
            float32x4_t acc = vld1q_f32(row_out + j);
            for (size_t l = l_start; l < l_limit; l++) {
                acc = vfmaq_n_f32(acc, vld1q_f32(&B[l * n + j]), a_row[l]);
            }
            vst1q_f32(row_out + j, acc);
        }
        for (; j < j_limit; j++) {
            float s = row_out[j];
            for (size_t l = l_start; l < l_limit; l++) {
                s += a_row[l] * B[l * n + j];
            }
            row_out[j] = s;
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
