/*
 * AVX2 and AVX-512 matrix-multiplication kernels for x86-64.
 *
 * These are the same algorithm as kernel_tiled_mr4() in core/matmul.c - cache
 * blocked, four rows of A in flight - with the innermost loop over j replaced
 * by vector loads, fused multiply-adds, and stores. Vectorizing over j rather
 * than over k is the whole reason these stay comparable to the reference: each
 * C element still accumulates its products over l in increasing order, so the
 * only numerical difference from the portable kernel is FMA contraction, not a
 * reassociated sum.
 *
 * The vector code is confined to functions carrying a `target` attribute so
 * this file compiles, and the resulting instructions are only ever reached
 * through cpu_isa_available(). That is what lets the default build ship AVX-512
 * kernels without -march and still run on a CPU that has never heard of them:
 * the instructions exist in the binary but nothing branches to them. See
 * core/cpu_features.h.
 *
 * The OpenMP loop deliberately sits in a plain untargeted function that calls
 * the targeted block through a pointer. Keeping the parallel region out of a
 * target-attributed function avoids the outlining-versus-target-attribute
 * interaction between compilers entirely, and the indirect call is amortized
 * over a whole tile x tile output block.
 */

#include "core/matmul_simd.h"

#ifdef DRANZER_HAVE_X86_SIMD

#include <immintrin.h>
#include <string.h>

/* One output block: rows [i_start, i_limit), columns [j_start, j_limit),
 * accumulating the k-slice [l_start, l_limit) into C. */
typedef void (*x86_block_fn)(const float *restrict A, const float *restrict B,
                             float *restrict C, size_t k, size_t n,
                             size_t i_start, size_t i_limit,
                             size_t j_start, size_t j_limit,
                             size_t l_start, size_t l_limit);

/* ---------------------------------------------------------------- AVX2 --- */

__attribute__((target("avx2,fma")))
static void block_avx2(const float *restrict A, const float *restrict B,
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
            const float a0s = A[i * k + l];
            const float a1s = A[(i + 1) * k + l];
            const float a2s = A[(i + 2) * k + l];
            const float a3s = A[(i + 3) * k + l];
            const __m256 a0 = _mm256_set1_ps(a0s);
            const __m256 a1 = _mm256_set1_ps(a1s);
            const __m256 a2 = _mm256_set1_ps(a2s);
            const __m256 a3 = _mm256_set1_ps(a3s);
            const float *row_b = &B[l * n];

            size_t j = j_start;
            for (; j + 8 <= j_limit; j += 8) {
                const __m256 b = _mm256_loadu_ps(row_b + j);
                _mm256_storeu_ps(c0 + j,
                                 _mm256_fmadd_ps(a0, b, _mm256_loadu_ps(c0 + j)));
                _mm256_storeu_ps(c1 + j,
                                 _mm256_fmadd_ps(a1, b, _mm256_loadu_ps(c1 + j)));
                _mm256_storeu_ps(c2 + j,
                                 _mm256_fmadd_ps(a2, b, _mm256_loadu_ps(c2 + j)));
                _mm256_storeu_ps(c3 + j,
                                 _mm256_fmadd_ps(a3, b, _mm256_loadu_ps(c3 + j)));
            }
            /* Columns past the last full vector. AVX2 has no store mask for
             * floats (maskstore exists but costs more than this loop on the
             * handful of columns a tail can hold), so the remainder is plain
             * scalar - the same arithmetic, one column at a time. */
            for (; j < j_limit; j++) {
                const float b = row_b[j];
                c0[j] += a0s * b;
                c1[j] += a1s * b;
                c2[j] += a2s * b;
                c3[j] += a3s * b;
            }
        }
    }

    /* Fewer than four rows left. Single-token decode shapes (m == 1) land
     * here for every block, so this path is vectorized too. */
    for (; i < i_limit; i++) {
        float *row_out = &C[i * n];
        for (size_t l = l_start; l < l_limit; l++) {
            const float a_scalar = A[i * k + l];
            const __m256 a = _mm256_set1_ps(a_scalar);
            const float *row_b = &B[l * n];

            size_t j = j_start;
            for (; j + 8 <= j_limit; j += 8) {
                _mm256_storeu_ps(row_out + j,
                                 _mm256_fmadd_ps(a, _mm256_loadu_ps(row_b + j),
                                                 _mm256_loadu_ps(row_out + j)));
            }
            for (; j < j_limit; j++) {
                row_out[j] += a_scalar * row_b[j];
            }
        }
    }
}

/* -------------------------------------------------------------- AVX-512 --- */

__attribute__((target("avx512f,avx512vl")))
static void block_avx512(const float *restrict A, const float *restrict B,
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
            const __m512 a0 = _mm512_set1_ps(A[i * k + l]);
            const __m512 a1 = _mm512_set1_ps(A[(i + 1) * k + l]);
            const __m512 a2 = _mm512_set1_ps(A[(i + 2) * k + l]);
            const __m512 a3 = _mm512_set1_ps(A[(i + 3) * k + l]);
            const float *row_b = &B[l * n];

            size_t j = j_start;
            for (; j + 16 <= j_limit; j += 16) {
                const __m512 b = _mm512_loadu_ps(row_b + j);
                _mm512_storeu_ps(c0 + j,
                                 _mm512_fmadd_ps(a0, b, _mm512_loadu_ps(c0 + j)));
                _mm512_storeu_ps(c1 + j,
                                 _mm512_fmadd_ps(a1, b, _mm512_loadu_ps(c1 + j)));
                _mm512_storeu_ps(c2 + j,
                                 _mm512_fmadd_ps(a2, b, _mm512_loadu_ps(c2 + j)));
                _mm512_storeu_ps(c3 + j,
                                 _mm512_fmadd_ps(a3, b, _mm512_loadu_ps(c3 + j)));
            }
            /* Mask-predicated tail: AVX-512 can load and store a partial
             * vector directly, so the remainder runs the same instructions as
             * the body instead of dropping to scalar. This is why the kernel
             * asks for VL as well as F - the model's decode shapes are narrow
             * enough that the tail is a meaningful share of the work. */
            if (j < j_limit) {
                const __mmask16 tail = (__mmask16)((1u << (j_limit - j)) - 1u);
                const __m512 b = _mm512_maskz_loadu_ps(tail, row_b + j);
                _mm512_mask_storeu_ps(
                    c0 + j, tail,
                    _mm512_fmadd_ps(a0, b, _mm512_maskz_loadu_ps(tail, c0 + j)));
                _mm512_mask_storeu_ps(
                    c1 + j, tail,
                    _mm512_fmadd_ps(a1, b, _mm512_maskz_loadu_ps(tail, c1 + j)));
                _mm512_mask_storeu_ps(
                    c2 + j, tail,
                    _mm512_fmadd_ps(a2, b, _mm512_maskz_loadu_ps(tail, c2 + j)));
                _mm512_mask_storeu_ps(
                    c3 + j, tail,
                    _mm512_fmadd_ps(a3, b, _mm512_maskz_loadu_ps(tail, c3 + j)));
            }
        }
    }

    for (; i < i_limit; i++) {
        float *row_out = &C[i * n];
        for (size_t l = l_start; l < l_limit; l++) {
            const __m512 a = _mm512_set1_ps(A[i * k + l]);
            const float *row_b = &B[l * n];

            size_t j = j_start;
            for (; j + 16 <= j_limit; j += 16) {
                _mm512_storeu_ps(row_out + j,
                                 _mm512_fmadd_ps(a, _mm512_loadu_ps(row_b + j),
                                                 _mm512_loadu_ps(row_out + j)));
            }
            if (j < j_limit) {
                const __mmask16 tail = (__mmask16)((1u << (j_limit - j)) - 1u);
                _mm512_mask_storeu_ps(
                    row_out + j, tail,
                    _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(tail, row_b + j),
                                    _mm512_maskz_loadu_ps(tail, row_out + j)));
            }
        }
    }
}

/* ------------------------------------------------------------ dispatch --- */

/* The blocking loop, shared by both kernels. Identical in structure to
 * kernel_tiled_mr4()'s: blocks own disjoint regions of C, so the OpenMP
 * variant performs no cross-thread reduction and agrees bit-for-bit with a
 * serial build. */
static void run_blocked(x86_block_fn block, const float *restrict A,
                        const float *restrict B, float *restrict C,
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
                block(A, B, C, k, n, ii, i_limit, jj, j_limit, ll, l_limit);
            }
        }
    }
}

void matmul_kernel_avx2_mr4(const float *restrict A, const float *restrict B,
                            float *restrict C,
                            size_t m, size_t k, size_t n, size_t tile) {
    run_blocked(block_avx2, A, B, C, m, k, n, tile);
}

void matmul_kernel_avx512_mr4(const float *restrict A, const float *restrict B,
                              float *restrict C,
                              size_t m, size_t k, size_t n, size_t tile) {
    run_blocked(block_avx512, A, B, C, m, k, n, tile);
}

#else /* !DRANZER_HAVE_X86_SIMD */

/* This file is compiled on every architecture so the build has one source
 * list rather than a conditional one. ISO C requires a translation unit to
 * declare something, so on non-x86 targets it declares this and nothing else. */
typedef int dranzer_matmul_x86_not_supported;

#endif /* DRANZER_HAVE_X86_SIMD */
