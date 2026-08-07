/*
 * bfloat16 conversion and the bf16-weight matmul. See core/bf16.h for why
 * this exists and what question it is meant to answer.
 */

#include "core/bf16.h"
#include "core/matmul.h"
#include "core/cpu_features.h"
#include "core/parallel.h"
#include <string.h>

#if defined(__x86_64__)
#define BF16_HAVE_X86_SIMD 1
#include <immintrin.h>
#endif

bf16_t bf16_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    /* NaN: keep it a NaN. Truncating the low 16 bits of a NaN whose payload
     * lives entirely down there produces an exponent-all-ones, mantissa-zero
     * pattern, which is an infinity. Force a payload bit so it cannot. */
    if (((bits >> 23) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) != 0u) {
        return (bf16_t)((bits >> 16) | 0x0040u);
    }

    /* Round to nearest, ties to even: add half an LSB of the target, plus one
     * more if the retained bit is already odd. */
    const uint32_t lsb = (bits >> 16) & 1u;
    return (bf16_t)((bits + 0x7FFFu + lsb) >> 16);
}

float bf16_to_f32(bf16_t value) {
    const uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

void bf16_encode_array(const float *restrict src, bf16_t *restrict dst, size_t count) {
    for (size_t i = 0; i < count; i++) dst[i] = bf16_from_f32(src[i]);
}

void bf16_decode_array(const bf16_t *restrict src, float *restrict dst, size_t count) {
    for (size_t i = 0; i < count; i++) dst[i] = bf16_to_f32(src[i]);
}

/* ------------------------------------------------------------- kernels ---
 *
 * These mirror core/matmul_x86.c's block_avx2/block_avx512 and their
 * run_blocked() driver *exactly*: the same three-level (i, j, l) tiling, the
 * same four-row register blocking, the same accumulate-into-C across l
 * blocks, the same masked or scalar tails, and the same i<4 remainder loop.
 * The single difference is that B's row is widened from bf16 as it is loaded.
 *
 * That is not stylistic. The first version of this file tiled only (i, j) and
 * ran the full k inside each block, and it measured 0.34x against fp32 at
 * 128x2048x2048 - a result that looked like "bf16 loses badly when B leaves
 * cache" and was in fact "a two-level kernel loses to a three-level one". The
 * benchmark must vary the format and nothing else, or it measures the author.
 */

/* Widen and accumulate one l-range into four register-resident C rows. */
static void bf16_block_portable(const float *restrict A, const bf16_t *restrict B,
                                float *restrict C, size_t k, size_t n,
                                size_t i_start, size_t i_limit,
                                size_t j_start, size_t j_limit,
                                size_t l_start, size_t l_limit) {
    for (size_t i = i_start; i < i_limit; i++) {
        float *row_out = &C[i * n];
        const float *a_row = &A[i * k];
        for (size_t j = j_start; j < j_limit; j++) {
            float sum = row_out[j];
            for (size_t l = l_start; l < l_limit; l++) {
                sum += a_row[l] * bf16_to_f32(B[l * n + j]);
            }
            row_out[j] = sum;
        }
    }
}

#ifdef BF16_HAVE_X86_SIMD

/* Widen eight bf16 to eight floats: zero-extend to 32 bits, shift into the
 * high half. Three instructions, no multiply and no table - the whole reason
 * bf16 is the right format to ask the bandwidth question with. */
__attribute__((target("avx2,fma")))
static inline __m256 bf16_widen8(const bf16_t *p) {
    const __m128i narrow = _mm_loadu_si128((const __m128i *)(const void *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(narrow), 16));
}

__attribute__((target("avx2,fma")))
static void bf16_block_avx2(const float *restrict A, const bf16_t *restrict B,
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
        const float *a0r = &A[i * k];
        const float *a1r = a0r + k;
        const float *a2r = a1r + k;
        const float *a3r = a2r + k;

        size_t j = j_start;
        for (; j + 8 <= j_limit; j += 8) {
            __m256 acc0 = _mm256_loadu_ps(c0 + j);
            __m256 acc1 = _mm256_loadu_ps(c1 + j);
            __m256 acc2 = _mm256_loadu_ps(c2 + j);
            __m256 acc3 = _mm256_loadu_ps(c3 + j);
            for (size_t l = l_start; l < l_limit; l++) {
                const __m256 b = bf16_widen8(&B[l * n + j]);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0r[l]), b, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1r[l]), b, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2r[l]), b, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3r[l]), b, acc3);
            }
            _mm256_storeu_ps(c0 + j, acc0);
            _mm256_storeu_ps(c1 + j, acc1);
            _mm256_storeu_ps(c2 + j, acc2);
            _mm256_storeu_ps(c3 + j, acc3);
        }
        /* Scalar tail: AVX2 has no mask-predicated load or store. */
        for (; j < j_limit; j++) {
            float s0 = c0[j], s1 = c1[j], s2 = c2[j], s3 = c3[j];
            for (size_t l = l_start; l < l_limit; l++) {
                const float b = bf16_to_f32(B[l * n + j]);
                s0 += a0r[l] * b; s1 += a1r[l] * b;
                s2 += a2r[l] * b; s3 += a3r[l] * b;
            }
            c0[j] = s0; c1[j] = s1; c2[j] = s2; c3[j] = s3;
        }
    }

    for (; i < i_limit; i++) {
        float *row_out = &C[i * n];
        const float *a_row = &A[i * k];
        size_t j = j_start;
        for (; j + 8 <= j_limit; j += 8) {
            __m256 acc = _mm256_loadu_ps(row_out + j);
            for (size_t l = l_start; l < l_limit; l++) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[l]),
                                      bf16_widen8(&B[l * n + j]), acc);
            }
            _mm256_storeu_ps(row_out + j, acc);
        }
        for (; j < j_limit; j++) {
            float sum = row_out[j];
            for (size_t l = l_start; l < l_limit; l++) {
                sum += a_row[l] * bf16_to_f32(B[l * n + j]);
            }
            row_out[j] = sum;
        }
    }
}

__attribute__((target("avx512f,avx512vl,avx512bw")))
static inline __m512 bf16_widen16(const bf16_t *p) {
    const __m256i narrow = _mm256_loadu_si256((const __m256i *)(const void *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(narrow), 16));
}

__attribute__((target("avx512f,avx512vl,avx512bw")))
static inline __m512 bf16_widen16_masked(const bf16_t *p, __mmask16 tail) {
    const __m256i narrow = _mm256_maskz_loadu_epi16(tail, (const void *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(narrow), 16));
}

__attribute__((target("avx512f,avx512vl,avx512bw")))
static void bf16_block_avx512(const float *restrict A, const bf16_t *restrict B,
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
        const float *a0r = &A[i * k];
        const float *a1r = a0r + k;
        const float *a2r = a1r + k;
        const float *a3r = a2r + k;

        size_t j = j_start;
        for (; j + 16 <= j_limit; j += 16) {
            __m512 acc0 = _mm512_loadu_ps(c0 + j);
            __m512 acc1 = _mm512_loadu_ps(c1 + j);
            __m512 acc2 = _mm512_loadu_ps(c2 + j);
            __m512 acc3 = _mm512_loadu_ps(c3 + j);
            for (size_t l = l_start; l < l_limit; l++) {
                const __m512 b = bf16_widen16(&B[l * n + j]);
                acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0r[l]), b, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1r[l]), b, acc1);
                acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2r[l]), b, acc2);
                acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3r[l]), b, acc3);
            }
            _mm512_storeu_ps(c0 + j, acc0);
            _mm512_storeu_ps(c1 + j, acc1);
            _mm512_storeu_ps(c2 + j, acc2);
            _mm512_storeu_ps(c3 + j, acc3);
        }
        if (j < j_limit) {
            const __mmask16 tail = (__mmask16)((1u << (j_limit - j)) - 1u);
            __m512 acc0 = _mm512_maskz_loadu_ps(tail, c0 + j);
            __m512 acc1 = _mm512_maskz_loadu_ps(tail, c1 + j);
            __m512 acc2 = _mm512_maskz_loadu_ps(tail, c2 + j);
            __m512 acc3 = _mm512_maskz_loadu_ps(tail, c3 + j);
            for (size_t l = l_start; l < l_limit; l++) {
                const __m512 b = bf16_widen16_masked(&B[l * n + j], tail);
                acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0r[l]), b, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1r[l]), b, acc1);
                acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2r[l]), b, acc2);
                acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3r[l]), b, acc3);
            }
            _mm512_mask_storeu_ps(c0 + j, tail, acc0);
            _mm512_mask_storeu_ps(c1 + j, tail, acc1);
            _mm512_mask_storeu_ps(c2 + j, tail, acc2);
            _mm512_mask_storeu_ps(c3 + j, tail, acc3);
        }
    }

    /* Fewer than four rows left. Without this the narrow decode shapes (m=1)
     * would fall to the portable body and measure scalar code against a
     * vectorized fp32 kernel - which is exactly what the first version of
     * this file did, reading 0.13x on a 1x64x1000 head. */
    for (; i < i_limit; i++) {
        float *row_out = &C[i * n];
        const float *a_row = &A[i * k];
        size_t j = j_start;
        for (; j + 16 <= j_limit; j += 16) {
            __m512 acc = _mm512_loadu_ps(row_out + j);
            for (size_t l = l_start; l < l_limit; l++) {
                acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[l]),
                                      bf16_widen16(&B[l * n + j]), acc);
            }
            _mm512_storeu_ps(row_out + j, acc);
        }
        if (j < j_limit) {
            const __mmask16 tail = (__mmask16)((1u << (j_limit - j)) - 1u);
            __m512 acc = _mm512_maskz_loadu_ps(tail, row_out + j);
            for (size_t l = l_start; l < l_limit; l++) {
                acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[l]),
                                      bf16_widen16_masked(&B[l * n + j], tail), acc);
            }
            _mm512_mask_storeu_ps(row_out + j, tail, acc);
        }
    }
}

#endif /* BF16_HAVE_X86_SIMD */

void matmul_bf16_weight(const float *restrict A, const bf16_t *restrict B,
                        float *restrict C, size_t m, size_t k, size_t n,
                        size_t tile) {
    if (m == 0 || k == 0 || n == 0) return;
    if (tile == 0) tile = matmul_tile_size();
    if (tile == 0) tile = 256;

    memset(C, 0, m * n * sizeof(float));

    const size_t i_blocks = (m + tile - 1) / tile;
    const size_t j_blocks = (n + tile - 1) / tile;

    /* The (ii, jj) nest flattened into one block index, and the ll loop
     * inside it - identical to run_blocked() in core/matmul_x86.c. Blocks own
     * disjoint regions of C, so a forked run is bit-identical to a serial
     * one. */
    DRANZER_PARALLEL_FOR(i_blocks * j_blocks, m * k * n, blk,
        size_t ii = (blk / j_blocks) * tile;
        size_t jj = (blk % j_blocks) * tile;
        size_t i_limit = (ii + tile > m) ? m : ii + tile;
        size_t j_limit = (jj + tile > n) ? n : jj + tile;
        for (size_t ll = 0; ll < k; ll += tile) {
            size_t l_limit = (ll + tile > k) ? k : ll + tile;
#ifdef BF16_HAVE_X86_SIMD
            if (cpu_isa_available(CPU_ISA_AVX512)) {
                bf16_block_avx512(A, B, C, k, n, ii, i_limit, jj, j_limit, ll, l_limit);
            } else if (cpu_isa_available(CPU_ISA_AVX2)) {
                bf16_block_avx2(A, B, C, k, n, ii, i_limit, jj, j_limit, ll, l_limit);
            } else {
                bf16_block_portable(A, B, C, k, n, ii, i_limit, jj, j_limit, ll, l_limit);
            }
#else
            bf16_block_portable(A, B, C, k, n, ii, i_limit, jj, j_limit, ll, l_limit);
#endif
        }
    );
}
