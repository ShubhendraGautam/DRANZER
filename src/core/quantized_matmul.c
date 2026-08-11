/* INT8/INT4 weight storage and float-accumulating matmul. */

#include "core/quantized_matmul.h"
#include "core/cpu_features.h"
#include "core/matmul.h"
#include "core/parallel.h"
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#define QUANT_MATMUL_HAVE_X86_SIMD 1
#include <immintrin.h>
#endif

int quantized_weight_matrix_encode(quantized_weight_matrix_t *out,
                                   const float *values,
                                   size_t rows, size_t cols,
                                   int bits,
                                   quant_granularity_t granularity) {
    if (!out || !values || out->scales || out->packed || rows == 0 || cols == 0 ||
        rows > SIZE_MAX / cols || (bits != 4 && bits != 8)) return -1;

    size_t scale_count = 0, packed_size = 0;
    if (quantized_scale_count(rows, cols, granularity, &scale_count) != 0 ||
        quantized_packed_size(rows * cols, bits, &packed_size) != 0 ||
        scale_count > SIZE_MAX / sizeof(float)) return -1;

    float *scales = malloc(scale_count * sizeof(*scales));
    uint8_t *packed = malloc(packed_size);
    if (!scales || !packed ||
        quantize_pack(values, rows, cols, bits, granularity,
                      scales, scale_count, packed, packed_size) != 0) {
        free(scales);
        free(packed);
        return -1;
    }

    out->rows = rows;
    out->cols = cols;
    out->bits = bits;
    out->granularity = granularity;
    out->scales = scales;
    out->scale_count = scale_count;
    out->packed = packed;
    out->packed_size = packed_size;
    return 0;
}

void quantized_weight_matrix_free(quantized_weight_matrix_t *matrix) {
    if (!matrix) return;
    free(matrix->scales);
    free(matrix->packed);
    memset(matrix, 0, sizeof(*matrix));
}

static int matrix_valid(const quantized_weight_matrix_t *matrix, size_t k) {
    size_t scale_count = 0, packed_size = 0;
    if (!matrix || !matrix->scales || !matrix->packed || matrix->rows != k ||
        matrix->rows == 0 || matrix->cols == 0 ||
        matrix->rows > SIZE_MAX / matrix->cols ||
        (matrix->bits != 4 && matrix->bits != 8) ||
        quantized_scale_count(matrix->rows, matrix->cols, matrix->granularity,
                              &scale_count) != 0 ||
        quantized_packed_size(matrix->rows * matrix->cols, matrix->bits,
                              &packed_size) != 0) return 0;
    return matrix->scale_count == scale_count && matrix->packed_size == packed_size;
}

static int quantized_level(const quantized_weight_matrix_t *matrix,
                           size_t index) {
    const int qmax = matrix->bits == 8 ? 127 : 7;
    uint32_t code;
    if (matrix->bits == 8) {
        code = matrix->packed[index];
    } else {
        uint8_t byte = matrix->packed[index / 2];
        code = (index & 1u) ? (uint32_t)(byte >> 4)
                            : (uint32_t)(byte & UINT8_C(0x0f));
    }
    return (int)code - qmax;
}

static float quantized_scale(const quantized_weight_matrix_t *matrix,
                             size_t row, size_t col) {
    if (matrix->granularity == QUANT_GRANULARITY_ROW) return matrix->scales[row];
    if (matrix->granularity == QUANT_GRANULARITY_COLUMN) return matrix->scales[col];
    return matrix->scales[0];
}

static float quantized_value(const quantized_weight_matrix_t *matrix,
                             size_t row, size_t col) {
    size_t index = row * matrix->cols + col;
    return (float)quantized_level(matrix, index) *
           quantized_scale(matrix, row, col);
}

static void quantized_block_portable(
    const float *restrict A, const quantized_weight_matrix_t *weights,
    float *restrict C, size_t k, size_t n,
    size_t i_start, size_t i_limit, size_t j_start, size_t j_limit,
    size_t l_start, size_t l_limit) {
    for (size_t i = i_start; i < i_limit; i++) {
        const float *a_row = &A[i * k];
        float *c_row = &C[i * n];
        for (size_t j = j_start; j < j_limit; j++) {
            float sum = c_row[j];
            for (size_t l = l_start; l < l_limit; l++) {
                sum += a_row[l] * quantized_value(weights, l, j);
            }
            c_row[j] = sum;
        }
    }
}

#ifdef QUANT_MATMUL_HAVE_X86_SIMD

__attribute__((target("avx2,fma")))
static inline __m256 quantized_levels8_avx2(
    const quantized_weight_matrix_t *weights, size_t row, size_t col) {
    size_t index = row * weights->cols + col;
    __m256i levels;
    if (weights->bits == 8) {
        __m128i bytes = _mm_loadl_epi64(
            (const __m128i *)(const void *)&weights->packed[index]);
        levels = _mm256_cvtepu8_epi32(bytes);
        levels = _mm256_sub_epi32(levels, _mm256_set1_epi32(127));
    } else {
        int32_t decoded[8];
        for (size_t lane = 0; lane < 8; lane++) {
            decoded[lane] = quantized_level(weights, index + lane);
        }
        levels = _mm256_loadu_si256((const __m256i *)(const void *)decoded);
    }
    return _mm256_cvtepi32_ps(levels);
}

__attribute__((target("avx2,fma")))
static inline __m256 quantized_scales8_avx2(
    const quantized_weight_matrix_t *weights, size_t row, size_t col) {
    if (weights->granularity == QUANT_GRANULARITY_COLUMN) {
        return _mm256_loadu_ps(&weights->scales[col]);
    }
    float scale = weights->granularity == QUANT_GRANULARITY_ROW
                ? weights->scales[row] : weights->scales[0];
    return _mm256_set1_ps(scale);
}

__attribute__((target("avx2,fma")))
static void quantized_block_avx2(
    const float *restrict A, const quantized_weight_matrix_t *weights,
    float *restrict C, size_t k, size_t n,
    size_t i_start, size_t i_limit, size_t j_start, size_t j_limit,
    size_t l_start, size_t l_limit) {
    size_t i = i_start;
    for (; i + 4 <= i_limit; i += 4) {
        float *c0 = &C[i * n], *c1 = c0 + n, *c2 = c1 + n, *c3 = c2 + n;
        const float *a0 = &A[i * k], *a1 = a0 + k, *a2 = a1 + k, *a3 = a2 + k;
        size_t j = j_start;
        for (; j + 8 <= j_limit; j += 8) {
            __m256 x0 = _mm256_loadu_ps(c0 + j), x1 = _mm256_loadu_ps(c1 + j);
            __m256 x2 = _mm256_loadu_ps(c2 + j), x3 = _mm256_loadu_ps(c3 + j);
            for (size_t l = l_start; l < l_limit; l++) {
                __m256 b = _mm256_mul_ps(quantized_levels8_avx2(weights, l, j),
                                         quantized_scales8_avx2(weights, l, j));
                x0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[l]), b, x0);
                x1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[l]), b, x1);
                x2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[l]), b, x2);
                x3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[l]), b, x3);
            }
            _mm256_storeu_ps(c0 + j, x0); _mm256_storeu_ps(c1 + j, x1);
            _mm256_storeu_ps(c2 + j, x2); _mm256_storeu_ps(c3 + j, x3);
        }
        if (j < j_limit) {
            quantized_block_portable(A, weights, C, k, n, i, i + 4, j,
                                     j_limit, l_start, l_limit);
        }
    }
    if (i < i_limit) {
        quantized_block_portable(A, weights, C, k, n, i, i_limit, j_start,
                                 j_limit, l_start, l_limit);
    }
}

__attribute__((target("avx512f,avx512vl,avx512bw")))
static inline __m512 quantized_levels16_avx512(
    const quantized_weight_matrix_t *weights, size_t row, size_t col) {
    size_t index = row * weights->cols + col;
    __m512i levels;
    if (weights->bits == 8) {
        __m128i bytes = _mm_loadu_si128(
            (const __m128i *)(const void *)&weights->packed[index]);
        levels = _mm512_cvtepu8_epi32(bytes);
        levels = _mm512_sub_epi32(levels, _mm512_set1_epi32(127));
    } else {
        int32_t decoded[16];
        for (size_t lane = 0; lane < 16; lane++) {
            decoded[lane] = quantized_level(weights, index + lane);
        }
        levels = _mm512_loadu_si512((const void *)decoded);
    }
    return _mm512_cvtepi32_ps(levels);
}

__attribute__((target("avx512f,avx512vl,avx512bw")))
static inline __m512 quantized_scales16_avx512(
    const quantized_weight_matrix_t *weights, size_t row, size_t col) {
    if (weights->granularity == QUANT_GRANULARITY_COLUMN) {
        return _mm512_loadu_ps(&weights->scales[col]);
    }
    float scale = weights->granularity == QUANT_GRANULARITY_ROW
                ? weights->scales[row] : weights->scales[0];
    return _mm512_set1_ps(scale);
}

__attribute__((target("avx512f,avx512vl,avx512bw")))
static void quantized_block_avx512(
    const float *restrict A, const quantized_weight_matrix_t *weights,
    float *restrict C, size_t k, size_t n,
    size_t i_start, size_t i_limit, size_t j_start, size_t j_limit,
    size_t l_start, size_t l_limit) {
    size_t i = i_start;
    for (; i + 4 <= i_limit; i += 4) {
        float *c0 = &C[i * n], *c1 = c0 + n, *c2 = c1 + n, *c3 = c2 + n;
        const float *a0 = &A[i * k], *a1 = a0 + k, *a2 = a1 + k, *a3 = a2 + k;
        size_t j = j_start;
        for (; j + 16 <= j_limit; j += 16) {
            __m512 x0 = _mm512_loadu_ps(c0 + j), x1 = _mm512_loadu_ps(c1 + j);
            __m512 x2 = _mm512_loadu_ps(c2 + j), x3 = _mm512_loadu_ps(c3 + j);
            for (size_t l = l_start; l < l_limit; l++) {
                __m512 b = _mm512_mul_ps(quantized_levels16_avx512(weights, l, j),
                                         quantized_scales16_avx512(weights, l, j));
                x0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[l]), b, x0);
                x1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[l]), b, x1);
                x2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[l]), b, x2);
                x3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[l]), b, x3);
            }
            _mm512_storeu_ps(c0 + j, x0); _mm512_storeu_ps(c1 + j, x1);
            _mm512_storeu_ps(c2 + j, x2); _mm512_storeu_ps(c3 + j, x3);
        }
        if (j < j_limit) {
            quantized_block_portable(A, weights, C, k, n, i, i + 4, j,
                                     j_limit, l_start, l_limit);
        }
    }
    if (i < i_limit) {
        quantized_block_portable(A, weights, C, k, n, i, i_limit, j_start,
                                 j_limit, l_start, l_limit);
    }
}

#endif /* QUANT_MATMUL_HAVE_X86_SIMD */

int matmul_quantized_weight(const float *restrict A,
                            const quantized_weight_matrix_t *weights,
                            float *restrict C,
                            size_t m, size_t k, size_t tile) {
    if (!A || !C || m == 0 || !matrix_valid(weights, k) ||
        weights->cols > SIZE_MAX / m ||
        m * weights->cols > SIZE_MAX / sizeof(*C) ||
        k > SIZE_MAX / m || m * k > SIZE_MAX / weights->cols) return -1;
    const size_t n = weights->cols;
    if (tile == 0) tile = matmul_tile_size();
    if (tile == 0) tile = 256;
    memset(C, 0, m * n * sizeof(*C));

    const size_t i_blocks = (m - 1) / tile + 1;
    const size_t j_blocks = (n - 1) / tile + 1;
    const size_t work = m * k * n;
    DRANZER_PARALLEL_FOR(i_blocks * j_blocks, work, blk,
        size_t ii = (blk / j_blocks) * tile;
        size_t jj = (blk % j_blocks) * tile;
        size_t i_limit = tile > m - ii ? m : ii + tile;
        size_t j_limit = tile > n - jj ? n : jj + tile;
        for (size_t ll = 0; ll < k; ) {
            size_t l_limit = tile > k - ll ? k : ll + tile;
#ifdef QUANT_MATMUL_HAVE_X86_SIMD
            if (cpu_isa_available(CPU_ISA_AVX512)) {
                quantized_block_avx512(A, weights, C, k, n, ii, i_limit,
                                       jj, j_limit, ll, l_limit);
            } else if (cpu_isa_available(CPU_ISA_AVX2)) {
                quantized_block_avx2(A, weights, C, k, n, ii, i_limit,
                                     jj, j_limit, ll, l_limit);
            } else {
                quantized_block_portable(A, weights, C, k, n, ii, i_limit,
                                         jj, j_limit, ll, l_limit);
            }
#else
            quantized_block_portable(A, weights, C, k, n, ii, i_limit,
                                     jj, j_limit, ll, l_limit);
#endif
            ll = l_limit;
        }
    );
    return 0;
}
