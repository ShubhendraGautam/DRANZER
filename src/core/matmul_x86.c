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
#include "core/parallel.h"

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

/* Loop order j/l with the accumulators held in registers, not l/j with them
 * held in memory.
 *
 * The obvious arrangement - j innermost, reading and writing C every
 * iteration - was measured at 1.47x over the portable kernel under GCC and
 * 0.92x under Clang, from identical source. Disassembling both explained it:
 * `fmadd(a, b, load(c))` puts the loaded value in the addend position, which
 * x86 can encode as a memory operand. GCC folded all four loads into their
 * FMAs; Clang folded one of four and emitted three extra `vmovups` per
 * iteration. Same instructions, different count, and the gap tracked it.
 *
 * Rather than coax one compiler into an encoding, this removes the load
 * entirely: for each 8-column strip, C is read once into four accumulator
 * registers, the whole k-range is accumulated into them, and they are written
 * back once. The inner loop then has no C traffic at all, so there is nothing
 * left to fold and both compilers generate the same shape of code.
 *
 * Nine ymm registers are live (four accumulators, four broadcasts, one B) out
 * of sixteen, so this does not spill. B is now walked with stride n across l
 * instead of contiguously, but a tile of it stays resident - that is what the
 * blocking in run_blocked() is for.
 *
 * Per-element accumulation order is unchanged: still l ascending. */
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
                const __m256 b = _mm256_loadu_ps(&B[l * n + j]);
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

        /* Columns past the last full vector. AVX2 has no store mask for
         * floats (maskstore exists but costs more than this loop on the
         * handful of columns a tail can hold), so the remainder is plain
         * scalar - the same arithmetic, one column at a time, and likewise
         * accumulated in registers across l. */
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

    /* Fewer than four rows left. Single-token decode shapes (m == 1) land
     * here for every block, so this path is vectorized too, and accumulates
     * in a register across l for the same reason as above. */
    for (; i < i_limit; i++) {
        float *row_out = &C[i * n];
        const float *a_row = &A[i * k];

        size_t j = j_start;
        for (; j + 8 <= j_limit; j += 8) {
            __m256 acc = _mm256_loadu_ps(row_out + j);
            for (size_t l = l_start; l < l_limit; l++) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[l]),
                                      _mm256_loadu_ps(&B[l * n + j]), acc);
            }
            _mm256_storeu_ps(row_out + j, acc);
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

/* -------------------------------------------------------------- AVX-512 --- */

__attribute__((target("avx512f,avx512vl")))
static void block_avx512(const float *restrict A, const float *restrict B,
                         float *restrict C, size_t k, size_t n,
                         size_t i_start, size_t i_limit,
                         size_t j_start, size_t j_limit,
                         size_t l_start, size_t l_limit) {
    size_t i = i_start;

    /* Same j/l order and register-resident accumulators as block_avx2 above,
     * and for the same measured reason: with the accumulators held in memory
     * this kernel was slower than the 8-wide one on five of six shapes under
     * both compilers, despite twice the width. Width does not help a loop
     * whose limit is C traffic. */
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
                const __m512 b = _mm512_loadu_ps(&B[l * n + j]);
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

        /* Mask-predicated tail: AVX-512 can load and store a partial vector
         * directly, so the remainder runs the same instructions as the body
         * instead of dropping to scalar. This is why the kernel asks for VL as
         * well as F - the model's decode shapes are narrow enough that the
         * tail is a meaningful share of the work. */
        if (j < j_limit) {
            const __mmask16 tail = (__mmask16)((1u << (j_limit - j)) - 1u);
            __m512 acc0 = _mm512_maskz_loadu_ps(tail, c0 + j);
            __m512 acc1 = _mm512_maskz_loadu_ps(tail, c1 + j);
            __m512 acc2 = _mm512_maskz_loadu_ps(tail, c2 + j);
            __m512 acc3 = _mm512_maskz_loadu_ps(tail, c3 + j);

            for (size_t l = l_start; l < l_limit; l++) {
                const __m512 b = _mm512_maskz_loadu_ps(tail, &B[l * n + j]);
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

    for (; i < i_limit; i++) {
        float *row_out = &C[i * n];
        const float *a_row = &A[i * k];

        size_t j = j_start;
        for (; j + 16 <= j_limit; j += 16) {
            __m512 acc = _mm512_loadu_ps(row_out + j);
            for (size_t l = l_start; l < l_limit; l++) {
                acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[l]),
                                      _mm512_loadu_ps(&B[l * n + j]), acc);
            }
            _mm512_storeu_ps(row_out + j, acc);
        }
        if (j < j_limit) {
            const __mmask16 tail = (__mmask16)((1u << (j_limit - j)) - 1u);
            __m512 acc = _mm512_maskz_loadu_ps(tail, row_out + j);
            for (size_t l = l_start; l < l_limit; l++) {
                acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[l]),
                                      _mm512_maskz_loadu_ps(tail, &B[l * n + j]), acc);
            }
            _mm512_mask_storeu_ps(row_out + j, tail, acc);
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

    /* The (ii, jj) nest flattened into one block index, matching
     * kernel_tiled_mr4() in core/matmul.c - `collapse(2)` with a static
     * schedule distributes this same space in this same order, and one loop is
     * what lets DRANZER_PARALLEL_FOR skip the region without a second copy of
     * the body. */
    const size_t i_blocks = (m + tile - 1) / tile;
    const size_t j_blocks = (n + tile - 1) / tile;

    DRANZER_PARALLEL_FOR(i_blocks * j_blocks, m * k * n, blk,
        size_t ii = (blk / j_blocks) * tile;
        size_t jj = (blk % j_blocks) * tile;
        size_t i_limit = (ii + tile > m) ? m : ii + tile;
        size_t j_limit = (jj + tile > n) ? n : jj + tile;
        for (size_t ll = 0; ll < k; ll += tile) {
            size_t l_limit = (ll + tile > k) ? k : ll + tile;
            block(A, B, C, k, n, ii, i_limit, jj, j_limit, ll, l_limit);
        }
    );
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

/* ------------------------------------------------------- backward pass ---
 *
 * Both mirror the portable versions in core/matmul.c exactly - same loop
 * order, same blocking - with the innermost loop over j issued as vector
 * FMAs. Each has an AVX2 and an AVX-512 form; see core/matmul_simd.h.
 */

/* Horizontal sum of eight lanes. AVX-512 has _mm512_reduce_add_ps for this;
 * AVX2 does not, so the input kernel's per-lane accumulators are folded by
 * hand. Halve to 128 bits, then two shuffle-add steps. */
__attribute__((target("avx2,fma")))
static inline float hsum256(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
}

/* dB rows [l, l+4) += A_transposed @ dC, over all of m. AVX2 form of
 * bw_weight_block_avx512() below: same four-accumulator shape, eight columns
 * per vector instead of sixteen, and a scalar tail because AVX2 has no
 * mask-predicated load/store. */
__attribute__((target("avx2,fma")))
static void bw_weight_block_avx2(const float *restrict A, const float *restrict dC,
                                 float *restrict dB, size_t m, size_t k, size_t n,
                                 size_t l) {
    float *out0 = &dB[l * n];
    float *out1 = out0 + n;
    float *out2 = out1 + n;
    float *out3 = out2 + n;

    for (size_t i = 0; i < m; i++) {
        const float *row_dc = &dC[i * n];
        const float *a = &A[i * k + l];
        const __m256 a0 = _mm256_set1_ps(a[0]);
        const __m256 a1 = _mm256_set1_ps(a[1]);
        const __m256 a2 = _mm256_set1_ps(a[2]);
        const __m256 a3 = _mm256_set1_ps(a[3]);

        size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 d = _mm256_loadu_ps(row_dc + j);
            _mm256_storeu_ps(out0 + j, _mm256_fmadd_ps(a0, d, _mm256_loadu_ps(out0 + j)));
            _mm256_storeu_ps(out1 + j, _mm256_fmadd_ps(a1, d, _mm256_loadu_ps(out1 + j)));
            _mm256_storeu_ps(out2 + j, _mm256_fmadd_ps(a2, d, _mm256_loadu_ps(out2 + j)));
            _mm256_storeu_ps(out3 + j, _mm256_fmadd_ps(a3, d, _mm256_loadu_ps(out3 + j)));
        }
        for (; j < n; j++) {
            const float d = row_dc[j];
            out0[j] += a[0] * d;
            out1[j] += a[1] * d;
            out2[j] += a[2] * d;
            out3[j] += a[3] * d;
        }
    }
}

/* One row of dB, for values of k that are not a multiple of four. */
__attribute__((target("avx2,fma")))
static void bw_weight_row_avx2(const float *restrict A, const float *restrict dC,
                               float *restrict dB, size_t m, size_t k, size_t n,
                               size_t l) {
    float *row_out = &dB[l * n];
    for (size_t i = 0; i < m; i++) {
        const float scalar = A[i * k + l];
        const __m256 a = _mm256_set1_ps(scalar);
        const float *row_dc = &dC[i * n];
        size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            _mm256_storeu_ps(row_out + j,
                _mm256_fmadd_ps(a, _mm256_loadu_ps(row_dc + j),
                                _mm256_loadu_ps(row_out + j)));
        }
        for (; j < n; j++) row_out[j] += scalar * row_dc[j];
    }
}

void matmul_backward_weight_avx2(const float *restrict A, const float *restrict dC,
                                 float *restrict dB, size_t m, size_t k, size_t n) {
    const size_t blocks = k / 4;

    /* Same disjoint-rows-per-block argument as the AVX-512 form: no
     * cross-thread reduction, so a forked result is bit-identical to serial. */
    DRANZER_PARALLEL_FOR(blocks, m * k * n, b,
        bw_weight_block_avx2(A, dC, dB, m, k, n, b * 4);
    );
    for (size_t l = blocks * 4; l < k; l++) {
        bw_weight_row_avx2(A, dC, dB, m, k, n, l);
    }
}

/* dA row i += dC row i @ B_transposed. AVX2 form of bw_input_row_avx512():
 * four rows of B consumed at once so one load of dC feeds four independent
 * chains, each folded across its lanes at the end. */
__attribute__((target("avx2,fma")))
static void bw_input_row_avx2(const float *restrict dC, const float *restrict B,
                              float *restrict dA, size_t k, size_t n, size_t i) {
    const float *row_dc = &dC[i * n];
    float *out = &dA[i * k];
    size_t l = 0;

    for (; l + 4 <= k; l += 4) {
        const float *b0 = &B[l * n];
        const float *b1 = b0 + n;
        const float *b2 = b1 + n;
        const float *b3 = b2 + n;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 d = _mm256_loadu_ps(row_dc + j);
            acc0 = _mm256_fmadd_ps(d, _mm256_loadu_ps(b0 + j), acc0);
            acc1 = _mm256_fmadd_ps(d, _mm256_loadu_ps(b1 + j), acc1);
            acc2 = _mm256_fmadd_ps(d, _mm256_loadu_ps(b2 + j), acc2);
            acc3 = _mm256_fmadd_ps(d, _mm256_loadu_ps(b3 + j), acc3);
        }

        float s0 = hsum256(acc0), s1 = hsum256(acc1);
        float s2 = hsum256(acc2), s3 = hsum256(acc3);
        for (; j < n; j++) {
            const float d = row_dc[j];
            s0 += d * b0[j];
            s1 += d * b1[j];
            s2 += d * b2[j];
            s3 += d * b3[j];
        }

        out[l]     += s0;
        out[l + 1] += s1;
        out[l + 2] += s2;
        out[l + 3] += s3;
    }

    for (; l < k; l++) {
        const float *b = &B[l * n];
        __m256 acc = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row_dc + j),
                                  _mm256_loadu_ps(b + j), acc);
        }
        float s = hsum256(acc);
        for (; j < n; j++) s += row_dc[j] * b[j];
        out[l] += s;
    }
}

void matmul_backward_input_avx2(const float *restrict dC, const float *restrict B,
                                float *restrict dA, size_t m, size_t k, size_t n) {
    /* Parallel over i: each iteration owns a disjoint row of dA. */
    DRANZER_PARALLEL_FOR(m, m * k * n, i,
        bw_input_row_avx2(dC, B, dA, k, n, i);
    );
}

/* dB rows [l, l+4) += A_transposed @ dC, over all of m.
 *
 * The same shape as block_avx512() above: four independent accumulator rows
 * fed by one broadcast scalar each, against a single shared load of dC. The
 * difference is that the four scalars come from four adjacent columns of A
 * (a[0..3], contiguous) rather than four rows. */
__attribute__((target("avx512f,avx512vl")))
static void bw_weight_block_avx512(const float *restrict A, const float *restrict dC,
                                   float *restrict dB, size_t m, size_t k, size_t n,
                                   size_t l) {
    float *out0 = &dB[l * n];
    float *out1 = out0 + n;
    float *out2 = out1 + n;
    float *out3 = out2 + n;

    for (size_t i = 0; i < m; i++) {
        const float *row_dc = &dC[i * n];
        const float *a = &A[i * k + l];
        const __m512 a0 = _mm512_set1_ps(a[0]);
        const __m512 a1 = _mm512_set1_ps(a[1]);
        const __m512 a2 = _mm512_set1_ps(a[2]);
        const __m512 a3 = _mm512_set1_ps(a[3]);

        size_t j = 0;
        for (; j + 16 <= n; j += 16) {
            const __m512 d = _mm512_loadu_ps(row_dc + j);
            _mm512_storeu_ps(out0 + j, _mm512_fmadd_ps(a0, d, _mm512_loadu_ps(out0 + j)));
            _mm512_storeu_ps(out1 + j, _mm512_fmadd_ps(a1, d, _mm512_loadu_ps(out1 + j)));
            _mm512_storeu_ps(out2 + j, _mm512_fmadd_ps(a2, d, _mm512_loadu_ps(out2 + j)));
            _mm512_storeu_ps(out3 + j, _mm512_fmadd_ps(a3, d, _mm512_loadu_ps(out3 + j)));
        }
        if (j < n) {
            const __mmask16 tail = (__mmask16)((1u << (n - j)) - 1u);
            const __m512 d = _mm512_maskz_loadu_ps(tail, row_dc + j);
            _mm512_mask_storeu_ps(out0 + j, tail,
                _mm512_fmadd_ps(a0, d, _mm512_maskz_loadu_ps(tail, out0 + j)));
            _mm512_mask_storeu_ps(out1 + j, tail,
                _mm512_fmadd_ps(a1, d, _mm512_maskz_loadu_ps(tail, out1 + j)));
            _mm512_mask_storeu_ps(out2 + j, tail,
                _mm512_fmadd_ps(a2, d, _mm512_maskz_loadu_ps(tail, out2 + j)));
            _mm512_mask_storeu_ps(out3 + j, tail,
                _mm512_fmadd_ps(a3, d, _mm512_maskz_loadu_ps(tail, out3 + j)));
        }
    }
}

/* One row of dB, for values of k that are not a multiple of four. */
__attribute__((target("avx512f,avx512vl")))
static void bw_weight_row_avx512(const float *restrict A, const float *restrict dC,
                                 float *restrict dB, size_t m, size_t k, size_t n,
                                 size_t l) {
    float *row_out = &dB[l * n];
    for (size_t i = 0; i < m; i++) {
        const __m512 a = _mm512_set1_ps(A[i * k + l]);
        const float *row_dc = &dC[i * n];
        size_t j = 0;
        for (; j + 16 <= n; j += 16) {
            _mm512_storeu_ps(row_out + j,
                _mm512_fmadd_ps(a, _mm512_loadu_ps(row_dc + j),
                                _mm512_loadu_ps(row_out + j)));
        }
        if (j < n) {
            const __mmask16 tail = (__mmask16)((1u << (n - j)) - 1u);
            _mm512_mask_storeu_ps(row_out + j, tail,
                _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(tail, row_dc + j),
                                _mm512_maskz_loadu_ps(tail, row_out + j)));
        }
    }
}

void matmul_backward_weight_avx512(const float *restrict A, const float *restrict dC,
                                   float *restrict dB, size_t m, size_t k, size_t n) {
    const size_t blocks = k / 4;

    /* Canonical OpenMP loop form: a plain counter with a simple bound. Each
     * block owns four disjoint rows of dB, so there is no cross-thread
     * reduction and results match a serial build bit for bit. */
    DRANZER_PARALLEL_FOR(blocks, m * k * n, b,
        bw_weight_block_avx512(A, dC, dB, m, k, n, b * 4);
    );
    for (size_t l = blocks * 4; l < k; l++) {
        bw_weight_row_avx512(A, dC, dB, m, k, n, l);
    }
}

/* dA row i += dC row i @ B_transposed.
 *
 * A reduction rather than an axpy: each output element is a dot product of
 * two contiguous rows. Four rows of B are consumed at once so the single load
 * of dC feeds four independent accumulator chains, then each is reduced
 * across its lanes at the end. */
__attribute__((target("avx512f,avx512vl")))
static void bw_input_row_avx512(const float *restrict dC, const float *restrict B,
                                float *restrict dA, size_t k, size_t n, size_t i) {
    const float *row_dc = &dC[i * n];
    float *out = &dA[i * k];
    size_t l = 0;

    for (; l + 4 <= k; l += 4) {
        const float *b0 = &B[l * n];
        const float *b1 = b0 + n;
        const float *b2 = b1 + n;
        const float *b3 = b2 + n;
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        size_t j = 0;
        for (; j + 16 <= n; j += 16) {
            const __m512 d = _mm512_loadu_ps(row_dc + j);
            acc0 = _mm512_fmadd_ps(d, _mm512_loadu_ps(b0 + j), acc0);
            acc1 = _mm512_fmadd_ps(d, _mm512_loadu_ps(b1 + j), acc1);
            acc2 = _mm512_fmadd_ps(d, _mm512_loadu_ps(b2 + j), acc2);
            acc3 = _mm512_fmadd_ps(d, _mm512_loadu_ps(b3 + j), acc3);
        }
        if (j < n) {
            const __mmask16 tail = (__mmask16)((1u << (n - j)) - 1u);
            const __m512 d = _mm512_maskz_loadu_ps(tail, row_dc + j);
            acc0 = _mm512_fmadd_ps(d, _mm512_maskz_loadu_ps(tail, b0 + j), acc0);
            acc1 = _mm512_fmadd_ps(d, _mm512_maskz_loadu_ps(tail, b1 + j), acc1);
            acc2 = _mm512_fmadd_ps(d, _mm512_maskz_loadu_ps(tail, b2 + j), acc2);
            acc3 = _mm512_fmadd_ps(d, _mm512_maskz_loadu_ps(tail, b3 + j), acc3);
        }

        out[l]     += _mm512_reduce_add_ps(acc0);
        out[l + 1] += _mm512_reduce_add_ps(acc1);
        out[l + 2] += _mm512_reduce_add_ps(acc2);
        out[l + 3] += _mm512_reduce_add_ps(acc3);
    }

    for (; l < k; l++) {
        const float *b = &B[l * n];
        __m512 acc = _mm512_setzero_ps();
        size_t j = 0;
        for (; j + 16 <= n; j += 16) {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(row_dc + j),
                                  _mm512_loadu_ps(b + j), acc);
        }
        if (j < n) {
            const __mmask16 tail = (__mmask16)((1u << (n - j)) - 1u);
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, row_dc + j),
                                  _mm512_maskz_loadu_ps(tail, b + j), acc);
        }
        out[l] += _mm512_reduce_add_ps(acc);
    }
}

void matmul_backward_input_avx512(const float *restrict dC, const float *restrict B,
                                  float *restrict dA, size_t m, size_t k, size_t n) {
    /* Parallel over i: each iteration owns a disjoint row of dA. */
    DRANZER_PARALLEL_FOR(m, m * k * n, i,
        bw_input_row_avx512(dC, B, dA, k, n, i);
    );
}

#else /* !DRANZER_HAVE_X86_SIMD */

/* This file is compiled on every architecture so the build has one source
 * list rather than a conditional one. ISO C requires a translation unit to
 * declare something, so on non-x86 targets it declares this and nothing else. */
typedef int dranzer_matmul_x86_not_supported;

#endif /* DRANZER_HAVE_X86_SIMD */
