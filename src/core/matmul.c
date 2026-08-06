/*
 * CPU matrix-multiplication kernels and the policy that picks between them.
 * See matmul.h for the contract and docs/matmul.md for the measurements the
 * default kernel and tile size were selected from.
 */

#include "core/matmul.h"
#include "core/matmul_simd.h"
#include "core/parallel.h"
#include <string.h>

#if DRANZER_MATMUL_BLOCK_SIZE < 1
#error "DRANZER_MATMUL_BLOCK_SIZE must be positive"
#endif

/* Process-wide configuration. Written once during startup (or by the
 * benchmark between measurements) and only read during compute, including
 * from inside OpenMP regions - see the threading note in matmul.h. */
static matmul_kernel_t active_kernel = MATMUL_KERNEL_AUTO;
static size_t active_tile = DRANZER_MATMUL_BLOCK_SIZE;

static int kernel_in_range(matmul_kernel_t kernel);

static void kernel_scalar(const float *restrict A, const float *restrict B,
                          float *restrict C,
                          size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* Unblocked i/l/j. C and B are walked contiguously in the inner loop, which
 * is what lets the compiler vectorize portably without any target-specific
 * intrinsics. */
static void kernel_rowwise(const float *restrict A, const float *restrict B,
                           float *restrict C,
                           size_t m, size_t k, size_t n) {
    memset(C, 0, m * n * sizeof(float));

    /* One chunk per row of C; the rows are disjoint. */
    DRANZER_PARALLEL_FOR(m, m * k * n, i,
        float *row_out = &C[i * n];
        for (size_t l = 0; l < k; l++) {
            float a = A[i * k + l];
            const float *row_b = &B[l * n];
            for (size_t j = 0; j < n; j++) {
                row_out[j] += a * row_b[j];
            }
        }
    );
}

/* Cache-blocked i/l/j. Keeps one output/B tile hot while walking k, with j
 * innermost so C and B stay contiguous. Each C element still accumulates l
 * in increasing order. */
static void kernel_tiled(const float *restrict A, const float *restrict B,
                         float *restrict C,
                         size_t m, size_t k, size_t n, size_t tile) {
    memset(C, 0, m * n * sizeof(float));

    /* The (ii, jj) nest flattened into one block index. `collapse(2)` with a
     * static schedule distributes exactly this iteration space in exactly this
     * order, so the work each thread gets is unchanged; a single loop is what
     * lets the same body serve both the forked and the serial path, and it
     * makes the chunk count the cutoff needs an expression rather than a
     * guess. */
    const size_t i_blocks = (m + tile - 1) / tile;
    const size_t j_blocks = (n + tile - 1) / tile;

    DRANZER_PARALLEL_FOR(i_blocks * j_blocks, m * k * n, blk,
        size_t ii = (blk / j_blocks) * tile;
        size_t jj = (blk % j_blocks) * tile;
        size_t i_limit = (ii + tile > m) ? m : ii + tile;
        size_t j_limit = (jj + tile > n) ? n : jj + tile;

        for (size_t ll = 0; ll < k; ll += tile) {
            size_t l_limit = (ll + tile > k) ? k : ll + tile;

            for (size_t i = ii; i < i_limit; i++) {
                for (size_t l = ll; l < l_limit; l++) {
                    float a = A[i * k + l];
                    for (size_t j = jj; j < j_limit; j++) {
                        C[i * n + j] += a * B[l * n + j];
                    }
                }
            }
        }
    );
}

/* Cache-blocked with four rows of A in flight. Every B element loaded in the
 * inner loop is consumed by four independent accumulator rows, so the same
 * number of B loads does four times the arithmetic; the four chains are also
 * independent, which keeps the FMA pipeline fed. Row counts that are not a
 * multiple of four fall back to the one-row body for the remainder, so the
 * accumulation order per C element is unchanged from kernel_tiled. */
static void kernel_tiled_mr4(const float *restrict A, const float *restrict B,
                             float *restrict C,
                             size_t m, size_t k, size_t n, size_t tile) {
    memset(C, 0, m * n * sizeof(float));

    /* Flattened for the same reason as kernel_tiled() above. */
    const size_t i_blocks = (m + tile - 1) / tile;
    const size_t j_blocks = (n + tile - 1) / tile;

    DRANZER_PARALLEL_FOR(i_blocks * j_blocks, m * k * n, blk,
        size_t ii = (blk / j_blocks) * tile;
        size_t jj = (blk % j_blocks) * tile;
        size_t i_limit = (ii + tile > m) ? m : ii + tile;
        size_t j_limit = (jj + tile > n) ? n : jj + tile;

        for (size_t ll = 0; ll < k; ll += tile) {
            size_t l_limit = (ll + tile > k) ? k : ll + tile;

            size_t i = ii;
            for (; i + 4 <= i_limit; i += 4) {
                float *c0 = &C[i * n];
                float *c1 = c0 + n;
                float *c2 = c1 + n;
                float *c3 = c2 + n;
                for (size_t l = ll; l < l_limit; l++) {
                    float a0 = A[i * k + l];
                    float a1 = A[(i + 1) * k + l];
                    float a2 = A[(i + 2) * k + l];
                    float a3 = A[(i + 3) * k + l];
                    const float *row_b = &B[l * n];
                    for (size_t j = jj; j < j_limit; j++) {
                        float b = row_b[j];
                        c0[j] += a0 * b;
                        c1[j] += a1 * b;
                        c2[j] += a2 * b;
                        c3[j] += a3 * b;
                    }
                }
            }
            for (; i < i_limit; i++) {
                for (size_t l = ll; l < l_limit; l++) {
                    float a = A[i * k + l];
                    const float *row_b = &B[l * n];
                    float *row_out = &C[i * n];
                    for (size_t j = jj; j < j_limit; j++) {
                        row_out[j] += a * row_b[j];
                    }
                }
            }
        }
    );
}

/* One kernel for every shape, and that is a measured result rather than a
 * missing feature.
 *
 * The sweep in docs/matmul.md compared all four kernels at five tile sizes
 * over the six shapes the model issues, on three model tiers, under GCC and
 * Clang. `tiled_mr4` at the default tile was the fastest single choice on
 * both compilers - never worse than 1.4x off the best candidate for a shape,
 * and faster than the scalar reference on all 36 measured combinations.
 *
 * Two shape splits looked attractive and were rejected because they did not
 * hold up across compilers or across sessions:
 *   - "single-row decode takes the unblocked kernel" is worth up to 1.26x
 *     under Clang and costs up to 1.5x under GCC, which prefers the blocked
 *     kernels even at m == 1.
 *   - "wide outputs (n >= 1024) take the plain tiled kernel" improved the
 *     average by about 1%, which is smaller than the run-to-run spread of
 *     the session that produced it.
 * Re-run the sweep before adding either; `matmul_set_kernel()` already
 * covers the case where a deployment knows better than the default.
 *
 * The parameters stay in the signature because the equivalence test pins the
 * contract that selection depends on nothing else, and because a future sweep
 * may yet find a split the current measurements do not support.
 *
 * Runtime SIMD dispatch adds one step in front of all of it: the widest
 * vector kernel this CPU can execute.
 *
 * That ordering was measured, rejected, and then measured again, and the
 * history matters more than the conclusion. The SIMD kernels originally held
 * their accumulators in memory, reading and writing C on every step of k.
 * Under that structure avx2_mr4 was worth 1.47x on GCC and 0.92x on Clang -
 * a regression on this project's default compiler - so the policy declined to
 * select it, correctly on the evidence available. Disassembly later located
 * the cause: `fmadd(a, b, load(c))` can encode the load as a memory operand,
 * GCC folded all four and Clang folded one of four.
 *
 * Rewriting both kernels to keep their accumulators in registers removed the
 * load rather than the disagreement, and the numbers moved to:
 *
 *   avx512_mr4   ~3.1x GCC, ~3.2x Clang, over tiled_mr4
 *   avx2_mr4     1.95x GCC, 2.02x Clang
 *
 * Both compilers now agree to within a few percent on both kernels, and the
 * width ordering holds on every shape but the narrowest. The lesson recorded
 * here is that "measured slower" is a fact about an implementation, not about
 * an instruction set: the right response to a kernel that loses is to find
 * out why before retiring it.
 *
 * neon_mr4 is still not selected, and now for one reason rather than two: no
 * AArch64 hardware was available to measure it. The old second reason - that
 * a hand-written kernel cannot beat what the compiler already emits for a
 * baseline instruction set - is exactly the claim the rewrite above
 * falsified, so it no longer carries weight. It remains selectable and
 * correctness-checked; see docs/matmul.md. */
matmul_kernel_t matmul_select(size_t m, size_t k, size_t n) {
    (void)m;
    (void)k;
    (void)n;

    if (matmul_kernel_available(MATMUL_KERNEL_AVX512_MR4)) {
        return MATMUL_KERNEL_AVX512_MR4;
    }
    if (matmul_kernel_available(MATMUL_KERNEL_AVX2_MR4)) {
        return MATMUL_KERNEL_AVX2_MR4;
    }
    return MATMUL_KERNEL_TILED_MR4;
}

cpu_isa_t matmul_kernel_isa(matmul_kernel_t kernel) {
    switch (kernel) {
        case MATMUL_KERNEL_AVX2_MR4:   return CPU_ISA_AVX2;
        case MATMUL_KERNEL_AVX512_MR4: return CPU_ISA_AVX512;
        case MATMUL_KERNEL_NEON_MR4:   return CPU_ISA_NEON;
        default:                       return CPU_ISA_BASELINE;
    }
}

int matmul_kernel_available(matmul_kernel_t kernel) {
    if (!kernel_in_range(kernel)) return 0;

    /* Compiled in for this architecture? The DRANZER_HAVE_* macros come from
     * core/matmul_simd.h, which is the single place that decides. */
    switch (kernel) {
        case MATMUL_KERNEL_AVX2_MR4:
        case MATMUL_KERNEL_AVX512_MR4:
#ifndef DRANZER_HAVE_X86_SIMD
            return 0;
#else
            break;
#endif
        case MATMUL_KERNEL_NEON_MR4:
#ifndef DRANZER_HAVE_NEON
            return 0;
#else
            break;
#endif
        default:
            break;
    }

    /* Executable on this CPU? */
    return cpu_isa_available(matmul_kernel_isa(kernel));
}

void matmul_run(matmul_kernel_t kernel, const float *restrict A,
                const float *restrict B, float *restrict C,
                size_t m, size_t k, size_t n) {
    if (m == 0 || n == 0) return;
    if (k == 0) {
        memset(C, 0, m * n * sizeof(float));
        return;
    }
    if (kernel == MATMUL_KERNEL_AUTO) kernel = matmul_select(m, k, n);

    /* An explicitly pinned SIMD kernel this machine cannot run becomes the
     * portable kernel rather than an illegal instruction. matmul_select()
     * never produces one - it only ever returns available kernels - so this
     * only fires for matmul_set_kernel(), --kernel, and DRANZER_CPU_ISA. */
    if (!matmul_kernel_available(kernel)) kernel = MATMUL_KERNEL_TILED_MR4;

    switch (kernel) {
        case MATMUL_KERNEL_SCALAR:
            kernel_scalar(A, B, C, m, k, n);
            break;
        case MATMUL_KERNEL_ROWWISE:
            kernel_rowwise(A, B, C, m, k, n);
            break;
        case MATMUL_KERNEL_TILED:
            kernel_tiled(A, B, C, m, k, n, active_tile);
            break;
#ifdef DRANZER_HAVE_X86_SIMD
        case MATMUL_KERNEL_AVX2_MR4:
            matmul_kernel_avx2_mr4(A, B, C, m, k, n, active_tile);
            break;
        case MATMUL_KERNEL_AVX512_MR4:
            matmul_kernel_avx512_mr4(A, B, C, m, k, n, active_tile);
            break;
#endif
#ifdef DRANZER_HAVE_NEON
        case MATMUL_KERNEL_NEON_MR4:
            matmul_kernel_neon_mr4(A, B, C, m, k, n, active_tile);
            break;
#endif
        case MATMUL_KERNEL_TILED_MR4:
        default:
            kernel_tiled_mr4(A, B, C, m, k, n, active_tile);
            break;
    }
}

/* Range check through int: matmul_kernel_t's underlying type is
 * implementation-defined and may be unsigned, where `kernel < 0` would be a
 * tautology the compiler warns about. */
static int kernel_in_range(matmul_kernel_t kernel) {
    int value = (int)kernel;
    return value >= 0 && value < (int)MATMUL_KERNEL_COUNT;
}

int matmul_set_kernel(matmul_kernel_t kernel) {
    if (!kernel_in_range(kernel)) return -1;
    active_kernel = kernel;
    return 0;
}

matmul_kernel_t matmul_get_kernel(void) {
    return active_kernel;
}

int matmul_set_tile_size(size_t tile) {
    if (tile == 0) return -1;
    active_tile = tile;
    return 0;
}

size_t matmul_tile_size(void) {
    return active_tile;
}

/* Kept in the enum's order; the compiler cannot check that for a designated-
 * initializer-free array, so a new kernel means a new name in the same slot. */
static const char *const kernel_names[MATMUL_KERNEL_COUNT] = {
    "auto", "scalar", "rowwise", "tiled", "tiled_mr4",
    "avx2_mr4", "avx512_mr4", "neon_mr4"
};

const char *matmul_kernel_name(matmul_kernel_t kernel) {
    if (!kernel_in_range(kernel)) return "unknown";
    return kernel_names[kernel];
}

int matmul_kernel_from_name(const char *name, matmul_kernel_t *kernel_out) {
    if (!name || !kernel_out) return -1;
    for (int i = 0; i < MATMUL_KERNEL_COUNT; i++) {
        if (strcmp(name, kernel_names[i]) == 0) {
            *kernel_out = (matmul_kernel_t)i;
            return 0;
        }
    }
    return -1;
}

void matrix_multiply(float *restrict A, float *restrict B, float *restrict C,
                     size_t m, size_t k, size_t n) {
    matmul_run(active_kernel, A, B, C, m, k, n);
}

void matrix_multiply_scalar(const float *restrict A, const float *restrict B,
                            float *restrict C, size_t m, size_t k, size_t n) {
    if (m == 0 || n == 0) return;
    if (k == 0) {
        memset(C, 0, m * n * sizeof(float));
        return;
    }
    kernel_scalar(A, B, C, m, k, n);
}

/* Both backward functions take the AVX-512 path where the CPU has it and the
 * portable one everywhere else. There is no kernel enum and no shape
 * threshold here, unlike matmul_select(): these have one vector
 * implementation, it beat the portable code on every measured shape, and a
 * caller has nothing to choose between. cpu_isa_available() is the only gate,
 * and it answers from cached CPUID state, so this costs a predictable branch
 * rather than a probe. */
void matmul_backward_input(float *restrict dC, float *restrict B, float *restrict dA,
                            size_t m, size_t k, size_t n) {
#ifdef DRANZER_HAVE_X86_SIMD
    if (cpu_isa_available(CPU_ISA_AVX512)) {
        matmul_backward_input_avx512(dC, B, dA, m, k, n);
        return;
    }
#endif

    /* One chunk per row of dA; the rows are disjoint. */
    DRANZER_PARALLEL_FOR(m, m * k * n, i,
        for (size_t l = 0; l < k; l++) {
            float sum = 0.0f;
            for (size_t j = 0; j < n; j++) {
                sum += dC[i * n + j] * B[l * n + j];
            }
            dA[i * k + l] += sum;
        }
    );
}

/* Loop order l/i/j, not l/j/i.
 *
 * The obvious way to write this - accumulate a dot product over i into a
 * register, store once per output element - puts i innermost, where it strides
 * A by k and dC by n simultaneously. Every iteration of the hot loop then
 * touches two new cache lines and uses one float from each, which neither
 * caches nor vectorizes. It measured about 40 ms at 128x1024x256 against
 * roughly 6 ms for a forward matmul of the same FLOP count (docs/matmul.md).
 *
 * Putting j innermost instead walks dB and dC contiguously and reduces A to a
 * single scalar load per (l, i). Four rows of dB are kept in flight for the
 * same reason kernel_tiled_mr4 keeps four rows of C: each dC element loaded
 * then feeds four independent accumulator rows, and the four chains hide FMA
 * latency. Values of k that are not a multiple of four finish in the one-row
 * body below.
 *
 * This reassociates the sum. The old order added a fully-accumulated dot
 * product to dB once; this one adds each i's contribution in turn, so results
 * differ in the last bits from builds before this change - the same class of
 * difference as switching matmul kernels, and covered by the gradient checks'
 * tolerances rather than by bit-identity. Within one build it stays
 * deterministic, which is what exact resume actually requires. */
void matmul_backward_weight(float *restrict A, float *restrict dC, float *restrict dB,
                             size_t m, size_t k, size_t n) {
#ifdef DRANZER_HAVE_X86_SIMD
    if (cpu_isa_available(CPU_ISA_AVX512)) {
        matmul_backward_weight_avx512(A, dC, dB, m, k, n);
        return;
    }
#endif

    /* One chunk per group of four rows of dB, which is what the strided loop
     * this replaced already handed OpenMP; the rows are disjoint. */
    DRANZER_PARALLEL_FOR((k + 3) / 4, m * k * n, blk,
        size_t l = blk * 4;
        if (l + 4 <= k) {
            float *out0 = &dB[l * n];
            float *out1 = out0 + n;
            float *out2 = out1 + n;
            float *out3 = out2 + n;
            for (size_t i = 0; i < m; i++) {
                const float *row_dc = &dC[i * n];
                const float *a = &A[i * k + l];
                float a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
                for (size_t j = 0; j < n; j++) {
                    float d = row_dc[j];
                    out0[j] += a0 * d;
                    out1[j] += a1 * d;
                    out2[j] += a2 * d;
                    out3[j] += a3 * d;
                }
            }
        } else {
            for (size_t ll = l; ll < k; ll++) {
                float *row_out = &dB[ll * n];
                for (size_t i = 0; i < m; i++) {
                    float a = A[i * k + ll];
                    const float *row_dc = &dC[i * n];
                    for (size_t j = 0; j < n; j++) {
                        row_out[j] += a * row_dc[j];
                    }
                }
            }
        }
    );
}
