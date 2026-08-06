#ifndef MATMUL_SIMD_H
#define MATMUL_SIMD_H

#include <stddef.h>

/* Architecture-specific matmul kernels, behind the one place that decides
 * which architectures have them.
 *
 * This header is internal to the matmul module: callers use core/matmul.h and
 * select these through MATMUL_KERNEL_* like any other kernel. It exists so the
 * "is there an implementation for this architecture" question is answered
 * once, here, and both core/matmul.c's dispatch and the kernels' own
 * translation units agree on the answer by construction.
 *
 * Availability is two separate questions and both must be yes:
 *   - DRANZER_HAVE_* below: was an implementation compiled into this binary?
 *     A compile-time property of the target architecture.
 *   - cpu_isa_available(): can this CPU execute it? A runtime property, and
 *     the reason the kernels are compiled with per-function target attributes
 *     instead of whole-binary -march flags.
 *
 * Every kernel here computes exactly what kernel_tiled_mr4() in core/matmul.c
 * computes, with the same blocking and the same four-row structure: they
 * vectorize the innermost loop over j, and because each C element still
 * accumulates over l in increasing order, they agree with the scalar
 * reference to within FMA contraction rather than to within a reassociation.
 * The signature matches the blocked kernels': C is fully defined, A/B/C must
 * not overlap, and `tile` is the block edge.
 */

#if defined(__x86_64__)
#define DRANZER_HAVE_X86_SIMD 1
#endif

#if defined(__aarch64__)
#define DRANZER_HAVE_NEON 1
#endif

#ifdef DRANZER_HAVE_X86_SIMD
/* Requires CPU_ISA_AVX2 (AVX2 + FMA). Eight columns per vector. */
void matmul_kernel_avx2_mr4(const float *restrict A, const float *restrict B,
                            float *restrict C,
                            size_t m, size_t k, size_t n, size_t tile);

/* Requires CPU_ISA_AVX512 (AVX-512F + VL). Sixteen columns per vector, with a
 * mask-predicated tail instead of a scalar one. */
void matmul_kernel_avx512_mr4(const float *restrict A, const float *restrict B,
                              float *restrict C,
                              size_t m, size_t k, size_t n, size_t tile);
#endif

#ifdef DRANZER_HAVE_NEON
/* Requires CPU_ISA_NEON. Four columns per vector. */
void matmul_kernel_neon_mr4(const float *restrict A, const float *restrict B,
                            float *restrict C,
                            size_t m, size_t k, size_t n, size_t tile);
#endif

#endif // MATMUL_SIMD_H
