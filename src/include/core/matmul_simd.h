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

/* ------------------------------------------------------- backward pass ---
 *
 * AVX2 and AVX-512 counterparts of core/matmul.c's matmul_backward_input()
 * and matmul_backward_weight(), with identical shapes and the same
 * accumulate-into-destination contract.
 *
 * These were AVX-512 only until an AVX2-only CI runner failed the backward-
 * versus-forward cost invariant in tests/perf/test_perf_invariants.c at
 * 4.09x against a 3.00x limit. The reason was not cache traversal, which is
 * what that test's message assumes: matmul_select() ships avx2_mr4 for the
 * forward path, so on a CPU with AVX2 and no AVX-512 the forward matmul was
 * vectorized and the backward one fell all the way back to portable C. The
 * invariant measured that gap and was right to complain.
 *
 * The original argument for the asymmetry was that T11 measured avx2_mr4 as
 * a regression under Clang, and that the backward functions have no
 * `--kernel` selection so an AVX2 version would be unreachable code. Both
 * halves are now false. The first was falsified by the register-resident
 * rewrite, which took avx2_mr4 to 1.95x/2.02x and promoted it in
 * matmul_select() wherever AVX-512 is absent. The second confused
 * "unselectable by flag" with "unreachable": hardware selects these, and an
 * AVX2-only CPU reaches an AVX2 version on every call - the CI runner above
 * is exactly that machine. See docs/matmul.md.
 *
 * NEON still has no backward kernel. That one remains genuinely unmeasured
 * for want of AArch64 hardware, which is a different situation from this one
 * and is tracked separately in docs/design-checklist.md.
 *
 * All of these reassociate their sums relative to the portable versions: the
 * weight kernels because the vector lanes accumulate independently, the input
 * kernels because the dot product is reduced across lanes at the end. The
 * AVX2 and AVX-512 forms also differ from EACH OTHER for the same reason -
 * eight lanes folded versus sixteen - so a result is reproducible for one
 * binary on one CPU, not across instruction sets. That is the same contract
 * T11 already narrowed for the forward kernels, and it is covered by the
 * gradient checks' tolerances rather than by bit-identity. */
#ifdef DRANZER_HAVE_X86_SIMD
void matmul_backward_input_avx2(const float *restrict dC, const float *restrict B,
                                float *restrict dA, size_t m, size_t k, size_t n);
void matmul_backward_weight_avx2(const float *restrict A, const float *restrict dC,
                                 float *restrict dB, size_t m, size_t k, size_t n);
void matmul_backward_input_avx512(const float *restrict dC, const float *restrict B,
                                  float *restrict dA, size_t m, size_t k, size_t n);
void matmul_backward_weight_avx512(const float *restrict A, const float *restrict dC,
                                   float *restrict dB, size_t m, size_t k, size_t n);
#endif

#endif // MATMUL_SIMD_H
