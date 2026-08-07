#ifndef MATMUL_DISPATCH_H
#define MATMUL_DISPATCH_H

#include "core/model_types.h"

/* Which matmul implementation a model call actually reaches.
 *
 * This policy used to exist as two private copies - a forward one in
 * transformer.c and a backward one in training.c - which was fine while
 * those were the only two callers. core/lm_head.c is a third, and it needs
 * both halves, so the policy moved here rather than being duplicated a
 * third time. The behaviour is unchanged from those copies; only the
 * location is new.
 *
 * The forward and backward rules are deliberately different, and the
 * asymmetry is structural rather than a preference. A forward matmul moves
 * two buffers (activations up, result down) and its weight operand is
 * already device-resident, so it goes to the GPU whenever one is usable. A
 * backward matmul moves four - both inputs up, and the destination both up
 * and down, because these functions accumulate into a gradient that earlier
 * call sites in the same minibatch have already written - so it only goes
 * above a measured shape threshold.
 *
 * Those thresholds have been re-derived three times, every time because the
 * CPU got faster:
 *
 *   2^20 / 2^23   original, against a matmul_backward_weight() that strided
 *                 both operands in its innermost loop
 *   2^23 / 2^23   after that loop order was fixed (CPU 5.6-21.8x faster)
 *   2^25 / 2^26   after the backward kernels gained AVX-512 (CPU ~2-3x
 *                 faster again)
 *
 * The GPU never got worse. Each time, the baseline it is measured against
 * got better, and the range of shapes where a round trip pays shrank. That
 * is the standing caution for these constants and for every speedup ratio
 * in this project: a threshold measures two implementations, not one.
 *
 * Only the weight threshold is measured. backward_weight wins 1.62x and
 * 1.32x at the two largest shapes the benchmark issues (both 2^25
 * multiply-accumulates) and loses at 2^23, so 2^25 sits on the far side of
 * a crossover that was actually observed. backward_input never won at any
 * measured shape - it reaches 0.95x at 2^25 and is still climbing - so its
 * threshold is an extrapolation of that trend, deliberately set beyond
 * every shape this project benchmarks. In practice that means
 * backward_input runs on the CPU for every model here, which is what the
 * measurements support; the path stays for larger models and faster cards,
 * where it should be re-measured rather than trusted.
 *
 * NOTE: supervising every sequence position (core/lm_head.c) turned the
 * output head's three matmuls from m=1 into m=seq_len, which is the first
 * change in this project to move shapes *up* rather than making the CPU
 * faster. These thresholds are therefore due a re-measurement in that
 * direction for the first time; until that happens they stay where the last
 * measurement put them. */
#define GPU_BACKWARD_WEIGHT_MIN_WORK (1u << 25)
#define GPU_BACKWARD_INPUT_MIN_WORK  (1u << 26)

/* C (m x n) = A (m x k) @ B (k x n).
 * Scalar reference when the model asks for it, GPU when the model opts in
 * and one is usable, dispatched CPU kernel otherwise. */
void model_dispatch_matmul(neural_model_t *model, float *A, float *B, float *C,
                           size_t m, size_t k, size_t n);

/* dA (m x k) += dC (m x n) @ B_transposed.
 * A failed GPU call is not an error: it falls through to the CPU, which is
 * what also happens when no GPU exists at all. */
void model_dispatch_backward_input(neural_model_t *model, float *dC, float *B,
                                   float *dA, size_t m, size_t k, size_t n);

/* dB (k x n) += A_transposed @ dC (m x n), same contract. */
void model_dispatch_backward_weight(neural_model_t *model, float *A, float *dC,
                                    float *dB, size_t m, size_t k, size_t n);

/* True when this shape is worth a backward round trip. Exposed so tests can
 * pin the threshold rather than rediscovering it. */
int model_gpu_backward_worthwhile(size_t m, size_t k, size_t n, size_t min_work);

#endif // MATMUL_DISPATCH_H
