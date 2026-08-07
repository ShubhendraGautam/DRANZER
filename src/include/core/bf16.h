#ifndef BF16_H
#define BF16_H

#include <stddef.h>
#include <stdint.h>

/* bfloat16 weight storage, and the matmul that consumes it.
 *
 * What this is for
 * ----------------
 * `docs/design-checklist.md` asks, under weight-only quantization part 3,
 * whether keeping weights narrow *in memory* and widening them per tile
 * inside the kernel is actually faster - the saving being bandwidth rather
 * than disk - and records that the dequantization cost eating the bandwidth
 * win is the plausible outcome at this project's shapes.
 *
 * bf16 is the sharpest possible test of that question, which is why it comes
 * before INT8. Its widening is a 16-bit left shift into the high half of a
 * float: three instructions per vector, no multiply, no per-tile scale to
 * load, no zero-point. Every other narrow format costs strictly more to
 * unpack. **So if bf16 does not pay for itself here, no weight-only scheme
 * will, and part 3 is answered in the negative for the whole family rather
 * than for one encoding.** A win, conversely, only establishes the ceiling.
 *
 * Why bf16 rather than fp16
 * -------------------------
 * Same exponent range as binary32, so a weight tensor converts without any
 * risk of overflow or of denormal flush, and no scaling factor is needed to
 * keep values in range. It pays for that with 8 explicit mantissa bits
 * against fp16's 10, which for weight storage is the right trade: the
 * accuracy work in docs/quantization.md found weight-space error mattering
 * through outliers and dynamic range, not through fine resolution near zero.
 *
 * Relationship to core/quantize.h
 * -------------------------------
 * Deliberately separate. That module is *simulated* quantization - quantize
 * then immediately dequantize in place, so nothing downstream changes
 * representation - because it isolates the accuracy question from the
 * engineering one. This module is the engineering one: values genuinely
 * occupy 16 bits in memory and are widened in the kernel. The two answer
 * different questions and must not be merged.
 */

/* Storage type. A bare uint16_t holding the top 16 bits of a binary32, so an
 * array of these is exactly half the bytes of the float array it came from.
 * Not a struct: it has to be memcpy-able and vector-loadable. */
typedef uint16_t bf16_t;

/* Round-to-nearest-even, the only rounding this project uses for bf16.
 *
 * Truncation is one instruction cheaper and is what a naive `>> 16` does, but
 * it biases every value toward zero, and a matmul sums k of them - so the
 * bias accumulates with the reduction length instead of cancelling. That is
 * the difference between an error that stays at the rounding floor and one
 * that grows with k. NaN is passed through with its payload made non-zero so
 * it cannot silently become an infinity. */
bf16_t bf16_from_f32(float value);
float bf16_to_f32(bf16_t value);

/* Bulk conversion. `count` elements, no aliasing. */
void bf16_encode_array(const float *restrict src, bf16_t *restrict dst, size_t count);
void bf16_decode_array(const bf16_t *restrict src, float *restrict dst, size_t count);

/* C (m x n) = A (m x k) @ B (k x n), where **B is bf16 and everything else is
 * binary32**. C is fully written, not accumulated. A, B, and C must not
 * overlap.
 *
 * Accumulation stays in binary32: the bf16 values are widened to float as
 * they are loaded and every product and partial sum is a full float. So the
 * only difference from matrix_multiply() on the same data is that B's values
 * have been rounded to 8 mantissa bits - not a narrower accumulator, which is
 * a separate and much more dangerous change.
 *
 * Dispatches on cpu_isa_available() exactly as the backward matmuls do,
 * widest rung first, with a portable fallback. `tile` is the block edge, as
 * in the other blocked kernels; 0 uses matmul_tile_size(). */
void matmul_bf16_weight(const float *restrict A, const bf16_t *restrict B,
                        float *restrict C, size_t m, size_t k, size_t n,
                        size_t tile);

#endif // BF16_H
