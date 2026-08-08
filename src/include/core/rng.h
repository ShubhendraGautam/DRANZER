#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/*
 * The project's own pseudo-random generator, and the only one any path that
 * affects model state or generated tokens may use.
 *
 * Why this module exists
 * ----------------------
 * Weight initialization, dropout, and token sampling all used to draw from
 * rand(). rand() is implementation-defined: glibc, musl, macOS, and Windows
 * all produce different sequences from the same seed. So "--seed 42" did not
 * name a run. It named a run *on one C library*, and the most basic
 * reproducibility property a research artifact has - that a reader can
 * reproduce a training run from the seed you published - did not hold, and
 * nothing in the tree said so.
 *
 * It also relied on a process-global stream, which couples unrelated code:
 * inserting one draw anywhere upstream shifts every later draw, so adding a
 * dropout call could silently change the initial weights of a model.
 *
 * Everything here is written down, fixed, and pinned by a test against
 * constants (tests/core/test_rng.c), so it is identical on every platform and
 * every compiler this project builds on. Adding an unpinned generator, or a
 * call to rand(), reopens the hole.
 *
 * The algorithm
 * -------------
 * xorshift64 followed by a multiply by an odd constant ("xorshift64*"), which
 * is what core/tensor_ops.c's dropout stream already used. It is not chosen
 * for statistical excellence - it is chosen because it is four operations,
 * needs no state beyond a uint64_t (so a checkpoint stores a stream position
 * in eight bytes), and is defined entirely in terms of exact integer
 * arithmetic, which is the property that makes it portable. Its weakness is
 * low-bit structure in the raw state; the multiply is what the output is taken
 * from, and dranzer_rng_unit() reads the top 53 bits, so no consumer sees it.
 *
 * If a study ever needs a stronger generator, replace it deliberately: it
 * changes every seeded result, so it is a versioned change, not a cleanup.
 *
 * Streams
 * -------
 * One user seed, several independent streams, mixed by SplitMix64. Weight
 * initialization cannot be perturbed by a change to how often dropout draws,
 * and a sampling run cannot be perturbed by either. Add a stream by naming a
 * new constant; never reuse one.
 */

#define DRANZER_RNG_STREAM_INIT     UINT64_C(0x9e3779b97f4a7c15) /* weight initialization */
#define DRANZER_RNG_STREAM_DROPOUT  UINT64_C(0xbf58476d1ce4e5b9) /* dropout masks */
#define DRANZER_RNG_STREAM_SAMPLING UINT64_C(0x94d049bb133111eb) /* token sampling */
#define DRANZER_RNG_STREAM_TESTING  UINT64_C(0x2545f4914f6cdd1d) /* test fixtures only */

/* The initial state for one (seed, stream) pair. Deterministic, and never
 * zero: zero is xorshift64's fixed point and would emit nothing but zero. */
uint64_t dranzer_rng_stream(uint64_t seed, uint64_t stream);

/* Advance the state and return the next 64 bits. */
uint64_t dranzer_rng_next(uint64_t *state);

/* Uniform in [0, 1), from the top 53 bits - the most a double can hold
 * without repeating a value. Never returns 1.0. */
double dranzer_rng_unit(uint64_t *state);

/* Uniform in [low, high). Computed in double and rounded once on return, so
 * the result does not depend on x87 versus SSE intermediate precision. */
float dranzer_rng_uniform(uint64_t *state, float low, float high);

#endif /* RNG_H */
