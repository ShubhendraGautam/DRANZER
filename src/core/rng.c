/*
 * The project's pseudo-random generator - see core/rng.h for why it exists and
 * why the algorithm is written down rather than delegated to libc.
 *
 * Every constant and every shift below is part of the reproducibility contract
 * (docs/reproducibility.md). Changing one changes every seeded result this
 * project has ever reported, so tests/core/test_rng.c pins the output against
 * literals: an accidental edit fails a test rather than silently invalidating a
 * corpus of results.
 */

#include "core/rng.h"

/* SplitMix64, used only to derive a starting state from (seed, stream). It is
 * a full-period counter mixer, so unlike feeding a small seed straight into
 * xorshift64 it gives seeds 0, 1, and 2 unrelated streams rather than three
 * nearly identical ones. */
static uint64_t splitmix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

uint64_t dranzer_rng_stream(uint64_t seed, uint64_t stream) {
    uint64_t state = splitmix64(seed ^ splitmix64(stream));
    /* xorshift64 maps 0 to 0, so a zero state emits nothing but zero forever.
     * SplitMix64 does produce 0 for exactly one input, and a user is entitled
     * to pass any seed, so this is a real case rather than a defensive
     * flourish. The substitute is the golden-ratio constant, which is what
     * core/model.c used for the same purpose before this module existed. */
    return state ? state : UINT64_C(0x9e3779b97f4a7c15);
}

uint64_t dranzer_rng_next(uint64_t *state) {
    /* Same guard as above, for a state that arrives from outside - a
     * deserialized checkpoint field, or a caller that zeroed its struct. */
    uint64_t x = *state ? *state : UINT64_C(0x9e3779b97f4a7c15);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

double dranzer_rng_unit(uint64_t *state) {
    /* Top 53 bits over 2^53: exactly representable, uniform over the doubles
     * with that spacing, and never 1.0. Dividing by RAND_MAX (as the rand()
     * code did) can return exactly 1.0, which makes a "< keep_prob" test
     * subtly wrong at the boundary. */
    return (double)(dranzer_rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

float dranzer_rng_uniform(uint64_t *state, float low, float high) {
    double unit = dranzer_rng_unit(state);
    return (float)((double)low + unit * ((double)high - (double)low));
}
