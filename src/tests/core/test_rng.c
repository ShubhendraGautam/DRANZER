/* The reproducibility contract for core/rng.h, pinned against literals.
 *
 * Every other determinism test in this suite compares a run against another run
 * of the same binary, so all of them would still pass if the generator changed.
 * This one would not. The constants below were produced by this project's
 * generator and are now part of its contract: a seed identifies a training run
 * only for as long as these numbers hold, and a diff that changes them is
 * invalidating every seeded result the project has ever reported.
 *
 * So a failure here is not "fix the test". It is a decision: either the change
 * was accidental and should be reverted, or it is deliberate and needs a
 * version bump, a note in docs/reproducibility.md, and the results measured
 * under the old generator marked as belonging to it.
 *
 * The three properties checked, in the order they matter:
 *   1. Fixed output. The stream derivation and the generator produce exactly
 *      these values, on every platform, compiler, and libc.
 *   2. Stream independence. One seed's streams are unrelated to each other, so
 *      a change to how often dropout draws cannot move the initial weights.
 *   3. Range and edge cases. Unit draws stay in [0, 1), a zero state cannot
 *      collapse the generator, and any seed is usable.
 */
#include "core/rng.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define DRAWS 6

static int failures = 0;

static void expect_u64(const char *what, uint64_t got, uint64_t want) {
    if (got != want) {
        printf("  FAIL %s: got 0x%016" PRIX64 ", contract says 0x%016" PRIX64 "\n",
               what, got, want);
        failures++;
    }
}

int main(void) {
    struct {
        const char *name;
        uint64_t stream;
        uint64_t initial_state;
        uint64_t draws[DRAWS];
    } pinned[] = {
        { "init", DRANZER_RNG_STREAM_INIT, UINT64_C(0x7C247ADEFCC8B7D8),
          { UINT64_C(0x74E6E38914990A99), UINT64_C(0xE691ECCDAFF973E7),
            UINT64_C(0xCDBCCD2D9CE51561), UINT64_C(0x2312AC28B968814A),
            UINT64_C(0xCC25BD84D515B7C8), UINT64_C(0x6E3591667D837B0F) } },
        { "dropout", DRANZER_RNG_STREAM_DROPOUT, UINT64_C(0x16A27C5E3E3F3C94),
          { UINT64_C(0x4A129E81ABCFD69D), UINT64_C(0xF862E617FC394C6F),
            UINT64_C(0x781752B023792B70), UINT64_C(0x0E406FCEFDBCBB04),
            UINT64_C(0xEBF205EBB93362B0), UINT64_C(0xF14F0245AA6151D6) } },
        { "sampling", DRANZER_RNG_STREAM_SAMPLING, UINT64_C(0xAA80027AC7B32959),
          { UINT64_C(0xF3FBD30AEAE63199), UINT64_C(0x6C467EA74180E1A3),
            UINT64_C(0x0009017CCD079354), UINT64_C(0x33875B26D07B1E58),
            UINT64_C(0x9EDE1A0789C7AF2F), UINT64_C(0x4823040D5887BF4A) } },
    };
    const size_t stream_count = sizeof(pinned) / sizeof(pinned[0]);

    /* 1. Fixed output, for seed 42 on each named stream. */
    printf("pinned draws for seed 42:\n");
    for (size_t s = 0; s < stream_count; s++) {
        uint64_t state = dranzer_rng_stream(42, pinned[s].stream);
        expect_u64(pinned[s].name, state, pinned[s].initial_state);
        for (size_t i = 0; i < DRAWS; i++) {
            expect_u64(pinned[s].name, dranzer_rng_next(&state), pinned[s].draws[i]);
        }
        printf("  %-9s %d values match the contract\n", pinned[s].name, DRAWS + 1);
    }

    /* 2. Stream independence: no two streams off one seed share a draw. A
     *    collision here would mean weight initialization and dropout are
     *    reading the same numbers, which is the coupling the streams exist to
     *    prevent. */
    for (size_t a = 0; a < stream_count; a++) {
        for (size_t b = a + 1; b < stream_count; b++) {
            for (size_t i = 0; i < DRAWS; i++) {
                for (size_t j = 0; j < DRAWS; j++) {
                    if (pinned[a].draws[i] == pinned[b].draws[j]) {
                        printf("  FAIL %s and %s streams overlap at draw %zu/%zu\n",
                               pinned[a].name, pinned[b].name, i, j);
                        failures++;
                    }
                }
            }
        }
    }
    printf("  streams are disjoint over the first %d draws\n", DRAWS);

    /* Adjacent seeds must give unrelated streams. Feeding a small seed straight
     * into xorshift64 does not: 1, 2, and 3 differ in one bit and take many
     * rounds to diverge, which is why dranzer_rng_stream() mixes first. */
    for (uint64_t seed = 0; seed < 8; seed++) {
        uint64_t a = dranzer_rng_stream(seed, DRANZER_RNG_STREAM_INIT);
        uint64_t b = dranzer_rng_stream(seed + 1, DRANZER_RNG_STREAM_INIT);
        uint64_t x = dranzer_rng_next(&a), y = dranzer_rng_next(&b);
        /* Popcount of the XOR: two unrelated 64-bit values differ in about 32
         * bits. Fewer than 12 would mean the mixing is not working. */
        int differing = 0;
        for (uint64_t bit = x ^ y; bit; bit >>= 1) differing += (int)(bit & 1);
        if (differing < 12) {
            printf("  FAIL seeds %" PRIu64 " and %" PRIu64 " give first draws "
                   "differing in only %d bits\n", seed, seed + 1, differing);
            failures++;
        }
    }
    printf("  adjacent seeds give unrelated streams\n");

    /* 3a. Unit draws: in [0, 1), and one particular sequence pinned so the
     *     conversion from bits to double is part of the contract too. Written
     *     as bit patterns, because -ffast-math makes a float comparison
     *     against a literal an unreliable assertion (include/common/fp_bits.h). */
    const uint64_t unit_bits[] = {
        UINT64_C(0x3FE54B94A5BC9DA3),  /* 0.66547615404572402 */
        UINT64_C(0x3FC9AC8DCE2DCC4C),  /* 0.20057842795685799 */
        UINT64_C(0x3FB4F41D979BD588),  /* 0.081849908365105972 */
        UINT64_C(0x3FCAC3BA2B5730E8),  /* 0.20909812084426638 */
    };
    uint64_t unit_state = dranzer_rng_stream(7, DRANZER_RNG_STREAM_TESTING);
    for (size_t i = 0; i < sizeof(unit_bits) / sizeof(unit_bits[0]); i++) {
        double value = dranzer_rng_unit(&unit_state);
        uint64_t bits;
        memcpy(&bits, &value, sizeof(bits));
        expect_u64("unit", bits, unit_bits[i]);
    }
    printf("  unit draws checked bit for bit\n");

    uint64_t range_state = dranzer_rng_stream(3, DRANZER_RNG_STREAM_TESTING);
    for (size_t i = 0; i < 100000; i++) {
        double unit = dranzer_rng_unit(&range_state);
        if (!(unit >= 0.0) || !(unit < 1.0)) {
            printf("  FAIL unit draw %zu left [0, 1): %.17g\n", i, unit);
            failures++;
            break;
        }
        float uniform = dranzer_rng_uniform(&range_state, -2.5f, 2.5f);
        if (!(uniform >= -2.5f) || !(uniform < 2.5f)) {
            printf("  FAIL uniform draw %zu left [-2.5, 2.5): %.9g\n", i,
                   (double)uniform);
            failures++;
            break;
        }
    }
    printf("  200000 draws stayed inside their stated ranges\n");

    /* 3b. Zero is xorshift64's fixed point: a zero state must be replaced, not
     *     iterated, or the generator emits nothing but zero forever. Reachable
     *     from a caller that zeroed its struct or read a zeroed field out of a
     *     truncated checkpoint. */
    if (dranzer_rng_stream(0, 0) == 0) {
        printf("  FAIL stream derivation produced a zero state\n");
        failures++;
    }
    uint64_t zero_state = 0;
    uint64_t first = dranzer_rng_next(&zero_state);
    uint64_t second = dranzer_rng_next(&zero_state);
    if (first == 0 || second == 0 || first == second) {
        printf("  FAIL a zero state collapsed the generator (%016" PRIX64
               ", %016" PRIX64 ")\n", first, second);
        failures++;
    }
    printf("  a zero state recovers instead of collapsing\n");

    /* Seed 0 must be as usable as any other: it is the default
     * (MODEL_DEFAULT_SEED) and a plausible thing for a user to pass. */
    uint64_t zero_seed = dranzer_rng_stream(0, DRANZER_RNG_STREAM_INIT);
    uint64_t one_seed = dranzer_rng_stream(1, DRANZER_RNG_STREAM_INIT);
    if (zero_seed == 0 || zero_seed == one_seed) {
        printf("  FAIL seed 0 is not a usable seed\n");
        failures++;
    }
    printf("  seed 0 behaves like any other seed\n");

    printf("\n%s\n", failures == 0 ? "RNG CONTRACT CHECK PASSED"
                                   : "RNG CONTRACT CHECK FAILED");
    return failures == 0 ? 0 : 1;
}
