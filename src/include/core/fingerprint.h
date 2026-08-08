#ifndef FINGERPRINT_H
#define FINGERPRINT_H

#include "core/model_types.h"
#include <stddef.h>
#include <stdint.h>

/* Stable content hashes: one number that answers "are these the same bytes?".
 *
 * FNV-1a, 64-bit. Not a cryptographic hash and not used as one - nothing here
 * defends against an adversary constructing a collision. It is used to compare
 * artifacts that should be identical: the same seed on two compilers, a
 * training run against its own rerun, a bundle against the weights it claims
 * to carry.
 *
 * The reason this is a module rather than a few lines in each caller is the
 * endianness handling below. A fingerprint that means different things on
 * different machines cannot answer the question it exists to answer, and there
 * are now four consumers (the bundle format, the determinism check, provenance
 * records, and the results store key). This lives in one place so all four
 * produce the same number for the same weights.
 */

#define DRANZER_FNV_OFFSET UINT64_C(14695981039346656037)

/* Hash `size` bytes into a running value. Start from DRANZER_FNV_OFFSET.
 * Chaining calls is equivalent to one call over the concatenation. */
uint64_t dranzer_fnv1a(uint64_t running, const void *data, size_t size);

/* Fingerprint of every trainable parameter, in layout order.
 *
 * Each float is hashed as its four IEEE-754 bytes in a fixed (little-endian)
 * order rather than in the host's, so a big-endian machine agrees with a
 * little-endian one - the property that lets this number be quoted in a paper
 * as identifying a set of weights. Hashing the raw buffer with memcpy would be
 * faster and would not have it.
 *
 * Covers params only: not optimizer moments, not step counters, not the RNG
 * position. Two models with the same fingerprint compute the same function;
 * they may be at different points in a training run. */
uint64_t dranzer_weights_fingerprint(const neural_model_t *model);

#endif /* FINGERPRINT_H */
