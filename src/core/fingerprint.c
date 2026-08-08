/*
 * Content hashes - see core/fingerprint.h.
 *
 * Extracted from core/bundle.c, which had the only copy, once three more
 * callers needed the identical number. The constants and the byte order are
 * unchanged from that version, so every bundle written before this module
 * existed still validates against it.
 */

#include "core/fingerprint.h"
#include <string.h>

#define FNV_PRIME UINT64_C(1099511628211)

uint64_t dranzer_fnv1a(uint64_t running, const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < size; i++) {
        running ^= bytes[i];
        running *= FNV_PRIME;
    }
    return running;
}

uint64_t dranzer_weights_fingerprint(const neural_model_t *model) {
    if (!model || !model->params) return DRANZER_FNV_OFFSET;

    uint64_t fingerprint = DRANZER_FNV_OFFSET;
    for (size_t i = 0; i < model->total_param_count; i++) {
        uint32_t bits = 0;
        memcpy(&bits, &model->params[i], sizeof(bits));
        /* Fixed little-endian order, independent of the host's - see the
         * header. Written out by shift rather than by memcpy of the uint32_t
         * for exactly that reason. */
        uint8_t encoded[4] = {
            (uint8_t)(bits & 0xFFu),
            (uint8_t)((bits >> 8) & 0xFFu),
            (uint8_t)((bits >> 16) & 0xFFu),
            (uint8_t)((bits >> 24) & 0xFFu),
        };
        fingerprint = dranzer_fnv1a(fingerprint, encoded, sizeof(encoded));
    }
    return fingerprint;
}
