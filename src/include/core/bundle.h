#ifndef MODEL_BUNDLE_H
#define MODEL_BUNDLE_H

#include "byte_pair_encoding.h"
#include "core/model_types.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
    BUNDLE_SUCCESS = 0,
    BUNDLE_NOT_BUNDLE,
    BUNDLE_IO_ERROR,
    BUNDLE_FORMAT_ERROR,
    BUNDLE_CHECKSUM_ERROR,
    BUNDLE_ALLOCATION_FAILURE,
    BUNDLE_UNSUPPORTED,
} bundle_errors_t;

typedef struct {
    size_t train_window;
    uint64_t seed;
    uint64_t input_fingerprint;
    uint64_t input_bytes;
} model_bundle_metadata_t;

/* Atomic, canonical little-endian model/tokenizer artifact. */
bundle_errors_t model_bundle_save(const neural_model_t *model,
                                  const bpe_encoder_t *encoder,
                                  const model_bundle_metadata_t *metadata,
                                  const char *filename);

/* model must be zero-initialized. Loads a newly allocated frozen encoder
 * into out_encoder. A non-bundle legacy artifact returns BUNDLE_NOT_BUNDLE
 * without modifying model. */
bundle_errors_t model_bundle_load(neural_model_t *model,
                                  bpe_encoder_t **out_encoder,
                                  model_bundle_metadata_t *out_metadata,
                                  const char *filename);

#endif /* MODEL_BUNDLE_H */
