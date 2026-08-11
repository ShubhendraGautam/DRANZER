#ifndef MODEL_BUNDLE_H
#define MODEL_BUNDLE_H

#include "byte_pair_encoding.h"
#include "core/model_quantize.h"
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

typedef struct {
    uint64_t artifact_bytes;
    uint64_t weight_payload_bytes;
    uint64_t tokenizer_bytes;
    size_t tensors_quantized;
    size_t values_quantized;
    size_t scales_stored;
} model_bundle_storage_report_t;

/* Atomic, canonical little-endian model/tokenizer artifact. */
bundle_errors_t model_bundle_save(const neural_model_t *model,
                                  const bpe_encoder_t *encoder,
                                  const model_bundle_metadata_t *metadata,
                                  const char *filename);

/* Store tensors selected by `config` as packed symmetric integer codes plus
 * float32 scales. Excluded tensors remain float32. Loading is representation
 * transparent: model_bundle_load() reconstructs an ordinary float model, so
 * inference and training kernels do not gain a quantized code path.
 *
 * Unlike model_quantize_weights(), this does not modify `model`. bits must be
 * in QUANT_MIN_BITS..QUANT_MAX_BITS; use model_bundle_save() for an unquantized
 * artifact. `out_report` may be NULL. */
bundle_errors_t model_bundle_save_quantized(
    const neural_model_t *model, const bpe_encoder_t *encoder,
    const model_bundle_metadata_t *metadata,
    const model_quant_config_t *config,
    model_bundle_storage_report_t *out_report,
    const char *filename);

/* model must be zero-initialized. Loads a newly allocated frozen encoder
 * into out_encoder. A non-bundle legacy artifact returns BUNDLE_NOT_BUNDLE
 * without modifying model. */
bundle_errors_t model_bundle_load(neural_model_t *model,
                                  bpe_encoder_t **out_encoder,
                                  model_bundle_metadata_t *out_metadata,
                                  const char *filename);

/* Map a version-1 float32 bundle and make every parameter view point directly
 * into its read-only weight payload. This avoids the parameter allocation,
 * initialization, and copy performed by model_bundle_load(); model_free()
 * releases the mapping. The resulting model is inference-only. Quantized
 * version-2 bundles require unpacking and therefore return BUNDLE_UNSUPPORTED.
 * As above, model must be zero-initialized. */
bundle_errors_t model_bundle_load_mmap(neural_model_t *model,
                                       bpe_encoder_t **out_encoder,
                                       model_bundle_metadata_t *out_metadata,
                                       const char *filename);

#endif /* MODEL_BUNDLE_H */
