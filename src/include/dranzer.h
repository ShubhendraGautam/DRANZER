#ifndef DRANZER_PUBLIC_H
#define DRANZER_PUBLIC_H

/* Stable, opaque embedding API. This header intentionally exposes no core,
 * CLI, tokenizer, or backend structure; consumers need only a C99 compiler
 * and the standard integer/size types. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DRANZER_API_VERSION UINT32_C(1)
#define DRANZER_VERSION_MAJOR UINT32_C(0)
#define DRANZER_VERSION_MINOR UINT32_C(5)
#define DRANZER_VERSION_PATCH UINT32_C(0)
#define DRANZER_VERSION_PRERELEASE "dev"
#define DRANZER_VERSION_STRING "0.5.0-dev"
#define DRANZER_BUNDLE_FORMAT_OLDEST_SUPPORTED UINT32_C(1)
#define DRANZER_BUNDLE_FORMAT_CURRENT UINT32_C(2)
#define DRANZER_BUNDLE_INFO_V1_SIZE UINT32_C(64)

typedef struct dranzer_model dranzer_model_t;
typedef struct dranzer_tokenizer dranzer_tokenizer_t;
typedef struct dranzer_cache dranzer_cache_t;
typedef struct dranzer_generation dranzer_generation_t;

typedef enum {
    DRANZER_OK = 0,
    DRANZER_INVALID_ARGUMENT = 1,
    DRANZER_IO_ERROR = 2,
    DRANZER_FORMAT_ERROR = 3,
    DRANZER_CHECKSUM_ERROR = 4,
    DRANZER_UNSUPPORTED = 5,
    DRANZER_OUT_OF_MEMORY = 6,
    DRANZER_BUFFER_TOO_SMALL = 7,
    DRANZER_MODEL_ERROR = 8,
    DRANZER_FINISHED = 9,
} dranzer_status_t;

typedef enum {
    DRANZER_LOAD_COPY = 0,
    DRANZER_LOAD_MMAP = 1,
} dranzer_load_mode_t;

typedef struct {
    uint32_t struct_size;
    uint32_t format_version;
    uint64_t train_window;
    uint64_t seed;
    uint64_t input_fingerprint;
    uint64_t input_bytes;
    uint64_t reserved[3];
} dranzer_bundle_info_t;

#define DRANZER_BUNDLE_INFO_INIT \
    { DRANZER_BUNDLE_INFO_V1_SIZE, 0, 0, 0, 0, 0, {0, 0, 0} }

/* Runtime/header compatibility check. API version 1 remains source- and
 * ABI-compatible until the next major API version. */
uint32_t dranzer_api_version(void);
const char *dranzer_version_string(void);

/* Load one bundle into separately owned opaque model and tokenizer handles.
 * On failure both outputs are NULL. MMAP accepts version-1 float bundles and
 * creates an inference-only model; COPY also accepts version 2. If out_info
 * is non-NULL, initialize it with DRANZER_BUNDLE_INFO_INIT. On return,
 * struct_size reports the bytes understood by the loaded library. */
dranzer_status_t dranzer_bundle_load(const char *path,
                                     dranzer_load_mode_t mode,
                                     dranzer_model_t **out_model,
                                     dranzer_tokenizer_t **out_tokenizer,
                                     dranzer_bundle_info_t *out_info);

void dranzer_model_free(dranzer_model_t *model);
void dranzer_tokenizer_free(dranzer_tokenizer_t *tokenizer);
size_t dranzer_model_vocab_size(const dranzer_model_t *model);
size_t dranzer_model_max_sequence(const dranzer_model_t *model);

/* Caller-buffer convention: the pointed-to count/size is capacity on entry
 * and required size on return. A NULL/short output returns
 * DRANZER_BUFFER_TOO_SMALL without a partial result. Byte strings are
 * length-delimited and may contain NUL. */
dranzer_status_t dranzer_tokenize(dranzer_tokenizer_t *tokenizer,
                                  const void *bytes, size_t byte_count,
                                  uint32_t *token_ids,
                                  size_t *inout_count);
dranzer_status_t dranzer_detokenize(dranzer_tokenizer_t *tokenizer,
                                    const uint32_t *token_ids,
                                    size_t token_count,
                                    void *bytes,
                                    size_t *inout_size);

dranzer_status_t dranzer_model_forward(dranzer_model_t *model,
                                       const uint32_t *token_ids,
                                       size_t token_count,
                                       float *logits,
                                       size_t logits_count);

/* Incremental cache handles retain their model, so destroying the caller's
 * model handle first is safe. A cache itself is stateful and not concurrent. */
dranzer_status_t dranzer_cache_create(dranzer_model_t *model,
                                      size_t capacity,
                                      dranzer_cache_t **out_cache);
void dranzer_cache_reset(dranzer_cache_t *cache);
void dranzer_cache_free(dranzer_cache_t *cache);
dranzer_status_t dranzer_cache_forward(dranzer_cache_t *cache,
                                       uint32_t token_id,
                                       float *logits,
                                       size_t logits_count);

/* Greedy generation session. reset tokenizes the prompt, prepends BOS for a
 * special-token tokenizer, retains the newest prompt tokens that fit, and
 * prefills the cache. next returns one token and its exact decoded byte piece;
 * EOS returns DRANZER_FINISHED with a zero-byte piece. The handle retains both
 * model and tokenizer. */
dranzer_status_t dranzer_generation_create(
    dranzer_model_t *model, dranzer_tokenizer_t *tokenizer,
    size_t context_capacity, dranzer_generation_t **out_generation);
void dranzer_generation_free(dranzer_generation_t *generation);
dranzer_status_t dranzer_generation_reset(dranzer_generation_t *generation,
                                           const void *prompt,
                                           size_t prompt_size);
dranzer_status_t dranzer_generation_next_greedy(
    dranzer_generation_t *generation, uint32_t *out_token_id,
    void *piece, size_t *inout_piece_size);

const char *dranzer_status_string(dranzer_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* DRANZER_PUBLIC_H */
