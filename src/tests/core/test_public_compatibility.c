/* Frozen source/ABI facts for public API version 1. This deliberately includes
 * no internal header; changing one of these facts requires a new API major. */

#include "dranzer.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ABI_ASSERT(name, condition) \
    typedef char abi_assert_##name[(condition) ? 1 : -1]

ABI_ASSERT(api_version, DRANZER_API_VERSION == 1);
ABI_ASSERT(oldest_bundle, DRANZER_BUNDLE_FORMAT_OLDEST_SUPPORTED == 1);
ABI_ASSERT(current_bundle, DRANZER_BUNDLE_FORMAT_CURRENT == 2);
ABI_ASSERT(status_ok, DRANZER_OK == 0);
ABI_ASSERT(status_invalid, DRANZER_INVALID_ARGUMENT == 1);
ABI_ASSERT(status_io, DRANZER_IO_ERROR == 2);
ABI_ASSERT(status_format, DRANZER_FORMAT_ERROR == 3);
ABI_ASSERT(status_checksum, DRANZER_CHECKSUM_ERROR == 4);
ABI_ASSERT(status_unsupported, DRANZER_UNSUPPORTED == 5);
ABI_ASSERT(status_memory, DRANZER_OUT_OF_MEMORY == 6);
ABI_ASSERT(status_buffer, DRANZER_BUFFER_TOO_SMALL == 7);
ABI_ASSERT(status_model, DRANZER_MODEL_ERROR == 8);
ABI_ASSERT(status_finished, DRANZER_FINISHED == 9);
ABI_ASSERT(load_copy, DRANZER_LOAD_COPY == 0);
ABI_ASSERT(load_mmap, DRANZER_LOAD_MMAP == 1);
ABI_ASSERT(bundle_info_size, sizeof(dranzer_bundle_info_t) == 64);
ABI_ASSERT(bundle_info_struct_size,
           offsetof(dranzer_bundle_info_t, struct_size) == 0);
ABI_ASSERT(bundle_info_format,
           offsetof(dranzer_bundle_info_t, format_version) == 4);
ABI_ASSERT(bundle_info_window,
           offsetof(dranzer_bundle_info_t, train_window) == 8);
ABI_ASSERT(bundle_info_seed, offsetof(dranzer_bundle_info_t, seed) == 16);
ABI_ASSERT(bundle_info_fingerprint,
           offsetof(dranzer_bundle_info_t, input_fingerprint) == 24);
ABI_ASSERT(bundle_info_bytes,
           offsetof(dranzer_bundle_info_t, input_bytes) == 32);
ABI_ASSERT(bundle_info_reserved,
           offsetof(dranzer_bundle_info_t, reserved) == 40);

static dranzer_status_t (*const bundle_load_v1)(
    const char *, dranzer_load_mode_t, dranzer_model_t **,
    dranzer_tokenizer_t **, dranzer_bundle_info_t *) = dranzer_bundle_load;
static uint32_t (*const api_version_v1)(void) = dranzer_api_version;
static const char *(*const version_string_v1)(void) = dranzer_version_string;
static void (*const model_free_v1)(dranzer_model_t *) = dranzer_model_free;
static void (*const tokenizer_free_v1)(dranzer_tokenizer_t *) =
    dranzer_tokenizer_free;
static size_t (*const vocab_size_v1)(const dranzer_model_t *) =
    dranzer_model_vocab_size;
static size_t (*const max_sequence_v1)(const dranzer_model_t *) =
    dranzer_model_max_sequence;
static dranzer_status_t (*const tokenize_v1)(
    dranzer_tokenizer_t *, const void *, size_t, uint32_t *, size_t *) =
    dranzer_tokenize;
static dranzer_status_t (*const detokenize_v1)(
    dranzer_tokenizer_t *, const uint32_t *, size_t, void *, size_t *) =
    dranzer_detokenize;
static dranzer_status_t (*const forward_v1)(
    dranzer_model_t *, const uint32_t *, size_t, float *, size_t) =
    dranzer_model_forward;
static dranzer_status_t (*const cache_create_v1)(
    dranzer_model_t *, size_t, dranzer_cache_t **) = dranzer_cache_create;
static void (*const cache_reset_v1)(dranzer_cache_t *) = dranzer_cache_reset;
static void (*const cache_free_v1)(dranzer_cache_t *) = dranzer_cache_free;
static dranzer_status_t (*const cache_forward_v1)(
    dranzer_cache_t *, uint32_t, float *, size_t) = dranzer_cache_forward;
static dranzer_status_t (*const generation_create_v1)(
    dranzer_model_t *, dranzer_tokenizer_t *, size_t,
    dranzer_generation_t **) = dranzer_generation_create;
static void (*const generation_free_v1)(dranzer_generation_t *) =
    dranzer_generation_free;
static dranzer_status_t (*const generation_reset_v1)(
    dranzer_generation_t *, const void *, size_t) = dranzer_generation_reset;
static dranzer_status_t (*const generation_next_v1)(
    dranzer_generation_t *, uint32_t *, void *, size_t *) =
    dranzer_generation_next_greedy;
static const char *(*const status_string_v1)(dranzer_status_t) =
    dranzer_status_string;

int main(void) {
    static const char *const status_text[] = {
        "success", "invalid argument", "I/O error", "invalid format",
        "checksum mismatch", "unsupported", "out of memory",
        "buffer too small", "model error", "generation finished",
    };
    int failed = api_version_v1() != DRANZER_API_VERSION ||
                 strcmp(version_string_v1(), DRANZER_VERSION_STRING) != 0;
    for (int i = DRANZER_OK; i <= DRANZER_FINISHED; i++) {
        if (strcmp(dranzer_status_string((dranzer_status_t)i),
                   status_text[i]) != 0) failed = 1;
    }

    dranzer_bundle_info_t too_small = DRANZER_BUNDLE_INFO_INIT;
    too_small.struct_size = DRANZER_BUNDLE_INFO_V1_SIZE - 1;
    dranzer_model_t *model = (dranzer_model_t *)(uintptr_t)1;
    dranzer_tokenizer_t *tokenizer = (dranzer_tokenizer_t *)(uintptr_t)1;
    if (bundle_load_v1("unused", DRANZER_LOAD_COPY, &model, &tokenizer,
                       &too_small) != DRANZER_INVALID_ARGUMENT ||
        model != NULL || tokenizer != NULL) {
        failed = 1;
    }

    /* Referencing every v1 signature above turns any prototype drift into a
     * compile failure without needing live model handles in this ABI test. */
    (void)model_free_v1;
    (void)tokenizer_free_v1;
    (void)vocab_size_v1;
    (void)max_sequence_v1;
    (void)tokenize_v1;
    (void)detokenize_v1;
    (void)forward_v1;
    (void)cache_create_v1;
    (void)cache_reset_v1;
    (void)cache_free_v1;
    (void)cache_forward_v1;
    (void)generation_create_v1;
    (void)generation_free_v1;
    (void)generation_reset_v1;
    (void)generation_next_v1;
    (void)status_string_v1;

    printf("%s\n", failed ? "PUBLIC COMPATIBILITY CHECK FAILED"
                           : "PUBLIC COMPATIBILITY CHECK PASSED");
    return failed ? 1 : 0;
}
