#include "dranzer.h"
#include "byte_pair_encoding.h"
#include "core/bundle.h"
#include "core/model.h"
#include <float.h>
#include <stdlib.h>
#include <string.h>

struct dranzer_model {
    neural_model_t value;
    size_t references;
};

struct dranzer_tokenizer {
    bpe_encoder_t value;
    size_t references;
};

struct dranzer_cache {
    dranzer_model_t *model;
    model_kv_cache_t value;
};

struct dranzer_generation {
    dranzer_model_t *model;
    dranzer_tokenizer_t *tokenizer;
    model_kv_cache_t cache;
    float *logits;
    size_t capacity;
    int ready;
    int finished;
};

uint32_t dranzer_api_version(void) {
    return DRANZER_API_VERSION;
}

const char *dranzer_version_string(void) {
    return DRANZER_VERSION_STRING;
}

static void model_retain(dranzer_model_t *model) {
    model->references++;
}

static void tokenizer_retain(dranzer_tokenizer_t *tokenizer) {
    tokenizer->references++;
}

void dranzer_model_free(dranzer_model_t *model) {
    if (!model) return;
    if (--model->references != 0) return;
    model_free(&model->value);
    free(model);
}

void dranzer_tokenizer_free(dranzer_tokenizer_t *tokenizer) {
    if (!tokenizer) return;
    if (--tokenizer->references != 0) return;
    bpe_encoder_free(&tokenizer->value);
    free(tokenizer);
}

static dranzer_status_t bundle_status(bundle_errors_t status) {
    switch (status) {
        case BUNDLE_SUCCESS: return DRANZER_OK;
        case BUNDLE_IO_ERROR: return DRANZER_IO_ERROR;
        case BUNDLE_CHECKSUM_ERROR: return DRANZER_CHECKSUM_ERROR;
        case BUNDLE_UNSUPPORTED: return DRANZER_UNSUPPORTED;
        case BUNDLE_ALLOCATION_FAILURE: return DRANZER_OUT_OF_MEMORY;
        case BUNDLE_NOT_BUNDLE:
        case BUNDLE_FORMAT_ERROR: return DRANZER_FORMAT_ERROR;
        default: return DRANZER_MODEL_ERROR;
    }
}

static dranzer_status_t model_status(model_errors_t status) {
    if (status == MODEL_SUCCESS) return DRANZER_OK;
    if (status == MODEL_INVALID_INPUT) return DRANZER_INVALID_ARGUMENT;
    if (status == MODEL_ALLOCATION_FAILURE) return DRANZER_OUT_OF_MEMORY;
    return DRANZER_MODEL_ERROR;
}

dranzer_status_t dranzer_bundle_load(const char *path,
                                     dranzer_load_mode_t mode,
                                     dranzer_model_t **out_model,
                                     dranzer_tokenizer_t **out_tokenizer,
                                     dranzer_bundle_info_t *out_info) {
    if (!out_model || !out_tokenizer) return DRANZER_INVALID_ARGUMENT;
    *out_model = NULL;
    *out_tokenizer = NULL;
    if (out_info) {
        uint32_t caller_size = out_info->struct_size;
        if (caller_size < DRANZER_BUNDLE_INFO_V1_SIZE) {
            return DRANZER_INVALID_ARGUMENT;
        }
        size_t clear_size = caller_size < sizeof(*out_info)
                                ? caller_size : sizeof(*out_info);
        memset(out_info, 0, clear_size);
        out_info->struct_size = (uint32_t)sizeof(*out_info);
    }
    if (!path || !path[0] ||
        (mode != DRANZER_LOAD_COPY && mode != DRANZER_LOAD_MMAP)) {
        return DRANZER_INVALID_ARGUMENT;
    }

    neural_model_t loaded = {0};
    bpe_encoder_t *encoder = NULL;
    model_bundle_metadata_t metadata = {0};
    bundle_errors_t rc = mode == DRANZER_LOAD_MMAP
        ? model_bundle_load_mmap(&loaded, &encoder, &metadata, path)
        : model_bundle_load(&loaded, &encoder, &metadata, path);
    if (rc != BUNDLE_SUCCESS) return bundle_status(rc);

    dranzer_model_t *model = calloc(1, sizeof(*model));
    dranzer_tokenizer_t *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!model || !tokenizer) {
        free(model);
        free(tokenizer);
        model_free(&loaded);
        bpe_encoder_free(encoder);
        free(encoder);
        return DRANZER_OUT_OF_MEMORY;
    }
    model->value = loaded;
    model->references = 1;
    tokenizer->value = *encoder;
    tokenizer->references = 1;
    free(encoder);

    if (out_info) {
        out_info->format_version = metadata.format_version;
        out_info->train_window = metadata.train_window;
        out_info->seed = metadata.seed;
        out_info->input_fingerprint = metadata.input_fingerprint;
        out_info->input_bytes = metadata.input_bytes;
    }
    *out_model = model;
    *out_tokenizer = tokenizer;
    return DRANZER_OK;
}

size_t dranzer_model_vocab_size(const dranzer_model_t *model) {
    return model ? model->value.vocab_size : 0;
}

size_t dranzer_model_max_sequence(const dranzer_model_t *model) {
    return model ? model->value.max_seq_len : 0;
}

dranzer_status_t dranzer_tokenize(dranzer_tokenizer_t *tokenizer,
                                  const void *bytes, size_t byte_count,
                                  uint32_t *token_ids,
                                  size_t *inout_count) {
    if (!tokenizer || (!bytes && byte_count != 0) || !inout_count) {
        return DRANZER_INVALID_ARGUMENT;
    }
    if (byte_count == 0) {
        *inout_count = 0;
        return DRANZER_OK;
    }
    static const char empty[] = "";
    bpe_tokens_t encoded = {0};
    bpe_errors_t rc = bpe_encode(&tokenizer->value,
                                 bytes ? (const char *)bytes : empty,
                                 byte_count, &encoded);
    if (rc != BPE_SUCCESS) {
        return rc == BPE_ALLOCATION_FAILURE
                   ? DRANZER_OUT_OF_MEMORY : DRANZER_FORMAT_ERROR;
    }
    size_t required = encoded.token_count;
    if ((!token_ids && required != 0) || *inout_count < required) {
        *inout_count = required;
        bpe_tokens_free(&encoded);
        return DRANZER_BUFFER_TOO_SMALL;
    }
    if (required != 0) {
        memcpy(token_ids, encoded.token_ids, required * sizeof(*token_ids));
    }
    *inout_count = required;
    bpe_tokens_free(&encoded);
    return DRANZER_OK;
}

dranzer_status_t dranzer_detokenize(dranzer_tokenizer_t *tokenizer,
                                    const uint32_t *token_ids,
                                    size_t token_count,
                                    void *bytes,
                                    size_t *inout_size) {
    if (!tokenizer || (!token_ids && token_count != 0) || !inout_size) {
        return DRANZER_INVALID_ARGUMENT;
    }
    if (token_count == 0) {
        *inout_size = 0;
        return DRANZER_OK;
    }
    char *decoded = NULL;
    size_t decoded_size = 0;
    bpe_errors_t rc = bpe_decode(&tokenizer->value, token_ids, token_count,
                                 &decoded, &decoded_size);
    if (rc != BPE_SUCCESS) {
        free(decoded);
        return rc == BPE_ALLOCATION_FAILURE
                   ? DRANZER_OUT_OF_MEMORY : DRANZER_FORMAT_ERROR;
    }
    if ((!bytes && decoded_size != 0) || *inout_size < decoded_size) {
        *inout_size = decoded_size;
        free(decoded);
        return DRANZER_BUFFER_TOO_SMALL;
    }
    if (decoded_size != 0) memcpy(bytes, decoded, decoded_size);
    *inout_size = decoded_size;
    free(decoded);
    return DRANZER_OK;
}

dranzer_status_t dranzer_model_forward(dranzer_model_t *model,
                                       const uint32_t *token_ids,
                                       size_t token_count,
                                       float *logits,
                                       size_t logits_count) {
    if (!model || !token_ids || !logits) return DRANZER_INVALID_ARGUMENT;
    if (logits_count < model->value.vocab_size) return DRANZER_BUFFER_TOO_SMALL;
    return model_status(model_forward(&model->value, (uint32_t *)token_ids,
                                      token_count, logits));
}

dranzer_status_t dranzer_cache_create(dranzer_model_t *model,
                                      size_t capacity,
                                      dranzer_cache_t **out_cache) {
    if (!out_cache) return DRANZER_INVALID_ARGUMENT;
    *out_cache = NULL;
    if (!model) return DRANZER_INVALID_ARGUMENT;
    if (capacity == 0) capacity = model->value.max_seq_len;
    dranzer_cache_t *cache = calloc(1, sizeof(*cache));
    if (!cache) return DRANZER_OUT_OF_MEMORY;
    model_errors_t rc = model_kv_cache_init_with_capacity(
        &cache->value, &model->value, capacity);
    if (rc != MODEL_SUCCESS) {
        free(cache);
        return model_status(rc);
    }
    cache->model = model;
    model_retain(model);
    *out_cache = cache;
    return DRANZER_OK;
}

void dranzer_cache_reset(dranzer_cache_t *cache) {
    if (cache) model_kv_cache_reset(&cache->value);
}

void dranzer_cache_free(dranzer_cache_t *cache) {
    if (!cache) return;
    model_kv_cache_free(&cache->value);
    dranzer_model_free(cache->model);
    free(cache);
}

dranzer_status_t dranzer_cache_forward(dranzer_cache_t *cache,
                                       uint32_t token_id,
                                       float *logits,
                                       size_t logits_count) {
    if (!cache || !logits) return DRANZER_INVALID_ARGUMENT;
    if (logits_count < cache->model->value.vocab_size) {
        return DRANZER_BUFFER_TOO_SMALL;
    }
    return model_status(model_forward_token(&cache->model->value,
                                            &cache->value, token_id, logits));
}

dranzer_status_t dranzer_generation_create(
    dranzer_model_t *model, dranzer_tokenizer_t *tokenizer,
    size_t context_capacity, dranzer_generation_t **out_generation) {
    if (!out_generation) return DRANZER_INVALID_ARGUMENT;
    *out_generation = NULL;
    if (!model || !tokenizer ||
        tokenizer->value.max_vocab_size != model->value.vocab_size) {
        return DRANZER_INVALID_ARGUMENT;
    }
    if (context_capacity == 0) context_capacity = model->value.max_seq_len;
    dranzer_generation_t *generation = calloc(1, sizeof(*generation));
    if (!generation) return DRANZER_OUT_OF_MEMORY;
    generation->logits = malloc(model->value.vocab_size * sizeof(float));
    if (!generation->logits) {
        free(generation);
        return DRANZER_OUT_OF_MEMORY;
    }
    model_errors_t rc = model_kv_cache_init_with_capacity(
        &generation->cache, &model->value, context_capacity);
    if (rc != MODEL_SUCCESS) {
        free(generation->logits);
        free(generation);
        return model_status(rc);
    }
    generation->model = model;
    generation->tokenizer = tokenizer;
    generation->capacity = context_capacity;
    model_retain(model);
    tokenizer_retain(tokenizer);
    *out_generation = generation;
    return DRANZER_OK;
}

void dranzer_generation_free(dranzer_generation_t *generation) {
    if (!generation) return;
    model_kv_cache_free(&generation->cache);
    free(generation->logits);
    dranzer_model_free(generation->model);
    dranzer_tokenizer_free(generation->tokenizer);
    free(generation);
}

dranzer_status_t dranzer_generation_reset(dranzer_generation_t *generation,
                                           const void *prompt,
                                           size_t prompt_size) {
    if (!generation || (!prompt && prompt_size != 0)) {
        return DRANZER_INVALID_ARGUMENT;
    }
    bpe_tokens_t encoded = {0};
    bpe_errors_t encode_rc = prompt_size == 0 ? BPE_SUCCESS : bpe_encode(
        &generation->tokenizer->value, (const char *)prompt,
        prompt_size, &encoded);
    if (encode_rc != BPE_SUCCESS) {
        return encode_rc == BPE_ALLOCATION_FAILURE
                   ? DRANZER_OUT_OF_MEMORY : DRANZER_FORMAT_ERROR;
    }

    const int has_bos = bpe_encoder_has_special_tokens(
        &generation->tokenizer->value);
    size_t available = generation->capacity - (has_bos ? 1u : 0u);
    size_t retained = encoded.token_count < available
                          ? encoded.token_count : available;
    if (!has_bos && retained == 0) {
        bpe_tokens_free(&encoded);
        return DRANZER_INVALID_ARGUMENT;
    }

    model_kv_cache_reset(&generation->cache);
    model_errors_t model_rc = MODEL_SUCCESS;
    if (has_bos) {
        model_rc = model_forward_token(
            &generation->model->value, &generation->cache,
            BPE_BOS_TOKEN_ID, generation->logits);
    }
    size_t start = encoded.token_count - retained;
    for (size_t i = start; model_rc == MODEL_SUCCESS &&
                           i < encoded.token_count; i++) {
        model_rc = model_forward_token(
            &generation->model->value, &generation->cache,
            encoded.token_ids[i], generation->logits);
    }
    bpe_tokens_free(&encoded);
    if (model_rc != MODEL_SUCCESS) return model_status(model_rc);
    generation->ready = 1;
    generation->finished = 0;
    return DRANZER_OK;
}

dranzer_status_t dranzer_generation_next_greedy(
    dranzer_generation_t *generation, uint32_t *out_token_id,
    void *piece, size_t *inout_piece_size) {
    if (!generation || !out_token_id || !inout_piece_size ||
        !generation->ready) return DRANZER_INVALID_ARGUMENT;
    if (generation->finished) {
        *inout_piece_size = 0;
        return DRANZER_FINISHED;
    }

    bpe_encoder_t *encoder = &generation->tokenizer->value;
    neural_model_t *model = &generation->model->value;
    if (bpe_encoder_has_special_tokens(encoder)) {
        const uint32_t excluded[] = {
            BPE_PAD_TOKEN_ID, BPE_UNK_TOKEN_ID, BPE_BOS_TOKEN_ID,
        };
        for (size_t i = 0; i < sizeof(excluded) / sizeof(excluded[0]); i++) {
            if (excluded[i] < model->vocab_size) {
                generation->logits[excluded[i]] = -FLT_MAX;
            }
        }
    }
    for (size_t i = encoder->vocab_size; i < model->vocab_size; i++) {
        generation->logits[i] = -FLT_MAX;
    }

    uint32_t next = 0;
    for (uint32_t i = 1; i < model->vocab_size; i++) {
        if (generation->logits[i] > generation->logits[next]) next = i;
    }
    const int eos = bpe_encoder_has_special_tokens(encoder) &&
                    next == BPE_EOS_TOKEN_ID;
    size_t required = 0;
    const char *token_bytes = NULL;
    if (!eos && next < encoder->vocab_size &&
        !bpe_token_is_control(encoder, next)) {
        token_bytes = encoder->tokens[next].token;
        required = encoder->tokens[next].length;
    }
    if ((!piece && required != 0) || *inout_piece_size < required) {
        *inout_piece_size = required;
        return DRANZER_BUFFER_TOO_SMALL;
    }
    if (required != 0) memcpy(piece, token_bytes, required);
    *inout_piece_size = required;
    *out_token_id = next;
    if (eos) {
        generation->finished = 1;
        return DRANZER_FINISHED;
    }

    model_errors_t rc = model_forward_token(model, &generation->cache, next,
                                             generation->logits);
    return model_status(rc);
}

const char *dranzer_status_string(dranzer_status_t status) {
    switch (status) {
        case DRANZER_OK: return "success";
        case DRANZER_INVALID_ARGUMENT: return "invalid argument";
        case DRANZER_IO_ERROR: return "I/O error";
        case DRANZER_FORMAT_ERROR: return "invalid format";
        case DRANZER_CHECKSUM_ERROR: return "checksum mismatch";
        case DRANZER_UNSUPPORTED: return "unsupported";
        case DRANZER_OUT_OF_MEMORY: return "out of memory";
        case DRANZER_BUFFER_TOO_SMALL: return "buffer too small";
        case DRANZER_MODEL_ERROR: return "model error";
        case DRANZER_FINISHED: return "generation finished";
        default: return "unknown status";
    }
}
