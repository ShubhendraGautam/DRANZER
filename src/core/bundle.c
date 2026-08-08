#include "core/bundle.h"
#include "core/model.h"
#include "core/fingerprint.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define BUNDLE_MAGIC "DRNZBNDL"
#define BUNDLE_FOOTER "DRNZDONE"
#define BUNDLE_MARKER UINT32_C(0x01020304)
#define BUNDLE_VERSION UINT32_C(1)
#define BUNDLE_NUMERIC_FLOAT32 UINT32_C(1)
#define BUNDLE_HEADER_SIZE UINT32_C(152)

static int float32_supported(void) {
    return sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
           FLT_MAX_EXP == 128;
}

static void encode_u32(uint8_t bytes[4], uint32_t value) {
    for (size_t i = 0; i < 4; i++) bytes[i] = (uint8_t)(value >> (8 * i));
}

static void encode_u64(uint8_t bytes[8], uint64_t value) {
    for (size_t i = 0; i < 8; i++) bytes[i] = (uint8_t)(value >> (8 * i));
}

static uint32_t decode_u32(const uint8_t bytes[4]) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; i++) value |= (uint32_t)bytes[i] << (8 * i);
    return value;
}

static uint64_t decode_u64(const uint8_t bytes[8]) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; i++) value |= (uint64_t)bytes[i] << (8 * i);
    return value;
}

/* The bundle's checksums are the shared FNV-1a from core/fingerprint.h. They
 * used to be a private copy here; three other callers needed the identical
 * number, and two copies of a hash is one copy too many. Same constants, same
 * byte order, so bundles written by earlier versions still validate. */
static uint64_t checksum_buffer(const void *data, size_t size) {
    return dranzer_fnv1a(DRANZER_FNV_OFFSET, data, size);
}

static int write_bytes(FILE *file, const void *data, size_t size) {
    return fwrite(data, 1, size, file) == size;
}

static int read_bytes(FILE *file, void *data, size_t size) {
    return fread(data, 1, size, file) == size;
}

static int write_u32(FILE *file, uint32_t value) {
    uint8_t bytes[4];
    encode_u32(bytes, value);
    return write_bytes(file, bytes, sizeof(bytes));
}


static int checked_file_size(uint64_t weights_bytes, uint64_t tokenizer_bytes,
                             uint64_t *out_size) {
    uint64_t size = BUNDLE_HEADER_SIZE;
    if (weights_bytes > UINT64_MAX - size) return 0;
    size += weights_bytes;
    if (tokenizer_bytes > UINT64_MAX - size) return 0;
    size += tokenizer_bytes;
    if (8 > UINT64_MAX - size) return 0;
    *out_size = size + 8;
    return 1;
}

static int checked_u64_multiply(uint64_t left, uint64_t right, uint64_t *out) {
    if (left != 0 && right > UINT64_MAX / left) return 0;
    *out = left * right;
    return 1;
}

static int checked_u64_add(uint64_t left, uint64_t right, uint64_t *out) {
    if (right > UINT64_MAX - left) return 0;
    *out = left + right;
    return 1;
}

/* Validate both the serialized parameter count and the largest quadratic
 * activation cache before model_new allocates anything. The workspace
 * ratio rejects tiny artifacts claiming enormous context windows while
 * remaining generous for ordinary transformer shapes. */
static int bundle_shape_valid(const uint64_t dims[6]) {
    uint64_t vocab = dims[0], embedding = dims[1], heads = dims[2];
    uint64_t layers = dims[3], sequence = dims[4], total = dims[5];
    uint64_t emb2, vocab_emb, global, layer_square, layer_linear;
    uint64_t layer_params, all_layers, computed_total, sequence_square;
    uint64_t attention_cache, workspace_limit;
    if (vocab == 0 || embedding == 0 || heads == 0 || layers == 0 ||
        sequence == 0 || embedding % heads != 0 ||
        !checked_u64_multiply(embedding, embedding, &emb2) ||
        !checked_u64_multiply(vocab, embedding, &vocab_emb) ||
        !checked_u64_multiply(vocab_emb, 2, &global) ||
        !checked_u64_add(global, vocab, &global) ||
        !checked_u64_multiply(emb2, 12, &layer_square) ||
        !checked_u64_multiply(embedding, 9, &layer_linear) ||
        !checked_u64_add(layer_square, layer_linear, &layer_params) ||
        !checked_u64_multiply(layers, layer_params, &all_layers) ||
        !checked_u64_add(global, all_layers, &computed_total) ||
        computed_total != total ||
        !checked_u64_multiply(sequence, sequence, &sequence_square) ||
        !checked_u64_multiply(heads, sequence_square, &attention_cache) ||
        !checked_u64_multiply(total, 64, &workspace_limit) ||
        attention_cache > workspace_limit) return 0;
    return 1;
}

bundle_errors_t model_bundle_save(const neural_model_t *model,
                                  const bpe_encoder_t *encoder,
                                  const model_bundle_metadata_t *metadata,
                                  const char *filename) {
    if (!model || !encoder || !metadata || !filename || !filename[0] ||
        !model->params || encoder->max_vocab_size != model->vocab_size ||
        !bpe_encoder_is_frozen(encoder)) return BUNDLE_FORMAT_ERROR;
    if (!float32_supported()) return BUNDLE_UNSUPPORTED;
    uint64_t dims[6] = {
        model->vocab_size, model->embedding_dim, model->num_heads,
        model->num_layers, model->max_seq_len, model->total_param_count,
    };
    if (!bundle_shape_valid(dims) || model->total_param_count > UINT64_MAX / 4 ||
        metadata->train_window == 0 || metadata->train_window > model->max_seq_len) {
        return BUNDLE_FORMAT_ERROR;
    }

    uint8_t *tokenizer_data = NULL;
    size_t tokenizer_size = 0;
    bpe_errors_t tokenizer_rc = bpe_encoder_serialize_portable(
        encoder, &tokenizer_data, &tokenizer_size);
    if (tokenizer_rc == BPE_ALLOCATION_FAILURE) return BUNDLE_ALLOCATION_FAILURE;
    if (tokenizer_rc != BPE_SUCCESS) return BUNDLE_FORMAT_ERROR;

    uint64_t weight_bytes = (uint64_t)model->total_param_count * 4;
    uint64_t weight_checksum = dranzer_weights_fingerprint(model);
    uint64_t tokenizer_checksum = checksum_buffer(tokenizer_data, tokenizer_size);
    uint32_t loss_bits = 0;
    memcpy(&loss_bits, &model->current_loss, sizeof(loss_bits));

    uint8_t header[BUNDLE_HEADER_SIZE] = {0};
    memcpy(header, BUNDLE_MAGIC, 8);
    encode_u32(header + 8, BUNDLE_VERSION);
    encode_u32(header + 12, BUNDLE_MARKER);
    encode_u32(header + 16, BUNDLE_NUMERIC_FLOAT32);
    encode_u32(header + 20, BUNDLE_HEADER_SIZE);
    encode_u64(header + 24, model->vocab_size);
    encode_u64(header + 32, model->embedding_dim);
    encode_u64(header + 40, model->num_heads);
    encode_u64(header + 48, model->num_layers);
    encode_u64(header + 56, model->max_seq_len);
    encode_u64(header + 64, model->total_param_count);
    encode_u32(header + 72, model->training_steps);
    encode_u32(header + 80, loss_bits);
    encode_u64(header + 88, metadata->train_window);
    encode_u64(header + 96, metadata->seed);
    encode_u64(header + 104, metadata->input_fingerprint);
    encode_u64(header + 112, metadata->input_bytes);
    encode_u64(header + 120, weight_bytes);
    encode_u64(header + 128, tokenizer_size);
    encode_u64(header + 136, weight_checksum);
    encode_u64(header + 144, tokenizer_checksum);
    uint64_t header_checksum = checksum_buffer(header, sizeof(header));
    encode_u32(header + 76, (uint32_t)header_checksum);
    encode_u32(header + 84, (uint32_t)(header_checksum >> 32));

    char temporary[2080];
    int temporary_length = snprintf(temporary, sizeof(temporary), "%s.tmp.%ld",
                                    filename, (long)getpid());
    if (temporary_length < 0 || (size_t)temporary_length >= sizeof(temporary)) {
        free(tokenizer_data);
        return BUNDLE_IO_ERROR;
    }
    FILE *file = fopen(temporary, "wb");
    if (!file) {
        free(tokenizer_data);
        return BUNDLE_IO_ERROR;
    }

    int ok = write_bytes(file, header, sizeof(header));

    for (size_t i = 0; ok && i < model->total_param_count; i++) {
        uint32_t bits = 0;
        memcpy(&bits, &model->params[i], sizeof(bits));
        ok = write_u32(file, bits);
    }
    ok = ok && write_bytes(file, tokenizer_data, tokenizer_size) &&
         write_bytes(file, BUNDLE_FOOTER, 8);
    free(tokenizer_data);
    if (ok) ok = fflush(file) == 0;
    if (ok) ok = fsync(fileno(file)) == 0;
    if (fclose(file) != 0) ok = 0;
    if (!ok || rename(temporary, filename) != 0) {
        remove(temporary);
        return BUNDLE_IO_ERROR;
    }
    return BUNDLE_SUCCESS;
}

bundle_errors_t model_bundle_load(neural_model_t *model,
                                  bpe_encoder_t **out_encoder,
                                  model_bundle_metadata_t *out_metadata,
                                  const char *filename) {
    if (!model || !out_encoder || !out_metadata || !filename) return BUNDLE_FORMAT_ERROR;
    *out_encoder = NULL;
    memset(out_metadata, 0, sizeof(*out_metadata));
    FILE *file = fopen(filename, "rb");
    if (!file) return BUNDLE_IO_ERROR;

    uint8_t header[BUNDLE_HEADER_SIZE];
    if (!read_bytes(file, header, 8) ||
        memcmp(header, BUNDLE_MAGIC, 8) != 0) {
        fclose(file);
        return BUNDLE_NOT_BUNDLE;
    }
    if (!read_bytes(file, header + 8, sizeof(header) - 8)) {
        fclose(file);
        return BUNDLE_FORMAT_ERROR;
    }
    uint64_t stored_header_checksum =
        (uint64_t)decode_u32(header + 76) |
        ((uint64_t)decode_u32(header + 84) << 32);
    memset(header + 76, 0, 4);
    memset(header + 84, 0, 4);
    if (checksum_buffer(header, sizeof(header)) != stored_header_checksum) {
        fclose(file);
        return BUNDLE_CHECKSUM_ERROR;
    }

    uint32_t version = decode_u32(header + 8);
    uint32_t marker = decode_u32(header + 12);
    uint32_t numeric = decode_u32(header + 16);
    uint32_t header_size = decode_u32(header + 20);
    uint32_t training_steps = decode_u32(header + 72);
    uint32_t loss_bits = decode_u32(header + 80);
    uint64_t dims[6];
    for (size_t i = 0; i < 6; i++) dims[i] = decode_u64(header + 24 + i * 8);
    uint64_t train_window = decode_u64(header + 88);
    uint64_t seed = decode_u64(header + 96);
    uint64_t input_fingerprint = decode_u64(header + 104);
    uint64_t input_bytes = decode_u64(header + 112);
    uint64_t weight_bytes = decode_u64(header + 120);
    uint64_t tokenizer_bytes = decode_u64(header + 128);
    uint64_t expected_weight_checksum = decode_u64(header + 136);
    uint64_t expected_tokenizer_checksum = decode_u64(header + 144);
    if (version != BUNDLE_VERSION || marker != BUNDLE_MARKER ||
        numeric != BUNDLE_NUMERIC_FLOAT32 || header_size != BUNDLE_HEADER_SIZE ||
        !float32_supported()) {
        fclose(file);
        return BUNDLE_UNSUPPORTED;
    }
    for (size_t i = 0; i < 6; i++) {
        if (dims[i] > SIZE_MAX) { fclose(file); return BUNDLE_FORMAT_ERROR; }
    }
    if (!bundle_shape_valid(dims) || dims[5] > UINT64_MAX / 4 ||
        weight_bytes != dims[5] * 4 ||
        tokenizer_bytes < 24 || tokenizer_bytes > SIZE_MAX || train_window > SIZE_MAX) {
        fclose(file);
        return BUNDLE_FORMAT_ERROR;
    }
    uint64_t expected_file_size = 0;
    struct stat status;
    if (!checked_file_size(weight_bytes, tokenizer_bytes, &expected_file_size) ||
        fstat(fileno(file), &status) != 0 || status.st_size < 0 ||
        (uint64_t)status.st_size != expected_file_size) {
        fclose(file);
        return BUNDLE_FORMAT_ERROR;
    }

    neural_model_t loaded = {0};
    model_errors_t model_rc = model_new(
        &loaded, (size_t)dims[0], (size_t)dims[1], (size_t)dims[2],
        (size_t)dims[3], (size_t)dims[4]);
    if (model_rc != MODEL_SUCCESS || loaded.total_param_count != (size_t)dims[5]) {
        model_free(&loaded);
        fclose(file);
        return model_rc == MODEL_ALLOCATION_FAILURE
                   ? BUNDLE_ALLOCATION_FAILURE : BUNDLE_FORMAT_ERROR;
    }
    uint64_t actual_weight_checksum = DRANZER_FNV_OFFSET;
    for (size_t i = 0; i < loaded.total_param_count; i++) {
        uint8_t bytes[4];
        if (!read_bytes(file, bytes, sizeof(bytes))) {
            model_free(&loaded);
            fclose(file);
            return BUNDLE_FORMAT_ERROR;
        }
        actual_weight_checksum = dranzer_fnv1a(actual_weight_checksum, bytes, sizeof(bytes));
        uint32_t bits = decode_u32(bytes);
        memcpy(&loaded.params[i], &bits, sizeof(bits));
    }
    if (actual_weight_checksum != expected_weight_checksum) {
        model_free(&loaded);
        fclose(file);
        return BUNDLE_CHECKSUM_ERROR;
    }

    uint8_t *tokenizer_data = malloc((size_t)tokenizer_bytes);
    if (!tokenizer_data) {
        model_free(&loaded);
        fclose(file);
        return BUNDLE_ALLOCATION_FAILURE;
    }
    char footer[8];
    if (!read_bytes(file, tokenizer_data, (size_t)tokenizer_bytes) ||
        !read_bytes(file, footer, sizeof(footer)) ||
        memcmp(footer, BUNDLE_FOOTER, sizeof(footer)) != 0 || fgetc(file) != EOF) {
        free(tokenizer_data);
        model_free(&loaded);
        fclose(file);
        return BUNDLE_FORMAT_ERROR;
    }
    fclose(file);
    if (checksum_buffer(tokenizer_data, (size_t)tokenizer_bytes) !=
        expected_tokenizer_checksum) {
        free(tokenizer_data);
        model_free(&loaded);
        return BUNDLE_CHECKSUM_ERROR;
    }
    /* Reject a mismatched or malicious allocation request before asking the
     * tokenizer decoder to allocate its vocabulary table. */
    if (decode_u64(tokenizer_data) != dims[0]) {
        free(tokenizer_data);
        model_free(&loaded);
        return BUNDLE_FORMAT_ERROR;
    }
    bpe_encoder_t *encoder = calloc(1, sizeof(*encoder));
    if (!encoder) {
        free(tokenizer_data);
        model_free(&loaded);
        return BUNDLE_ALLOCATION_FAILURE;
    }
    bpe_errors_t tokenizer_rc = bpe_encoder_deserialize_portable(
        encoder, tokenizer_data, (size_t)tokenizer_bytes);
    free(tokenizer_data);
    if (tokenizer_rc != BPE_SUCCESS || encoder->max_vocab_size != loaded.vocab_size) {
        bpe_encoder_free(encoder);
        free(encoder);
        model_free(&loaded);
        return tokenizer_rc == BPE_ALLOCATION_FAILURE
                   ? BUNDLE_ALLOCATION_FAILURE : BUNDLE_FORMAT_ERROR;
    }

    loaded.training_steps = training_steps;
    memcpy(&loaded.current_loss, &loss_bits, sizeof(loss_bits));
    out_metadata->train_window = (size_t)train_window;
    out_metadata->seed = seed;
    out_metadata->input_fingerprint = input_fingerprint;
    out_metadata->input_bytes = input_bytes;
    *model = loaded;
    *out_encoder = encoder;
    return BUNDLE_SUCCESS;
}
