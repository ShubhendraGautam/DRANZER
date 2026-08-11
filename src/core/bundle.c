#include "core/bundle.h"
#include "core/model.h"
#include "core/fingerprint.h"
#include "core/model_params.h"
#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define BUNDLE_MAGIC "DRNZBNDL"
#define BUNDLE_FOOTER "DRNZDONE"
#define BUNDLE_MARKER UINT32_C(0x01020304)
#define BUNDLE_NUMERIC_FLOAT32 UINT32_C(1)
#define BUNDLE_NUMERIC_QUANTIZED UINT32_C(2)
#define BUNDLE_HEADER_V1_SIZE UINT32_C(152)
#define BUNDLE_HEADER_V2_SIZE UINT32_C(184)
#define BUNDLE_HEADER_V3_SIZE UINT32_C(192)
#define BUNDLE_HEADER_MAX_SIZE BUNDLE_HEADER_V3_SIZE
#define BUNDLE_TENSOR_RECORD_SIZE UINT32_C(32)
#define BUNDLE_TENSOR_FLOAT32 UINT32_C(1)
#define BUNDLE_TENSOR_QUANTIZED UINT32_C(2)
#define BUNDLE_POLICY_EMBEDDINGS UINT32_C(1)
#define BUNDLE_POLICY_BIASES_NORMS UINT32_C(2)

static int float32_supported(void) {
    return sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
           FLT_MAX_EXP == 128;
}

static int host_is_little_endian(void) {
    const uint32_t value = UINT32_C(1);
    return *(const uint8_t *)&value == 1;
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

static int checked_file_size(uint32_t header_size, uint64_t weights_bytes,
                             uint64_t tokenizer_bytes,
                             uint64_t *out_size) {
    uint64_t size = header_size;
    if (weights_bytes > UINT64_MAX - size) return 0;
    size += weights_bytes;
    if (tokenizer_bytes > UINT64_MAX - size) return 0;
    size += tokenizer_bytes;
    if (8 > UINT64_MAX - size) return 0;
    *out_size = size + 8;
    return 1;
}

static int write_checksum_bytes(FILE *file, const void *data, size_t size,
                                uint64_t *checksum) {
    if (!write_bytes(file, data, size)) return 0;
    *checksum = dranzer_fnv1a(*checksum, data, size);
    return 1;
}

static int write_checksum_u32(FILE *file, uint32_t value, uint64_t *checksum) {
    uint8_t bytes[4];
    encode_u32(bytes, value);
    return write_checksum_bytes(file, bytes, sizeof(bytes), checksum);
}

static int file_region_checksum(FILE *file, uint64_t size, uint64_t *out) {
    uint8_t buffer[8192];
    uint64_t checksum = DRANZER_FNV_OFFSET;
    while (size > 0) {
        size_t chunk = size < sizeof(buffer) ? (size_t)size : sizeof(buffer);
        if (!read_bytes(file, buffer, chunk)) return 0;
        checksum = dranzer_fnv1a(checksum, buffer, chunk);
        size -= chunk;
    }
    *out = checksum;
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
static int bundle_shape_valid(const uint64_t dims[6], uint32_t architecture_flags) {
    uint64_t vocab = dims[0], embedding = dims[1], heads = dims[2];
    uint64_t layers = dims[3], sequence = dims[4], total = dims[5];
    uint64_t emb2, vocab_emb, global, layer_square, layer_linear;
    uint64_t layer_params, all_layers, computed_total, sequence_square;
    uint64_t attention_cache, workspace_limit;
    if ((architecture_flags & ~MODEL_ARCHITECTURE_SUPPORTED_MASK) != 0 ||
        vocab == 0 || embedding == 0 || heads == 0 || layers == 0 ||
        sequence == 0 || embedding % heads != 0 ||
        !checked_u64_multiply(embedding, embedding, &emb2) ||
        !checked_u64_multiply(vocab, embedding, &vocab_emb) ||
        !checked_u64_multiply(
            vocab_emb,
            (architecture_flags & MODEL_ARCH_TIED_EMBEDDINGS) ? 1 : 2,
            &global) ||
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
    if (!bundle_shape_valid(dims, model->architecture_flags) ||
        model->total_param_count > UINT64_MAX / 4 ||
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

    uint32_t version = model->architecture_flags == 0
                           ? MODEL_BUNDLE_FORMAT_V1 : MODEL_BUNDLE_FORMAT_V3;
    uint32_t header_size = model->architecture_flags == 0
                               ? BUNDLE_HEADER_V1_SIZE : BUNDLE_HEADER_V3_SIZE;
    uint8_t header[BUNDLE_HEADER_MAX_SIZE] = {0};
    memcpy(header, BUNDLE_MAGIC, 8);
    encode_u32(header + 8, version);
    encode_u32(header + 12, BUNDLE_MARKER);
    encode_u32(header + 16, BUNDLE_NUMERIC_FLOAT32);
    encode_u32(header + 20, header_size);
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
    if (version == MODEL_BUNDLE_FORMAT_V3)
        encode_u32(header + 184, model->architecture_flags);
    uint64_t header_checksum = checksum_buffer(header, header_size);
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

    int ok = write_bytes(file, header, header_size);

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

static int quantized_config_valid(const model_quant_config_t *config) {
    return config && config->bits >= QUANT_MIN_BITS &&
           config->bits <= QUANT_MAX_BITS &&
           (int)config->granularity >= 0 &&
           (int)config->granularity < (int)QUANT_GRANULARITY_COUNT;
}

static int add_payload_bytes(uint64_t *total, uint64_t amount) {
    return checked_u64_add(*total, amount, total);
}

bundle_errors_t model_bundle_save_quantized(
    const neural_model_t *model, const bpe_encoder_t *encoder,
    const model_bundle_metadata_t *metadata,
    const model_quant_config_t *config,
    model_bundle_storage_report_t *out_report,
    const char *filename) {
    if (out_report) memset(out_report, 0, sizeof(*out_report));
    if (!model || !encoder || !metadata || !filename || !filename[0] ||
        !model->params || encoder->max_vocab_size != model->vocab_size ||
        !bpe_encoder_is_frozen(encoder) || !quantized_config_valid(config)) {
        return BUNDLE_FORMAT_ERROR;
    }
    if (!float32_supported()) return BUNDLE_UNSUPPORTED;

    uint64_t dims[6] = {
        model->vocab_size, model->embedding_dim, model->num_heads,
        model->num_layers, model->max_seq_len, model->total_param_count,
    };
    if (!bundle_shape_valid(dims, model->architecture_flags) ||
        metadata->train_window == 0 ||
        metadata->train_window > model->max_seq_len) return BUNDLE_FORMAT_ERROR;

    size_t tensor_count = model_param_tensor_count(model);
    if (tensor_count == 0 || tensor_count > UINT32_MAX ||
        tensor_count > SIZE_MAX / sizeof(param_tensor_t)) return BUNDLE_FORMAT_ERROR;
    param_tensor_t *tensors = malloc(tensor_count * sizeof(*tensors));
    if (!tensors) return BUNDLE_ALLOCATION_FAILURE;
    if (model_param_tensors(model, tensors, tensor_count) != tensor_count) {
        free(tensors);
        return BUNDLE_FORMAT_ERROR;
    }

    uint64_t weight_bytes = 0, values_quantized = 0, scales_stored = 0;
    size_t tensors_quantized = 0;
    for (size_t i = 0; i < tensor_count; i++) {
        const param_tensor_t *tensor = &tensors[i];
        if (tensor->rows == 0 || tensor->cols == 0 ||
            tensor->rows > SIZE_MAX / tensor->cols ||
            !add_payload_bytes(&weight_bytes, BUNDLE_TENSOR_RECORD_SIZE)) {
            free(tensors);
            return BUNDLE_FORMAT_ERROR;
        }
        size_t count = tensor->rows * tensor->cols;
        if (model_quantize_includes(config, tensor->kind)) {
            size_t scale_count = 0, packed_size = 0;
            uint64_t scale_bytes = 0;
            if (quantized_scale_count(tensor->rows, tensor->cols,
                                      config->granularity, &scale_count) != 0 ||
                quantized_packed_size(count, config->bits, &packed_size) != 0 ||
                !checked_u64_multiply(scale_count, 4, &scale_bytes) ||
                !add_payload_bytes(&weight_bytes, scale_bytes) ||
                !add_payload_bytes(&weight_bytes, packed_size) ||
                !checked_u64_add(values_quantized, count, &values_quantized) ||
                !checked_u64_add(scales_stored, scale_count, &scales_stored)) {
                free(tensors);
                return BUNDLE_FORMAT_ERROR;
            }
            tensors_quantized++;
        } else {
            uint64_t raw_bytes = 0;
            if (!checked_u64_multiply(count, 4, &raw_bytes) ||
                !add_payload_bytes(&weight_bytes, raw_bytes)) {
                free(tensors);
                return BUNDLE_FORMAT_ERROR;
            }
        }
    }

    uint8_t *tokenizer_data = NULL;
    size_t tokenizer_size = 0;
    bpe_errors_t tokenizer_rc = bpe_encoder_serialize_portable(
        encoder, &tokenizer_data, &tokenizer_size);
    if (tokenizer_rc != BPE_SUCCESS) {
        free(tensors);
        return tokenizer_rc == BPE_ALLOCATION_FAILURE
                   ? BUNDLE_ALLOCATION_FAILURE : BUNDLE_FORMAT_ERROR;
    }
    uint32_t version = model->architecture_flags == 0
                           ? MODEL_BUNDLE_FORMAT_V2 : MODEL_BUNDLE_FORMAT_V3;
    uint32_t header_size = model->architecture_flags == 0
                               ? BUNDLE_HEADER_V2_SIZE : BUNDLE_HEADER_V3_SIZE;
    uint64_t artifact_bytes = 0;
    if (!checked_file_size(header_size, weight_bytes,
                           tokenizer_size, &artifact_bytes)) {
        free(tokenizer_data);
        free(tensors);
        return BUNDLE_FORMAT_ERROR;
    }

    char temporary[2080];
    int temporary_length = snprintf(temporary, sizeof(temporary), "%s.tmp.%ld",
                                    filename, (long)getpid());
    if (temporary_length < 0 || (size_t)temporary_length >= sizeof(temporary)) {
        free(tokenizer_data);
        free(tensors);
        return BUNDLE_IO_ERROR;
    }
    FILE *file = fopen(temporary, "wb");
    if (!file) {
        free(tokenizer_data);
        free(tensors);
        return BUNDLE_IO_ERROR;
    }

    uint8_t empty_header[BUNDLE_HEADER_MAX_SIZE] = {0};
    uint64_t weight_checksum = DRANZER_FNV_OFFSET;
    bundle_errors_t write_error = BUNDLE_IO_ERROR;
    int ok = write_bytes(file, empty_header, header_size);
    for (size_t i = 0; ok && i < tensor_count; i++) {
        const param_tensor_t *tensor = &tensors[i];
        const int included = model_quantize_includes(config, tensor->kind);
        uint8_t record[BUNDLE_TENSOR_RECORD_SIZE] = {0};
        encode_u32(record, (uint32_t)i);
        encode_u32(record + 4, (uint32_t)tensor->kind);
        encode_u64(record + 8, tensor->rows);
        encode_u64(record + 16, tensor->cols);
        encode_u32(record + 24, included ? BUNDLE_TENSOR_QUANTIZED
                                         : BUNDLE_TENSOR_FLOAT32);
        ok = write_checksum_bytes(file, record, sizeof(record), &weight_checksum);

        size_t count = tensor->rows * tensor->cols;
        if (ok && included) {
            size_t scale_count = 0, packed_size = 0;
            quantized_scale_count(tensor->rows, tensor->cols,
                                  config->granularity, &scale_count);
            quantized_packed_size(count, config->bits, &packed_size);
            float *scales = scale_count <= SIZE_MAX / sizeof(*scales)
                          ? malloc(scale_count * sizeof(*scales)) : NULL;
            uint8_t *packed = malloc(packed_size);
            if (!scales || !packed) {
                free(scales);
                free(packed);
                write_error = BUNDLE_ALLOCATION_FAILURE;
                ok = 0;
            } else if (quantize_pack(tensor->values, tensor->rows, tensor->cols,
                                     config->bits, config->granularity,
                                     scales, scale_count, packed, packed_size) != 0) {
                write_error = BUNDLE_FORMAT_ERROR;
                ok = 0;
            } else {
                for (size_t s = 0; ok && s < scale_count; s++) {
                    uint32_t bits = 0;
                    memcpy(&bits, &scales[s], sizeof(bits));
                    ok = write_checksum_u32(file, bits, &weight_checksum);
                }
                if (ok) ok = write_checksum_bytes(file, packed, packed_size,
                                                  &weight_checksum);
            }
            free(scales);
            free(packed);
        } else if (ok) {
            for (size_t j = 0; ok && j < count; j++) {
                uint32_t bits = 0;
                memcpy(&bits, &tensor->values[j], sizeof(bits));
                ok = write_checksum_u32(file, bits, &weight_checksum);
            }
        }
    }

    uint64_t tokenizer_checksum = checksum_buffer(tokenizer_data, tokenizer_size);
    ok = ok && write_bytes(file, tokenizer_data, tokenizer_size) &&
         write_bytes(file, BUNDLE_FOOTER, 8);

    uint8_t header[BUNDLE_HEADER_MAX_SIZE] = {0};
    uint32_t loss_bits = 0;
    memcpy(&loss_bits, &model->current_loss, sizeof(loss_bits));
    memcpy(header, BUNDLE_MAGIC, 8);
    encode_u32(header + 8, version);
    encode_u32(header + 12, BUNDLE_MARKER);
    encode_u32(header + 16, BUNDLE_NUMERIC_QUANTIZED);
    encode_u32(header + 20, header_size);
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
    encode_u32(header + 152, (uint32_t)config->bits);
    encode_u32(header + 156, (uint32_t)config->granularity);
    uint32_t flags = config->include_embeddings ? BUNDLE_POLICY_EMBEDDINGS : 0;
    if (config->include_biases_and_norms) flags |= BUNDLE_POLICY_BIASES_NORMS;
    encode_u32(header + 160, flags);
    encode_u32(header + 164, (uint32_t)tensor_count);
    encode_u64(header + 168, values_quantized);
    encode_u64(header + 176, scales_stored);
    if (version == MODEL_BUNDLE_FORMAT_V3)
        encode_u32(header + 184, model->architecture_flags);
    uint64_t header_checksum = checksum_buffer(header, header_size);
    encode_u32(header + 76, (uint32_t)header_checksum);
    encode_u32(header + 84, (uint32_t)(header_checksum >> 32));

    if (ok) ok = fseek(file, 0, SEEK_SET) == 0 &&
                 write_bytes(file, header, header_size);
    free(tokenizer_data);
    free(tensors);
    if (ok) ok = fflush(file) == 0;
    if (ok) ok = fsync(fileno(file)) == 0;
    if (fclose(file) != 0) ok = 0;
    if (!ok || rename(temporary, filename) != 0) {
        remove(temporary);
        return ok ? BUNDLE_IO_ERROR : write_error;
    }
    if (out_report) {
        out_report->artifact_bytes = artifact_bytes;
        out_report->weight_payload_bytes = weight_bytes;
        out_report->tokenizer_bytes = tokenizer_size;
        out_report->tensors_quantized = tensors_quantized;
        out_report->values_quantized = (size_t)values_quantized;
        out_report->scales_stored = (size_t)scales_stored;
    }
    return BUNDLE_SUCCESS;
}

typedef struct {
    FILE *file;
    uint64_t remaining;
} weight_reader_t;

static int weight_read(weight_reader_t *reader, void *data, size_t size) {
    if ((uint64_t)size > reader->remaining ||
        !read_bytes(reader->file, data, size)) return 0;
    reader->remaining -= size;
    return 1;
}

static int weight_read_float(weight_reader_t *reader, float *out) {
    uint8_t bytes[4];
    if (!weight_read(reader, bytes, sizeof(bytes))) return 0;
    uint32_t bits = decode_u32(bytes);
    memcpy(out, &bits, sizeof(bits));
    return 1;
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

    uint8_t header[BUNDLE_HEADER_MAX_SIZE] = {0};
    if (!read_bytes(file, header, 8) || memcmp(header, BUNDLE_MAGIC, 8) != 0) {
        fclose(file);
        return BUNDLE_NOT_BUNDLE;
    }
    if (!read_bytes(file, header + 8, 16)) {
        fclose(file);
        return BUNDLE_FORMAT_ERROR;
    }
    uint32_t version = decode_u32(header + 8);
    uint32_t numeric = decode_u32(header + 16);
    uint32_t header_size = decode_u32(header + 20);
    const int is_v3 = version == MODEL_BUNDLE_FORMAT_V3 &&
                      header_size == BUNDLE_HEADER_V3_SIZE;
    const int is_float = numeric == BUNDLE_NUMERIC_FLOAT32 &&
                         ((version == MODEL_BUNDLE_FORMAT_V1 &&
                           header_size == BUNDLE_HEADER_V1_SIZE) || is_v3);
    const int is_quantized = numeric == BUNDLE_NUMERIC_QUANTIZED &&
                             ((version == MODEL_BUNDLE_FORMAT_V2 &&
                               header_size == BUNDLE_HEADER_V2_SIZE) || is_v3);
    if ((!is_float && !is_quantized) || !float32_supported()) {
        fclose(file);
        return BUNDLE_UNSUPPORTED;
    }
    if (!read_bytes(file, header + 24, header_size - 24)) {
        fclose(file);
        return BUNDLE_FORMAT_ERROR;
    }
    uint64_t stored_header_checksum =
        (uint64_t)decode_u32(header + 76) |
        ((uint64_t)decode_u32(header + 84) << 32);
    memset(header + 76, 0, 4);
    memset(header + 84, 0, 4);
    if (checksum_buffer(header, header_size) != stored_header_checksum) {
        fclose(file);
        return BUNDLE_CHECKSUM_ERROR;
    }
    if (decode_u32(header + 12) != BUNDLE_MARKER) {
        fclose(file);
        return BUNDLE_UNSUPPORTED;
    }

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
    uint32_t architecture_flags = is_v3 ? decode_u32(header + 184) : 0;
    for (size_t i = 0; i < 6; i++) {
        if (dims[i] > SIZE_MAX) { fclose(file); return BUNDLE_FORMAT_ERROR; }
    }
    if ((is_v3 && (architecture_flags == 0 || decode_u32(header + 188) != 0)) ||
        !bundle_shape_valid(dims, architecture_flags) || tokenizer_bytes < 24 ||
        tokenizer_bytes > SIZE_MAX || train_window == 0 ||
        train_window > dims[4] || train_window > SIZE_MAX ||
        (is_float && (dims[5] > UINT64_MAX / 4 ||
                      weight_bytes != dims[5] * 4))) {
        fclose(file);
        return BUNDLE_FORMAT_ERROR;
    }
    if (is_v3 && is_float) {
        for (size_t i = 152; i < 184; i++) {
            if (header[i] != 0) {
                fclose(file);
                return BUNDLE_FORMAT_ERROR;
            }
        }
    }

    model_quant_config_t quant_config = {0};
    uint32_t stored_tensor_count = 0;
    uint64_t stored_values_quantized = 0, stored_scales = 0;
    if (is_quantized) {
        uint32_t flags = decode_u32(header + 160);
        quant_config.bits = (int)decode_u32(header + 152);
        quant_config.granularity = (quant_granularity_t)decode_u32(header + 156);
        quant_config.include_embeddings = (flags & BUNDLE_POLICY_EMBEDDINGS) != 0;
        quant_config.include_biases_and_norms =
            (flags & BUNDLE_POLICY_BIASES_NORMS) != 0;
        stored_tensor_count = decode_u32(header + 164);
        stored_values_quantized = decode_u64(header + 168);
        stored_scales = decode_u64(header + 176);
        if ((flags & ~(BUNDLE_POLICY_EMBEDDINGS |
                       BUNDLE_POLICY_BIASES_NORMS)) != 0 ||
            !quantized_config_valid(&quant_config) ||
            stored_tensor_count == 0 || stored_values_quantized > dims[5] ||
            stored_values_quantized > SIZE_MAX || stored_scales > SIZE_MAX) {
            fclose(file);
            return BUNDLE_FORMAT_ERROR;
        }
    }

    uint64_t expected_file_size = 0;
    struct stat status;
    if (!checked_file_size(header_size, weight_bytes, tokenizer_bytes,
                           &expected_file_size) ||
        fstat(fileno(file), &status) != 0 || status.st_size < 0 ||
        (uint64_t)status.st_size != expected_file_size) {
        fclose(file);
        return BUNDLE_FORMAT_ERROR;
    }
    uint64_t actual_weight_checksum = 0;
    if (!file_region_checksum(file, weight_bytes, &actual_weight_checksum)) {
        fclose(file);
        return BUNDLE_FORMAT_ERROR;
    }
    if (actual_weight_checksum != expected_weight_checksum) {
        fclose(file);
        return BUNDLE_CHECKSUM_ERROR;
    }
    if (fseek(file, (long)header_size, SEEK_SET) != 0) {
        fclose(file);
        return BUNDLE_IO_ERROR;
    }

    neural_model_t loaded = {0};
    model_errors_t model_rc = model_new_seeded_architecture(
        &loaded, (size_t)dims[0], (size_t)dims[1], (size_t)dims[2],
        (size_t)dims[3], (size_t)dims[4], MODEL_DEFAULT_SEED,
        architecture_flags);
    if (model_rc != MODEL_SUCCESS || loaded.total_param_count != (size_t)dims[5]) {
        model_free(&loaded);
        fclose(file);
        return model_rc == MODEL_ALLOCATION_FAILURE
                   ? BUNDLE_ALLOCATION_FAILURE : BUNDLE_FORMAT_ERROR;
    }

    bundle_errors_t weight_error = BUNDLE_SUCCESS;
    weight_reader_t reader = {.file = file, .remaining = weight_bytes};
    if (is_float) {
        for (size_t i = 0; i < loaded.total_param_count; i++) {
            if (!weight_read_float(&reader, &loaded.params[i])) {
                weight_error = BUNDLE_FORMAT_ERROR;
                break;
            }
        }
    } else {
        size_t tensor_count = model_param_tensor_count(&loaded);
        param_tensor_t *tensors = NULL;
        if (tensor_count != stored_tensor_count ||
            tensor_count > SIZE_MAX / sizeof(*tensors)) {
            weight_error = BUNDLE_FORMAT_ERROR;
        } else {
            tensors = malloc(tensor_count * sizeof(*tensors));
            if (!tensors) weight_error = BUNDLE_ALLOCATION_FAILURE;
            else if (model_param_tensors(&loaded, tensors, tensor_count) != tensor_count) {
                weight_error = BUNDLE_FORMAT_ERROR;
            }
        }

        uint64_t actual_values_quantized = 0, actual_scales = 0;
        for (size_t i = 0; weight_error == BUNDLE_SUCCESS && i < tensor_count; i++) {
            uint8_t record[BUNDLE_TENSOR_RECORD_SIZE];
            if (!weight_read(&reader, record, sizeof(record))) {
                weight_error = BUNDLE_FORMAT_ERROR;
                break;
            }
            param_tensor_t *tensor = &tensors[i];
            const int included = model_quantize_includes(&quant_config, tensor->kind);
            uint32_t storage = decode_u32(record + 24);
            if (decode_u32(record) != i || decode_u32(record + 4) != (uint32_t)tensor->kind ||
                decode_u64(record + 8) != tensor->rows ||
                decode_u64(record + 16) != tensor->cols ||
                decode_u32(record + 28) != 0 ||
                storage != (included ? BUNDLE_TENSOR_QUANTIZED
                                     : BUNDLE_TENSOR_FLOAT32)) {
                weight_error = BUNDLE_FORMAT_ERROR;
                break;
            }
            size_t count = tensor->rows * tensor->cols;
            if (!included) {
                for (size_t j = 0; j < count; j++) {
                    if (!weight_read_float(&reader, &tensor->values[j])) {
                        weight_error = BUNDLE_FORMAT_ERROR;
                        break;
                    }
                }
                continue;
            }

            size_t scale_count = 0, packed_size = 0;
            uint64_t scale_bytes = 0, tensor_payload_bytes = 0;
            if (quantized_scale_count(tensor->rows, tensor->cols,
                                      quant_config.granularity, &scale_count) != 0 ||
                quantized_packed_size(count, quant_config.bits, &packed_size) != 0 ||
                !checked_u64_multiply(scale_count, 4, &scale_bytes) ||
                !checked_u64_add(scale_bytes, packed_size, &tensor_payload_bytes) ||
                tensor_payload_bytes > reader.remaining ||
                scale_count > SIZE_MAX / sizeof(float)) {
                weight_error = BUNDLE_FORMAT_ERROR;
                break;
            }
            float *scales = malloc(scale_count * sizeof(*scales));
            uint8_t *packed = malloc(packed_size);
            if (!scales || !packed) {
                free(scales);
                free(packed);
                weight_error = BUNDLE_ALLOCATION_FAILURE;
                break;
            }
            for (size_t s = 0; s < scale_count; s++) {
                if (!weight_read_float(&reader, &scales[s])) {
                    weight_error = BUNDLE_FORMAT_ERROR;
                    break;
                }
            }
            if (weight_error == BUNDLE_SUCCESS &&
                (!weight_read(&reader, packed, packed_size) ||
                 quantize_unpack(scales, scale_count, packed, packed_size,
                                 tensor->rows, tensor->cols, quant_config.bits,
                                 quant_config.granularity, tensor->values) != 0)) {
                weight_error = BUNDLE_FORMAT_ERROR;
            }
            free(scales);
            free(packed);
            if (!checked_u64_add(actual_values_quantized, count,
                                 &actual_values_quantized) ||
                !checked_u64_add(actual_scales, scale_count, &actual_scales)) {
                weight_error = BUNDLE_FORMAT_ERROR;
            }
        }
        free(tensors);
        if (weight_error == BUNDLE_SUCCESS &&
            (actual_values_quantized != stored_values_quantized ||
             actual_scales != stored_scales)) weight_error = BUNDLE_FORMAT_ERROR;
    }
    if (weight_error == BUNDLE_SUCCESS && reader.remaining != 0) {
        weight_error = BUNDLE_FORMAT_ERROR;
    }
    if (weight_error != BUNDLE_SUCCESS) {
        model_free(&loaded);
        fclose(file);
        return weight_error;
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
    uint64_t tokenizer_max_vocab = 0;
    if (bpe_encoder_portable_max_vocab(tokenizer_data,
                                       (size_t)tokenizer_bytes,
                                       &tokenizer_max_vocab) != BPE_SUCCESS ||
        tokenizer_max_vocab != dims[0]) {
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
    out_metadata->format_version = version;
    *model = loaded;
    *out_encoder = encoder;
    return BUNDLE_SUCCESS;
}

bundle_errors_t model_bundle_load_mmap(neural_model_t *model,
                                       bpe_encoder_t **out_encoder,
                                       model_bundle_metadata_t *out_metadata,
                                       const char *filename) {
    if (!model || !out_encoder || !out_metadata || !filename) {
        return BUNDLE_FORMAT_ERROR;
    }
    *out_encoder = NULL;
    memset(out_metadata, 0, sizeof(*out_metadata));

    int fd = open(filename, O_RDONLY);
    if (fd < 0) return BUNDLE_IO_ERROR;
    struct stat status;
    if (fstat(fd, &status) != 0 || status.st_size < 0) {
        close(fd);
        return BUNDLE_IO_ERROR;
    }
    uint64_t file_size_u64 = (uint64_t)status.st_size;
    if (file_size_u64 < 8) {
        close(fd);
        return BUNDLE_NOT_BUNDLE;
    }
    if (file_size_u64 > SIZE_MAX) {
        close(fd);
        return BUNDLE_UNSUPPORTED;
    }
    size_t file_size = (size_t)file_size_u64;
    uint8_t *mapping = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) return BUNDLE_IO_ERROR;

    if (memcmp(mapping, BUNDLE_MAGIC, 8) != 0) {
        munmap(mapping, file_size);
        return BUNDLE_NOT_BUNDLE;
    }
    if (file_size < BUNDLE_HEADER_V1_SIZE + 8) {
        munmap(mapping, file_size);
        return BUNDLE_FORMAT_ERROR;
    }
    uint32_t version = decode_u32(mapping + 8);
    uint32_t numeric = decode_u32(mapping + 16);
    uint32_t header_size = decode_u32(mapping + 20);
    const int is_v1_float = version == MODEL_BUNDLE_FORMAT_V1 &&
                            header_size == BUNDLE_HEADER_V1_SIZE;
    const int is_v3_float = version == MODEL_BUNDLE_FORMAT_V3 &&
                            header_size == BUNDLE_HEADER_V3_SIZE;
    if ((!is_v1_float && !is_v3_float) ||
        numeric != BUNDLE_NUMERIC_FLOAT32 ||
        !float32_supported() || !host_is_little_endian()) {
        munmap(mapping, file_size);
        return BUNDLE_UNSUPPORTED;
    }
    if (decode_u32(mapping + 12) != BUNDLE_MARKER) {
        munmap(mapping, file_size);
        return BUNDLE_UNSUPPORTED;
    }

    if (file_size < (size_t)header_size + 8) {
        munmap(mapping, file_size);
        return BUNDLE_FORMAT_ERROR;
    }
    uint8_t header[BUNDLE_HEADER_MAX_SIZE] = {0};
    memcpy(header, mapping, header_size);
    uint64_t stored_header_checksum =
        (uint64_t)decode_u32(header + 76) |
        ((uint64_t)decode_u32(header + 84) << 32);
    memset(header + 76, 0, 4);
    memset(header + 84, 0, 4);
    if (checksum_buffer(header, header_size) != stored_header_checksum) {
        munmap(mapping, file_size);
        return BUNDLE_CHECKSUM_ERROR;
    }

    uint64_t dims[6];
    for (size_t i = 0; i < 6; i++) {
        dims[i] = decode_u64(mapping + 24 + i * 8);
        if (dims[i] > SIZE_MAX) {
            munmap(mapping, file_size);
            return BUNDLE_FORMAT_ERROR;
        }
    }
    uint32_t training_steps = decode_u32(mapping + 72);
    uint32_t loss_bits = decode_u32(mapping + 80);
    uint64_t train_window = decode_u64(mapping + 88);
    uint64_t seed = decode_u64(mapping + 96);
    uint64_t input_fingerprint = decode_u64(mapping + 104);
    uint64_t input_bytes = decode_u64(mapping + 112);
    uint64_t weight_bytes = decode_u64(mapping + 120);
    uint64_t tokenizer_bytes = decode_u64(mapping + 128);
    uint64_t expected_weight_checksum = decode_u64(mapping + 136);
    uint64_t expected_tokenizer_checksum = decode_u64(mapping + 144);
    uint32_t architecture_flags = is_v3_float ? decode_u32(mapping + 184) : 0;
    if ((is_v3_float &&
         (architecture_flags == 0 || decode_u32(mapping + 188) != 0)) ||
        !bundle_shape_valid(dims, architecture_flags) ||
        dims[5] > UINT64_MAX / 4 ||
        weight_bytes != dims[5] * 4 || tokenizer_bytes < 24 ||
        tokenizer_bytes > SIZE_MAX || train_window == 0 ||
        train_window > dims[4] || train_window > SIZE_MAX) {
        munmap(mapping, file_size);
        return BUNDLE_FORMAT_ERROR;
    }
    uint64_t expected_file_size = 0;
    if (is_v3_float) {
        for (size_t i = 152; i < 184; i++) {
            if (mapping[i] != 0) {
                munmap(mapping, file_size);
                return BUNDLE_FORMAT_ERROR;
            }
        }
    }
    if (!checked_file_size(header_size, weight_bytes,
                           tokenizer_bytes, &expected_file_size) ||
        expected_file_size != file_size_u64) {
        munmap(mapping, file_size);
        return BUNDLE_FORMAT_ERROR;
    }

    const uint8_t *weight_data = mapping + header_size;
    const uint8_t *tokenizer_data = weight_data + (size_t)weight_bytes;
    if (checksum_buffer(weight_data, (size_t)weight_bytes) !=
        expected_weight_checksum ||
        checksum_buffer(tokenizer_data, (size_t)tokenizer_bytes) !=
        expected_tokenizer_checksum) {
        munmap(mapping, file_size);
        return BUNDLE_CHECKSUM_ERROR;
    }
    if (memcmp(mapping + file_size - 8, BUNDLE_FOOTER, 8) != 0) {
        munmap(mapping, file_size);
        return BUNDLE_FORMAT_ERROR;
    }
    uint64_t tokenizer_max_vocab = 0;
    if (bpe_encoder_portable_max_vocab(tokenizer_data,
                                       (size_t)tokenizer_bytes,
                                       &tokenizer_max_vocab) != BPE_SUCCESS ||
        tokenizer_max_vocab != dims[0]) {
        munmap(mapping, file_size);
        return BUNDLE_FORMAT_ERROR;
    }

    neural_model_t loaded = {0};
    model_errors_t model_rc = model_new_external_parameters_architecture(
        &loaded, (size_t)dims[0], (size_t)dims[1], (size_t)dims[2],
        (size_t)dims[3], (size_t)dims[4],
        architecture_flags,
        (float *)(void *)weight_data);
    if (model_rc != MODEL_SUCCESS ||
        loaded.total_param_count != (size_t)dims[5]) {
        model_free(&loaded);
        munmap(mapping, file_size);
        return model_rc == MODEL_ALLOCATION_FAILURE
                   ? BUNDLE_ALLOCATION_FAILURE : BUNDLE_FORMAT_ERROR;
    }
    /* From this point model_free owns the mapping on every exit path. */
    loaded.params_mapping = mapping;
    loaded.params_mapping_size = file_size;

    bpe_encoder_t *encoder = calloc(1, sizeof(*encoder));
    if (!encoder) {
        model_free(&loaded);
        return BUNDLE_ALLOCATION_FAILURE;
    }
    bpe_errors_t tokenizer_rc = bpe_encoder_deserialize_portable(
        encoder, tokenizer_data, (size_t)tokenizer_bytes);
    if (tokenizer_rc != BPE_SUCCESS ||
        encoder->max_vocab_size != loaded.vocab_size) {
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
    out_metadata->format_version = version;
    *model = loaded;
    *out_encoder = encoder;
    return BUNDLE_SUCCESS;
}
