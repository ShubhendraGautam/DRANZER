/*
 * Model weight (de)serialization. Every trainable weight is a contiguous
 * view into model->params (model_types.h / model_new's flat layout), so
 * writing/reading the whole parameter set is a single fwrite/fread instead
 * of one line per named field - adding a new parameter array anywhere in
 * the model needs no change here.
 */

#include "core/serialization.h"
#include "core/model.h"
#include "common/debug.h"
#include <stdlib.h>

model_errors_t model_write_state(const neural_model_t *model, FILE *f) {
    if (!model || !f) {
        return MODEL_INVALID_INPUT;
    }
    /* The host-native legacy layout has no architecture field. Refuse to
     * emit a tied model that its own reader would reinterpret as untied.
     * Portable bundles carry architecture flags and are the supported path. */
    if (model->architecture_flags != 0) return MODEL_INVALID_INPUT;

    fwrite(&model->vocab_size, sizeof(size_t), 1, f);
    fwrite(&model->embedding_dim, sizeof(size_t), 1, f);
    fwrite(&model->num_heads, sizeof(size_t), 1, f);
    fwrite(&model->num_layers, sizeof(size_t), 1, f);
    fwrite(&model->max_seq_len, sizeof(size_t), 1, f);
    fwrite(&model->training_steps, sizeof(uint32_t), 1, f);
    fwrite(&model->current_loss, sizeof(float), 1, f);

    fwrite(&model->total_param_count, sizeof(size_t), 1, f);
    fwrite(model->params, sizeof(float), model->total_param_count, f);

    return MODEL_SUCCESS;
}

model_errors_t model_read_state(neural_model_t *model, FILE *f) {
    if (!model || !f) {
        return MODEL_INVALID_INPUT;
    }

    size_t vocab_size, embedding_dim, num_heads, num_layers, max_seq_len;
    uint32_t training_steps;
    float current_loss;
    size_t total_param_count;

    if (fread(&vocab_size, sizeof(size_t), 1, f) != 1 ||
        fread(&embedding_dim, sizeof(size_t), 1, f) != 1 ||
        fread(&num_heads, sizeof(size_t), 1, f) != 1 ||
        fread(&num_layers, sizeof(size_t), 1, f) != 1 ||
        fread(&max_seq_len, sizeof(size_t), 1, f) != 1 ||
        fread(&training_steps, sizeof(uint32_t), 1, f) != 1 ||
        fread(&current_loss, sizeof(float), 1, f) != 1 ||
        fread(&total_param_count, sizeof(size_t), 1, f) != 1) {
        return MODEL_IO_ERROR;
    }

    if (model->vocab_size != vocab_size || model->embedding_dim != embedding_dim ||
        model->num_heads != num_heads || model->num_layers != num_layers ||
        model->max_seq_len != max_seq_len) {
        model_free(model);
        model_errors_t rc = model_new(model, vocab_size, embedding_dim, num_heads, num_layers, max_seq_len);
        if (rc != MODEL_SUCCESS) {
            return rc;
        }
    }

    if (total_param_count != model->total_param_count) {
        return MODEL_IO_ERROR; /* corrupt file or a model_new layout mismatch */
    }

    model->training_steps = training_steps;
    model->current_loss = current_loss;

    if (fread(model->params, sizeof(float), total_param_count, f) != total_param_count) {
        return MODEL_IO_ERROR;
    }

    return MODEL_SUCCESS;
}

model_errors_t model_save(neural_model_t *model, const char *filename) {
    if (!model || !filename || model->architecture_flags != 0) {
        return MODEL_INVALID_INPUT;
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        return MODEL_IO_ERROR;
    }

    DEBUG_PRINT("Saving model to %s\n", filename);
    model_errors_t rc = model_write_state(model, f);
    fclose(f);
    if (rc == MODEL_SUCCESS) {
        DEBUG_PRINT("Model saved successfully\n");
    }
    return rc;
}

model_errors_t model_load(neural_model_t *model, const char *filename) {
    if (!model || !filename) {
        return MODEL_INVALID_INPUT;
    }

    FILE *f = fopen(filename, "rb");
    if (!f) {
        return MODEL_IO_ERROR;
    }

    DEBUG_PRINT("Loading model from %s\n", filename);
    model_errors_t rc = model_read_state(model, f);
    fclose(f);
    if (rc == MODEL_SUCCESS) {
        DEBUG_PRINT("Model loaded successfully (training_steps=%u, loss=%.4f)\n",
                    model->training_steps, model->current_loss);
    }
    return rc;
}
