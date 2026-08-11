/*
 * Walks the flat parameter layout model_new() builds and describes each tensor
 * in it. See core/model_params.h.
 *
 * This file and core/model.c's layout_layer() describe the same memory twice.
 * That duplication is real and deliberate: the alternative is for model_new()
 * to build a descriptor table at construction time and pay for it in every
 * model, including the ones that never quantize or report anything.
 * test_model_params.c is what keeps the two in step - it checks every
 * descriptor against the model's own named pointers and requires the
 * descriptors to tile `params` exactly, with no gap and no overlap, so a
 * tensor added to one and not the other fails rather than being skipped.
 */

#include "core/model_params.h"
#include <stdio.h>

/* Append one descriptor if there is room, and count it either way, so the
 * caller learns the true total from a single pass with a zero-capacity array. */
static void emit(param_tensor_t *out, size_t capacity, size_t *count,
                 const char *name, size_t layer, float *values,
                 size_t rows, size_t cols, param_kind_t kind) {
    if (out && *count < capacity) {
        param_tensor_t *tensor = &out[*count];
        if (layer == (size_t)-1) {
            snprintf(tensor->name, sizeof(tensor->name), "%s", name);
        } else {
            snprintf(tensor->name, sizeof(tensor->name), "layer%zu.%s", layer, name);
        }
        tensor->values = values;
        tensor->rows = rows;
        tensor->cols = cols;
        tensor->kind = kind;
        tensor->layer = layer;
    }
    (*count)++;
}

size_t model_param_tensors(const neural_model_t *model,
                           param_tensor_t *out, size_t capacity) {
    if (!model || !model->params || !model->layers) return 0;

    const size_t embedding_dim = model->embedding_dim;
    const size_t vocab_size = model->vocab_size;
    const size_t ffn_dim = embedding_dim * 4;
    const size_t global = (size_t)-1;
    size_t count = 0;

    emit(out, capacity, &count, "token_embeddings", global,
         model->token_embeddings, vocab_size, embedding_dim,
         PARAM_KIND_EMBEDDING);
    if ((model->architecture_flags & MODEL_ARCH_TIED_EMBEDDINGS) == 0) {
        emit(out, capacity, &count, "output_projection", global,
             model->output_projection, embedding_dim, vocab_size,
             PARAM_KIND_PROJECTION);
    }
    emit(out, capacity, &count, "output_bias", global,
         model->output_bias, 1, vocab_size, PARAM_KIND_BIAS);

    for (size_t l = 0; l < model->num_layers; l++) {
        transformer_layer_t *layer = &model->layers[l];

        emit(out, capacity, &count, "W_q", l, layer->W_q,
             embedding_dim, embedding_dim, PARAM_KIND_PROJECTION);
        emit(out, capacity, &count, "W_k", l, layer->W_k,
             embedding_dim, embedding_dim, PARAM_KIND_PROJECTION);
        emit(out, capacity, &count, "W_v", l, layer->W_v,
             embedding_dim, embedding_dim, PARAM_KIND_PROJECTION);
        emit(out, capacity, &count, "W_o", l, layer->W_o,
             embedding_dim, embedding_dim, PARAM_KIND_PROJECTION);

        emit(out, capacity, &count, "ln_gamma_attn", l, layer->ln_gamma_attn,
             1, embedding_dim, PARAM_KIND_NORM);
        emit(out, capacity, &count, "ln_beta_attn", l, layer->ln_beta_attn,
             1, embedding_dim, PARAM_KIND_NORM);

        emit(out, capacity, &count, "W_ff1", l, layer->W_ff1,
             embedding_dim, ffn_dim, PARAM_KIND_PROJECTION);
        emit(out, capacity, &count, "b_ff1", l, layer->b_ff1,
             1, ffn_dim, PARAM_KIND_BIAS);
        emit(out, capacity, &count, "W_ff2", l, layer->W_ff2,
             ffn_dim, embedding_dim, PARAM_KIND_PROJECTION);
        emit(out, capacity, &count, "b_ff2", l, layer->b_ff2,
             1, embedding_dim, PARAM_KIND_BIAS);

        emit(out, capacity, &count, "ln_gamma_ffn", l, layer->ln_gamma_ffn,
             1, embedding_dim, PARAM_KIND_NORM);
        emit(out, capacity, &count, "ln_beta_ffn", l, layer->ln_beta_ffn,
             1, embedding_dim, PARAM_KIND_NORM);
    }

    return count;
}

size_t model_param_tensor_count(const neural_model_t *model) {
    return model_param_tensors(model, NULL, 0);
}

static const char *const kind_names[PARAM_KIND_COUNT] = {
    "projection", "embedding", "bias", "norm"
};

const char *param_kind_name(param_kind_t kind) {
    int value = (int)kind;
    if (value < 0 || value >= (int)PARAM_KIND_COUNT) return "unknown";
    return kind_names[kind];
}
