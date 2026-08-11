/*
 * The parameter inventory must describe the same memory model_new() laid out.
 *
 * core/model_params.c walks the flat `params` buffer a second time, by hand,
 * and that duplication is only safe if something checks the two against each
 * other. The strongest available check is not "are the pointers right" but
 * "do the descriptors tile the buffer exactly": every float in `params` covered
 * once, with no gap and no overlap. A tensor added to model_new() and forgotten
 * here leaves a gap; a shape copied wrongly creates an overlap. Both fail here
 * rather than silently excluding weights from quantization - which would look
 * like a quantization scheme that happened to be unusually accurate.
 */

#include "core/model.h"
#include "core/model_params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    failures++;
}

typedef struct { size_t vocab, embedding_dim, heads, layers, seq; } config_t;

static const config_t configs[] = {
    { 12,  8, 2, 1,  8 },
    { 32, 64, 4, 2,  4 },
    { 96, 16, 2, 3, 16 },
};
#define CONFIG_COUNT (sizeof(configs) / sizeof(configs[0]))

static void check_model(const config_t *config, uint32_t architecture_flags) {
    neural_model_t model = {0};
    if (model_new_seeded_architecture(
            &model, config->vocab, config->embedding_dim, config->heads,
            config->layers, config->seq, 7,
            architecture_flags) != MODEL_SUCCESS) {
        fail("model_new failed");
        return;
    }

    const size_t count = model_param_tensor_count(&model);
    if (count == 0) {
        fail("the inventory is empty");
        model_free(&model);
        return;
    }

    param_tensor_t *tensors = malloc(count * sizeof(*tensors));
    unsigned char *covered = calloc(model.total_param_count, 1);
    if (!tensors || !covered) {
        fail("allocation");
        free(tensors); free(covered);
        model_free(&model);
        return;
    }

    if (model_param_tensors(&model, tensors, count) != count) {
        fail("the count changed between calls");
    }

    /* A short buffer must still report the true total and write only what
     * fits, so a caller can size its array from one call. */
    param_tensor_t one;
    memset(&one, 0, sizeof(one));
    if (count > 1 && model_param_tensors(&model, &one, 1) != count) {
        fail("a short buffer did not report the true total");
    }
    if (count > 1 && strcmp(one.name, tensors[0].name) != 0) {
        fail("a short buffer did not write the first descriptor");
    }

    for (size_t i = 0; i < count; i++) {
        const param_tensor_t *tensor = &tensors[i];

        if (tensor->rows == 0 || tensor->cols == 0) {
            fail("a descriptor has a zero extent");
            continue;
        }
        if (tensor->name[0] == '\0') fail("a descriptor has no name");
        if (strcmp(param_kind_name(tensor->kind), "unknown") == 0) {
            fail("a descriptor has an unknown kind");
        }

        /* Inside the flat buffer, and exactly once. */
        if (tensor->values < model.params ||
            tensor->values + tensor->rows * tensor->cols >
                model.params + model.total_param_count) {
            fprintf(stderr, "  %s escapes the parameter buffer\n", tensor->name);
            fail("a descriptor points outside params");
            continue;
        }

        size_t offset = (size_t)(tensor->values - model.params);
        for (size_t j = 0; j < tensor->rows * tensor->cols; j++) {
            if (covered[offset + j]) {
                fprintf(stderr, "  %s overlaps an earlier tensor at +%zu\n",
                        tensor->name, offset + j);
                fail("two descriptors cover the same parameter");
                break;
            }
            covered[offset + j] = 1;
        }

        /* Names must be unique: a report keyed on them would otherwise merge
         * two tensors silently. */
        for (size_t j = 0; j < i; j++) {
            if (strcmp(tensors[i].name, tensors[j].name) == 0) {
                fprintf(stderr, "  duplicate name %s\n", tensors[i].name);
                fail("two descriptors share a name");
                break;
            }
        }
    }

    size_t uncovered = 0;
    for (size_t i = 0; i < model.total_param_count; i++) {
        if (!covered[i]) uncovered++;
    }
    if (uncovered != 0) {
        fprintf(stderr, "  %zu of %zu parameters are in no descriptor\n",
                uncovered, model.total_param_count);
        fail("the inventory does not cover every parameter");
    }

    /* The named views must be the ones described, not merely equivalent
     * regions - this is what catches two tensors of the same shape being
     * swapped. */
    int found_embeddings = 0, found_output = 0;
    for (size_t i = 0; i < count; i++) {
        if (strcmp(tensors[i].name, "token_embeddings") == 0) {
            found_embeddings = 1;
            if (tensors[i].values != model.token_embeddings ||
                tensors[i].rows != config->vocab ||
                tensors[i].cols != config->embedding_dim ||
                tensors[i].kind != PARAM_KIND_EMBEDDING) {
                fail("token_embeddings is described wrongly");
            }
        }
        if (strcmp(tensors[i].name, "output_projection") == 0) {
            found_output = 1;
            if (tensors[i].values != model.output_projection ||
                tensors[i].rows != config->embedding_dim ||
                tensors[i].cols != config->vocab ||
                tensors[i].kind != PARAM_KIND_PROJECTION) {
                fail("output_projection is described wrongly");
            }
        }
    }
    if (!found_embeddings ||
        (found_output ==
         ((architecture_flags & MODEL_ARCH_TIED_EMBEDDINGS) != 0))) {
        fail("a known global tensor does not match the architecture");
    }

    for (size_t l = 0; l < config->layers; l++) {
        char wanted[48];
        snprintf(wanted, sizeof(wanted), "layer%zu.W_ff1", l);
        int found = 0;
        for (size_t i = 0; i < count; i++) {
            if (strcmp(tensors[i].name, wanted) != 0) continue;
            found = 1;
            if (tensors[i].values != model.layers[l].W_ff1 ||
                tensors[i].rows != config->embedding_dim ||
                tensors[i].cols != config->embedding_dim * 4 ||
                tensors[i].layer != l) {
                fail("a layer tensor is described wrongly");
            }
        }
        if (!found) fail("a layer tensor is missing from the inventory");
    }

    free(tensors);
    free(covered);
    model_free(&model);
}

int main(void) {
    static const uint32_t architecture_flags[] = {
        0,
        MODEL_ARCH_TIED_EMBEDDINGS,
        MODEL_ARCH_RMSNORM,
        MODEL_ARCH_TIED_EMBEDDINGS | MODEL_ARCH_RMSNORM,
    };
    for (size_t c = 0; c < CONFIG_COUNT; c++) {
        for (size_t variant = 0;
             variant < sizeof(architecture_flags) / sizeof(architecture_flags[0]);
             variant++) check_model(&configs[c], architecture_flags[variant]);
    }

    /* A model that was never constructed must produce nothing rather than
     * walking null pointers. */
    neural_model_t empty = {0};
    if (model_param_tensor_count(&empty) != 0) {
        fail("an unconstructed model produced descriptors");
    }
    if (model_param_tensors(NULL, NULL, 0) != 0) {
        fail("a null model produced descriptors");
    }

    printf("configurations=%zu\n",
           CONFIG_COUNT * sizeof(architecture_flags) / sizeof(architecture_flags[0]));
    if (failures != 0) {
        printf("\nPARAMETER INVENTORY CHECK FAILED (%d problem%s)\n",
               failures, failures == 1 ? "" : "s");
        return 1;
    }
    printf("\nPARAMETER INVENTORY CHECK PASSED\n");
    return 0;
}
