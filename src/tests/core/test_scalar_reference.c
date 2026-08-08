#include "core/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int compare(const float *left, const float *right, size_t count,
                   float tolerance, float *worst) {
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(left[i] - right[i]);
        if (diff > *worst) *worst = diff;
        if (diff > tolerance) return -1;
    }
    return 0;
}

int main(void) {
    const size_t m = 5, k = 73, n = 67;
    float *a = malloc(m * k * sizeof(*a));
    float *b = malloc(k * n * sizeof(*b));
    float *tiled = malloc(m * n * sizeof(*tiled));
    float *scalar = malloc(m * n * sizeof(*scalar));
    if (!a || !b || !tiled || !scalar) {
        free(a); free(b); free(tiled); free(scalar);
        return 1;
    }
    for (size_t i = 0; i < m * k; i++)
        a[i] = (float)((int)(i % 19) - 9) / 11.0f;
    for (size_t i = 0; i < k * n; i++)
        b[i] = (float)((int)(i % 23) - 11) / 13.0f;

    matrix_multiply(a, b, tiled, m, k, n);
    matrix_multiply_scalar(a, b, scalar, m, k, n);
    float worst = 0.0f;
    if (compare(tiled, scalar, m * n, 2e-5f, &worst) != 0) {
        fprintf(stderr, "tiled matmul diverged from scalar reference\n");
        free(a); free(b); free(tiled); free(scalar);
        return 1;
    }
    free(a); free(b); free(tiled); free(scalar);

    neural_model_t model = {0};
    if (model_new_seeded(&model, 80, 72, 8, 2, 8, 29) != MODEL_SUCCESS) {
        fprintf(stderr, "reference model allocation failed\n");
        return 1;
    }
    model.is_training = 0;
    uint32_t tokens[] = {3, 17, 9, 41, 7, 22};
    float tiled_logits[80], scalar_logits[80];

    model.use_scalar_matmul = 0;
    if (model_forward(&model, tokens, 6, tiled_logits) != MODEL_SUCCESS) {
        model_free(&model);
        return 1;
    }
    model.use_scalar_matmul = 1;
    if (model_forward(&model, tokens, 6, scalar_logits) != MODEL_SUCCESS ||
        compare(tiled_logits, scalar_logits, 80, 3e-5f, &worst) != 0) {
        fprintf(stderr, "model forward diverged from scalar reference\n");
        model_free(&model);
        return 1;
    }

    model_kv_cache_t tiled_cache = {0}, scalar_cache = {0};
    if (model_kv_cache_init(&tiled_cache, &model) != MODEL_SUCCESS ||
        model_kv_cache_init(&scalar_cache, &model) != MODEL_SUCCESS) {
        model_kv_cache_free(&tiled_cache);
        model_kv_cache_free(&scalar_cache);
        model_free(&model);
        return 1;
    }
    for (size_t i = 0; i < 6; i++) {
        model.use_scalar_matmul = 0;
        model_errors_t tiled_rc = model_forward_token(
            &model, &tiled_cache, tokens[i], tiled_logits);
        model.use_scalar_matmul = 1;
        model_errors_t scalar_rc = model_forward_token(
            &model, &scalar_cache, tokens[i], scalar_logits);
        if (tiled_rc != MODEL_SUCCESS || scalar_rc != MODEL_SUCCESS ||
            compare(tiled_logits, scalar_logits, 80, 3e-5f, &worst) != 0) {
            fprintf(stderr, "cached decode diverged at position %zu\n", i);
            model_kv_cache_free(&tiled_cache);
            model_kv_cache_free(&scalar_cache);
            model_free(&model);
            return 1;
        }
    }

    printf("worst scalar-vs-dispatch difference: %.8g\n", worst);
    printf("\nSCALAR REFERENCE CHECK PASSED\n");
    model_kv_cache_free(&tiled_cache);
    model_kv_cache_free(&scalar_cache);
    model_free(&model);
    return 0;
}
