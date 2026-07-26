/* Compares a full model_forward() pass (embeddings -> 2 stacked
 * transformer layers -> output head) with model->use_gpu = 0 against the
 * same model/weights/tokens with use_gpu = 1, going well beyond
 * test_gpu_matmul.c's single-matmul comparison: this exercises every
 * forward-pass dispatch_matmul() call site in transformer.c (Q/K/V/output
 * projections and both FFN matmuls, across both layers) chained together.
 *
 * Tolerance is looser than a single matmul comparison on purpose: CPU
 * matrix_multiply() (cache-blocked accumulation order) and the GPU PTX
 * kernel (straightforward per-thread K-loop) don't sum in the same order,
 * so per-matmul rounding differences are expected and can compound across
 * 2 stacked layers' worth of chained matmuls - this project already
 * builds with -ffast-math, which permits reassociation, so bit-identical
 * output was never the bar even between different CPU configurations.
 * Self-skips (reports success, not failure) on machines without a usable
 * CUDA GPU. */
#include "core/model.h"
#include "backends/gpu/gpu_matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB 40
#define EMB 16
#define HEADS 2
#define LAYERS 2
#define MAX_SEQ 12
#define SEQ_LEN 8

int main(void) {
    if (!gpu_matmul_available()) {
        printf("SKIP: no usable CUDA GPU detected (see gpu_probe.out for details) - not a failure.\n");
        return 0;
    }

    srand(5);
    neural_model_t model = {0};
    if (model_new(&model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        return 1;
    }
    model.is_training = 0; /* no dropout - keep this a pure matmul-path comparison */

    uint32_t tokens[SEQ_LEN] = {1, 5, 3, 9, 2, 7, 4, 6};

    float logits_cpu[VOCAB], logits_gpu[VOCAB];

    model.use_gpu = 0;
    model_forward(&model, tokens, SEQ_LEN, logits_cpu);

    model.use_gpu = 1;
    model_forward(&model, tokens, SEQ_LEN, logits_gpu);

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < VOCAB; i++) {
        float diff = fabsf(logits_cpu[i] - logits_gpu[i]);
        float denom = fmaxf(fabsf(logits_cpu[i]), fabsf(logits_gpu[i]));
        float rel = (denom > 1e-6f) ? diff / denom : diff;
        if (diff > max_abs_diff) max_abs_diff = diff;
        if (rel > max_rel_diff) max_rel_diff = rel;
    }
    printf("max abs diff: %e, max rel diff: %e (over %d logits, %zu-token context, %d layers)\n",
           max_abs_diff, max_rel_diff, VOCAB, (size_t)SEQ_LEN, LAYERS);

    /* Loose but meaningful: catches a real dispatch bug (wrong buffer,
     * wrong shape, garbage data) while tolerating genuine
     * summation-order floating-point drift compounded across layers. */
    int pass = (max_abs_diff < 0.05f) && (max_rel_diff < 0.05f);

    model_free(&model);
    printf("\n%s\n", pass ? "GPU MODEL FORWARD CHECK PASSED" : "GPU MODEL FORWARD CHECK FAILED");
    return pass ? 0 : 1;
}
