/* End-to-end validation of the GPU caching path in real usage: trains two
 * identically-initialized models for several steps, one with
 * model->use_gpu = 0 and one with = 1, and compares final loss and a
 * sample of final weights. This is what actually exercises the risk
 * surface test_gpu_weight_cache.c checks in isolation - weights changing
 * between calls via model_optimizer_step - in the real call pattern
 * (transformer.c's dispatch_matmul, training.c's invalidation hook)
 * rather than a synthetic single mutation. Self-skips on machines without
 * a CUDA GPU. */
#include "core/model.h"
#include "backends/gpu/gpu_matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB 30
#define EMB 16
#define HEADS 2
#define LAYERS 2
#define MAX_SEQ 10
#define SEQ_LEN 6
#define STEPS 15

static void init_identical(neural_model_t *model) {
    srand(11);
    if (model_new(model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        exit(1);
    }
    model->learning_rate = 0.01f;
}

int main(void) {
    if (!gpu_matmul_available()) {
        printf("SKIP: no usable CUDA GPU detected (see gpu_probe.out for details) - not a failure.\n");
        return 0;
    }

    uint32_t tokens[SEQ_LEN] = {1, 4, 2, 7, 3, 5};
    uint32_t target = 9;

    neural_model_t cpu_model = {0}, gpu_model = {0};
    init_identical(&cpu_model);
    init_identical(&gpu_model);
    cpu_model.use_gpu = 0;
    gpu_model.use_gpu = 1;

    float cpu_loss = 0.0f, gpu_loss = 0.0f;
    for (int step = 0; step < STEPS; step++) {
        model_train_step(&cpu_model, tokens, target, SEQ_LEN);
        model_train_step(&gpu_model, tokens, target, SEQ_LEN);
        cpu_loss = cpu_model.current_loss;
        gpu_loss = gpu_model.current_loss;
    }
    printf("after %d steps: CPU loss=%.6f, GPU loss=%.6f, diff=%.6e\n",
           STEPS, cpu_loss, gpu_loss, fabsf(cpu_loss - gpu_loss));

    /* Compare a sample of final weights across every layer + the
     * embeddings/output head - if the weight cache ever served stale
     * data mid-training, these would have diverged from the CPU run by
     * more than floating-point summation-order noise. */
    float max_weight_diff = 0.0f;
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.layers[0].W_q[3] - gpu_model.layers[0].W_q[3]));
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.layers[0].W_ff2[7] - gpu_model.layers[0].W_ff2[7]));
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.layers[1].W_o[2] - gpu_model.layers[1].W_o[2]));
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.layers[1].W_ff1[10] - gpu_model.layers[1].W_ff1[10]));
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.output_projection[5] - gpu_model.output_projection[5]));
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.token_embeddings[20] - gpu_model.token_embeddings[20]));
    printf("max weight diff after %d steps (sampled across both layers + embeddings + output head): %e\n",
           STEPS, max_weight_diff);

    /* Loose but meaningful: 15 steps of compounded per-matmul
     * summation-order drift (CPU cache-blocked vs GPU naive K-loop) is
     * expected to grow beyond a single forward pass's ~1e-6, but a stale
     * weight cache bug would show up as a difference many orders of
     * magnitude larger than genuine float noise, not a borderline case. */
    int pass = fabsf(cpu_loss - gpu_loss) < 0.05f && max_weight_diff < 0.01f;

    model_free(&cpu_model);
    model_free(&gpu_model);

    printf("\n%s\n", pass ? "GPU TRAINING STEP CHECK PASSED" : "GPU TRAINING STEP CHECK FAILED");
    return pass ? 0 : 1;
}
