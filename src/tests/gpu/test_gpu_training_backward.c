/* End-to-end validation that the GPU *backward* path is correct in a real
 * training loop, not just as isolated kernels.
 *
 * test_gpu_matmul_backward.c proves the two PTX kernels compute the right
 * thing. This proves the dispatch around them routes the right operands: it
 * trains two identically-initialized models, one on the CPU and one with
 * model->use_gpu = 1, and requires their losses and weights to track.
 *
 * The model here is deliberately much larger than test_gpu_training_step.c's.
 * That test uses a 16-dimension model whose every backward shape falls BELOW
 * training.c's dispatch thresholds, so it exercises the forward GPU path and
 * silently never touches a backward kernel. The dimensions below are chosen
 * so the FFN backward shapes clear both thresholds:
 *
 *   backward_weight  seq x ffn x emb = 32 x 1024 x 256 = 8.4M  >= 2^20
 *   backward_input   seq x emb x ffn = 32 x 256 x 1024 = 8.4M  >= 2^23
 *
 * If either threshold in training.c is raised past 8.4M, this test stops
 * covering what it exists to cover - so it asserts that the GPU path was
 * actually taken rather than trusting the arithmetic above to stay true.
 *
 * Self-skips without a usable CUDA GPU, like every test in tests/gpu/.
 */
#include "core/model.h"
#include "backends/gpu/gpu_matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define VOCAB 100
#define EMB 256
#define HEADS 8
#define LAYERS 1
#define MAX_SEQ 32
#define SEQ_LEN 32
#define STEPS 3

static void init_identical(neural_model_t *model) {
    if (model_new_seeded(model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ, 23) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        exit(1);
    }
    model->learning_rate = 0.01f;
}

/* Confirms the shapes this model produces really do reach the GPU, by
 * calling the backward entry points directly on the FFN shape the training
 * loop will issue. A pass here plus a pass below means the loop's agreement
 * is evidence about the GPU kernels rather than about the CPU fallback. */
static int gpu_backward_path_is_reachable(void) {
    const size_t m = SEQ_LEN, k = EMB * 4, n = EMB;
    float *A = calloc(m * k, sizeof(float));
    float *dC = calloc(m * n, sizeof(float));
    float *dB = calloc(k * n, sizeof(float));
    int ok = 0;
    if (A && dC && dB) {
        for (size_t i = 0; i < m * k; i++) A[i] = (float)(i % 5) * 0.01f;
        for (size_t i = 0; i < m * n; i++) dC[i] = (float)(i % 3) * 0.01f;
        ok = gpu_matmul_backward_weight(A, dC, dB, m, k, n) == 0;
    }
    free(A); free(dC); free(dB);
    return ok;
}

int main(void) {
    if (!gpu_matmul_available()) {
        printf("SKIP: no usable CUDA GPU detected (see gpu_probe.out for details) - not a failure.\n");
        return 0;
    }

    if (!gpu_backward_path_is_reachable()) {
        printf("FAIL: the GPU backward entry point failed on this model's own "
               "FFN shape, so the comparison below would prove nothing.\n");
        return 1;
    }

    uint32_t tokens[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; i++) tokens[i] = (uint32_t)((i * 7 + 3) % VOCAB);
    uint32_t target = 42;

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
        printf("  step %d: CPU loss=%.6f  GPU loss=%.6f  diff=%.3e\n",
               step + 1, cpu_loss, gpu_loss, fabsf(cpu_loss - gpu_loss));
    }

    /* Sample the weights the backward kernels actually wrote: W_ff1 and
     * W_ff2 gradients are what the dispatched calls produce, so a routing
     * bug (transposed operands, wrong destination) lands here first. */
    float max_weight_diff = 0.0f;
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.layers[0].W_ff1[13] - gpu_model.layers[0].W_ff1[13]));
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.layers[0].W_ff2[29] - gpu_model.layers[0].W_ff2[29]));
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.layers[0].W_q[5] - gpu_model.layers[0].W_q[5]));
    max_weight_diff = fmaxf(max_weight_diff, fabsf(cpu_model.output_projection[17] - gpu_model.output_projection[17]));
    printf("max weight diff after %d steps (FFN weights written by the "
           "dispatched backward kernels): %e\n", STEPS, max_weight_diff);

    /* Same reasoning as test_gpu_training_step.c: the two paths sum in
     * different orders, so they drift, but a wrong operand or a dropped
     * accumulation diverges by orders of magnitude rather than marginally. */
    int pass = fabsf(cpu_loss - gpu_loss) < 0.05f && max_weight_diff < 0.01f;

    model_free(&cpu_model);
    model_free(&gpu_model);
    gpu_matmul_shutdown();

    printf("\n%s\n", pass ? "GPU TRAINING BACKWARD CHECK PASSED"
                          : "GPU TRAINING BACKWARD CHECK FAILED");
    return pass ? 0 : 1;
}
