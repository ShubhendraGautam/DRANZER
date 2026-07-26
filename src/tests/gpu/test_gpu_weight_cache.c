/* Verifies gpu_matmul.c's weight-buffer cache invalidation contract: after
 * gpu_matmul_invalidate_weights() is called, a subsequent gpu_matmul()
 * call MUST reflect the current values at the weight pointer, not
 * whatever was uploaded the first time. This is the single most
 * safety-critical property of the caching optimization - training.c calls
 * gpu_matmul_invalidate_weights() after every optimizer step precisely
 * because weights change there, and a bug here would silently train
 * against stale GPU-side weights forever after the first step (wrong
 * results, not a crash - exactly the kind of bug that's easy to miss
 * without a dedicated test). Self-skips on machines without a CUDA GPU. */
#include "core/tensor_ops.h"
#include "backends/gpu/gpu_matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 8
#define K 12
#define N 10

int main(void) {
    if (!gpu_matmul_available()) {
        printf("SKIP: no usable CUDA GPU detected (see gpu_probe.out for details) - not a failure.\n");
        return 0;
    }

    float A[M * K];
    float B[K * N];
    float C_gpu[M * N];
    float C_cpu_ref[M * N];

    for (int i = 0; i < M * K; i++) A[i] = (float)((i % 5) - 2) * 0.3f;
    for (int i = 0; i < K * N; i++) B[i] = (float)((i % 3) - 1) * 0.5f;

    /* First call: populate the cache with the original B. */
    if (gpu_matmul(A, B, C_gpu, M, K, N) != 0) {
        printf("FAIL: first gpu_matmul() call failed\n");
        return 1;
    }
    matrix_multiply(A, B, C_cpu_ref, M, K, N);
    float first_call_diff = 0.0f;
    for (int i = 0; i < M * N; i++) first_call_diff = fmaxf(first_call_diff, fabsf(C_gpu[i] - C_cpu_ref[i]));
    printf("first call (B original): max diff vs CPU = %e\n", first_call_diff);

    /* Mutate B in place - same pointer, different values, exactly what an
     * optimizer step does to a model's weight buffer. */
    for (int i = 0; i < K * N; i++) B[i] = (float)((i % 7) - 3) * 0.8f;

    /* The contract: after invalidating, gpu_matmul() must pick up B's new
     * values, not the ones cached from the first call. */
    gpu_matmul_invalidate_weights();

    if (gpu_matmul(A, B, C_gpu, M, K, N) != 0) {
        printf("FAIL: second gpu_matmul() call failed\n");
        return 1;
    }
    matrix_multiply(A, B, C_cpu_ref, M, K, N); /* CPU reference against the NEW B */

    float second_call_diff = 0.0f;
    for (int i = 0; i < M * N; i++) second_call_diff = fmaxf(second_call_diff, fabsf(C_gpu[i] - C_cpu_ref[i]));
    printf("second call (B mutated, after invalidate): max diff vs CPU (recomputed with new B) = %e\n",
           second_call_diff);

    int pass = (first_call_diff < 1e-3f) && (second_call_diff < 1e-3f);

    gpu_matmul_shutdown();
    printf("\n%s\n", pass ? "GPU WEIGHT CACHE INVALIDATION CHECK PASSED" : "GPU WEIGHT CACHE INVALIDATION CHECK FAILED");
    return pass ? 0 : 1;
}
