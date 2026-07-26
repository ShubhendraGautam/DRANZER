/* Compares gpu_matmul() (hand-written PTX, via the CUDA Driver API) against
 * tensor_ops.c's matrix_multiply() (the CPU reference used everywhere else
 * in this project) at a deliberately non-power-of-2, multi-thread-block
 * size. Self-skips (reports success, not failure) on any machine without
 * a usable CUDA GPU - this test is inherently hardware-dependent, unlike
 * every other test in this suite, and its absence of a GPU is not a
 * regression. */
#include "../include/tensor_ops.h"
#include "../include/gpu_matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 37
#define K 53
#define N 41

int main(void) {
    if (!gpu_matmul_available()) {
        printf("SKIP: no usable CUDA GPU detected (see gpu_probe.out for details) - not a failure.\n");
        return 0;
    }

    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C_gpu = malloc(M * N * sizeof(float));
    float *C_cpu = malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = (float)((i % 7) - 3) * 0.5f;
    for (int i = 0; i < K * N; i++) B[i] = (float)((i % 5) - 2) * 0.25f;

    matrix_multiply(A, B, C_cpu, M, K, N);

    if (gpu_matmul(A, B, C_gpu, M, K, N) != 0) {
        printf("FAIL: gpu_matmul() reported available but the call itself failed.\n");
        return 1;
    }

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float d = fabsf(C_gpu[i] - C_cpu[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("M=%d K=%d N=%d, max diff GPU vs CPU: %e\n", M, K, N, max_diff);

    int pass = max_diff < 1e-3f;

    free(A); free(B); free(C_gpu); free(C_cpu);
    gpu_matmul_shutdown();

    printf("\n%s\n", pass ? "GPU MATMUL CHECK PASSED" : "GPU MATMUL CHECK FAILED");
    return pass ? 0 : 1;
}
