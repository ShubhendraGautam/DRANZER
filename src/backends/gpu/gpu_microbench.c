/*
 * Empirical throughput measurement via hand-written PTX, loaded through
 * gpu_cuda.h (same pattern as gpu_matmul.c). See gpu_microbench.h.
 */

#include "backends/gpu/gpu_microbench.h"
#include "backends/gpu/gpu_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* C[i] = A[i] for i < N. Identical pattern to the validated copy kernel
 * from this project's PTX feasibility testing - one read, one write, no
 * cross-thread dependency, so achieved bandwidth is bound only by the
 * memory subsystem. */
static const char *COPY_PTX =
".version 7.0\n.target sm_75\n.address_size 64\n\n"
".visible .entry copy1d(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_N\n"
")\n{\n"
"    .reg .pred %p1;\n"
"    .reg .f32 %f1;\n"
"    .reg .b32 %r1, %r2, %r3, %r4, %r5;\n"
"    .reg .b64 %rd1, %rd2, %rd3, %rd4, %rd5;\n\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_C];\n"
"    ld.param.u32 %r2, [param_N];\n"
"    mov.u32 %r3, %ntid.x;\n"
"    mov.u32 %r4, %ctaid.x;\n"
"    mov.u32 %r5, %tid.x;\n"
"    mad.lo.s32 %r1, %r4, %r3, %r5;\n"
"    setp.ge.s32 %p1, %r1, %r2;\n"
"    @%p1 bra DONE;\n\n"
"    mul.wide.s32 %rd3, %r1, 4;\n"
"    cvta.to.global.u64 %rd4, %rd1;\n"
"    add.s64 %rd4, %rd4, %rd3;\n"
"    ld.global.f32 %f1, [%rd4];\n"
"    cvta.to.global.u64 %rd5, %rd2;\n"
"    add.s64 %rd5, %rd5, %rd3;\n"
"    st.global.f32 [%rd5], %f1;\n\n"
"DONE:\n"
"    ret;\n"
"}\n";

/* Each thread runs a long serially-dependent FMA chain (each iteration
 * depends on the previous one's result, so the loop can't be reordered
 * away) and writes the final value once. mult/add are passed as runtime
 * kernel parameters rather than baked into the PTX as immediates -
 * that's what prevents the driver's JIT from constant-folding or
 * eliminating the loop: it has no way to know their values at compile
 * time. Using mult=1.0 keeps the accumulator numerically bounded
 * regardless of iteration count (pure bounded accumulation, not
 * exponential growth), so this is safe to run with a large iteration
 * count without risking inf/nan. */
static const char *FMA_PTX =
".version 7.0\n.target sm_75\n.address_size 64\n\n"
".visible .entry fma_bench(\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_iters,\n"
"    .param .f32 param_mult,\n"
"    .param .f32 param_add\n"
")\n{\n"
"    .reg .pred %p1;\n"
"    .reg .f32 %f1, %f2, %f3;\n"
"    .reg .b32 %r1, %r2, %r3, %r4, %r5, %r6;\n"
"    .reg .b64 %rd1, %rd2, %rd3;\n\n"
"    ld.param.u64 %rd1, [param_C];\n"
"    ld.param.u32 %r2, [param_iters];\n"
"    ld.param.f32 %f2, [param_mult];\n"
"    ld.param.f32 %f3, [param_add];\n"
"    mov.u32 %r3, %ntid.x;\n"
"    mov.u32 %r4, %ctaid.x;\n"
"    mov.u32 %r5, %tid.x;\n"
"    mad.lo.s32 %r1, %r4, %r3, %r5;\n"
"    cvt.rn.f32.s32 %f1, %r1;\n"
"    mov.u32 %r6, 0;\n\n"
"LOOP:\n"
"    setp.ge.s32 %p1, %r6, %r2;\n"
"    @%p1 bra DONE;\n"
"    fma.rn.f32 %f1, %f1, %f2, %f3;\n"
"    add.s32 %r6, %r6, 1;\n"
"    bra LOOP;\n\n"
"DONE:\n"
"    mul.wide.s32 %rd2, %r1, 4;\n"
"    cvta.to.global.u64 %rd3, %rd1;\n"
"    add.s64 %rd3, %rd3, %rd2;\n"
"    st.global.f32 [%rd3], %f1;\n"
"    ret;\n"
"}\n";

#define COPY_N (32 * 1024 * 1024) /* 32M floats = 128 MB per buffer */
#define FMA_THREADS_PER_BLOCK 256
#define FMA_BLOCKS 1024 /* 262144 threads total */
#define FMA_ITERS 200000

static int bench_bandwidth(gpu_cuda_ctx_t *ctx, double *out_gbps) {
    if (gpu_cuda_load_kernel(ctx, COPY_PTX, "copy1d") != 0) return -1;

    uint64_t d_A = gpu_cuda_alloc(ctx, (size_t)COPY_N * sizeof(float));
    uint64_t d_C = gpu_cuda_alloc(ctx, (size_t)COPY_N * sizeof(float));
    if (!d_A || !d_C) { gpu_cuda_free(ctx, d_A); gpu_cuda_free(ctx, d_C); return -1; }

    unsigned int n = COPY_N;
    void *args[] = { &d_A, &d_C, &n };
    unsigned int block = 256;
    unsigned int grid = (COPY_N + block - 1) / block;

    if (gpu_cuda_launch_2d(ctx, grid, 1, block, 1, args) != 0) { /* warm up */
        gpu_cuda_free(ctx, d_A); gpu_cuda_free(ctx, d_C);
        return -1;
    }

    const int iters = 10;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        if (gpu_cuda_launch_2d(ctx, grid, 1, block, 1, args) != 0) {
            gpu_cuda_free(ctx, d_A); gpu_cuda_free(ctx, d_C);
            return -1;
        }
    }
    double elapsed = now_sec() - t0;

    double bytes_per_iter = 2.0 * (double)COPY_N * sizeof(float); /* 1 read + 1 write */
    *out_gbps = (bytes_per_iter * iters) / elapsed / 1e9;

    gpu_cuda_free(ctx, d_A);
    gpu_cuda_free(ctx, d_C);
    return 0;
}

static int bench_fma(gpu_cuda_ctx_t *ctx, double *out_gflops) {
    if (gpu_cuda_load_kernel(ctx, FMA_PTX, "fma_bench") != 0) return -1;

    unsigned int total_threads = FMA_THREADS_PER_BLOCK * FMA_BLOCKS;
    uint64_t d_C = gpu_cuda_alloc(ctx, (size_t)total_threads * sizeof(float));
    if (!d_C) return -1;

    unsigned int iters = FMA_ITERS;
    float mult = 1.0f, add = 0.0000001f;
    void *args[] = { &d_C, &iters, &mult, &add };

    if (gpu_cuda_launch_2d(ctx, FMA_BLOCKS, 1, FMA_THREADS_PER_BLOCK, 1, args) != 0) { /* warm up */
        gpu_cuda_free(ctx, d_C);
        return -1;
    }

    double t0 = now_sec();
    if (gpu_cuda_launch_2d(ctx, FMA_BLOCKS, 1, FMA_THREADS_PER_BLOCK, 1, args) != 0) {
        gpu_cuda_free(ctx, d_C);
        return -1;
    }
    double elapsed = now_sec() - t0;

    double total_flops = (double)total_threads * (double)iters * 2.0; /* FMA = 2 FLOPs */
    *out_gflops = total_flops / elapsed / 1e9;

    gpu_cuda_free(ctx, d_C);
    return 0;
}

void gpu_microbench_run(gpu_microbench_result_t *out) {
    out->available = 0;
    out->bandwidth_gbps = 0.0;
    out->fma_gflops = 0.0;

    gpu_cuda_ctx_t *ctx = gpu_cuda_init();
    if (!ctx) return;

    int ok_bw = (bench_bandwidth(ctx, &out->bandwidth_gbps) == 0);
    int ok_fma = (bench_fma(ctx, &out->fma_gflops) == 0);
    out->available = ok_bw || ok_fma;

    gpu_cuda_shutdown(ctx);
}

void gpu_microbench_print(const gpu_microbench_result_t *r) {
    if (!r->available) {
        printf("    Not available.\n");
        return;
    }
    printf("    Measured memory bandwidth (copy kernel): %.1f GB/s\n", r->bandwidth_gbps);
    printf("    Measured FP32 FMA throughput: %.2f GFLOPS\n", r->fma_gflops);
}
