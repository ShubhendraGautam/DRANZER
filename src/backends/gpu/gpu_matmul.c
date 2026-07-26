/*
 * GPU matmul via a hand-written PTX kernel (naive, one thread per output
 * element, no tiling/shared-memory - correctness-first, matching this
 * project's existing CPU matmul in spirit before any GPU-side
 * optimization). Validated against tensor_ops.c's matrix_multiply() in
 * tests/test_gpu_matmul.c.
 *
 * PTX below computes C[row][col] = sum_l A[row][l] * B[l][col] for the
 * thread's (row, col), skipping out-of-bounds threads via a predicated
 * branch. Row/col come from a 2D grid of 2D blocks (ctaid/ntid/tid.x for
 * col, .y for row).
 */

#include "include/gpu_matmul.h"
#include "include/gpu_cuda.h"
#include <stddef.h>

static const char *MATMUL_PTX =
".version 7.0\n.target sm_75\n.address_size 64\n\n"
".visible .entry matmul_naive(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_K,\n"
"    .param .u32 param_N\n"
")\n{\n"
"    .reg .pred %p1, %p2, %p3;\n"
"    .reg .f32 %f1, %f2, %f3;\n"
"    .reg .b32 %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10;\n"
"    .reg .b64 %rd1, %rd2, %rd3, %rd4, %rd5, %rd6, %rd7, %rd8, %rd9, %rd10;\n\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_C];\n"
"    ld.param.u32 %r6, [param_M];\n"
"    ld.param.u32 %r7, [param_K];\n"
"    ld.param.u32 %r8, [param_N];\n\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.s32 %r4, %r1, %r2, %r3;\n"
"    mov.u32 %r1, %ctaid.y;\n"
"    mov.u32 %r2, %ntid.y;\n"
"    mov.u32 %r3, %tid.y;\n"
"    mad.lo.s32 %r5, %r1, %r2, %r3;\n\n"
"    setp.ge.s32 %p1, %r5, %r6;\n"
"    setp.ge.s32 %p2, %r4, %r8;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra DONE;\n\n"
"    cvta.to.global.u64 %rd4, %rd1;\n"
"    cvta.to.global.u64 %rd5, %rd2;\n"
"    cvta.to.global.u64 %rd6, %rd3;\n\n"
"    mul.lo.s32 %r9, %r5, %r7;\n"
"    mul.wide.s32 %rd7, %r9, 4;\n"
"    add.s64 %rd7, %rd4, %rd7;\n\n"
"    mul.wide.s32 %rd8, %r4, 4;\n"
"    add.s64 %rd8, %rd5, %rd8;\n\n"
"    mul.wide.s32 %rd9, %r8, 4;\n\n"
"    mov.f32 %f3, 0f00000000;\n"
"    mov.u32 %r10, 0;\n\n"
"LOOP_START:\n"
"    setp.ge.s32 %p1, %r10, %r7;\n"
"    @%p1 bra LOOP_END;\n\n"
"    ld.global.f32 %f1, [%rd7];\n"
"    ld.global.f32 %f2, [%rd8];\n"
"    fma.rn.f32 %f3, %f1, %f2, %f3;\n\n"
"    add.s64 %rd7, %rd7, 4;\n"
"    add.s64 %rd8, %rd8, %rd9;\n"
"    add.s32 %r10, %r10, 1;\n"
"    bra LOOP_START;\n\n"
"LOOP_END:\n"
"    mul.lo.s32 %r9, %r5, %r8;\n"
"    add.s32 %r9, %r9, %r4;\n"
"    mul.wide.s32 %rd10, %r9, 4;\n"
"    add.s64 %rd10, %rd6, %rd10;\n"
"    st.global.f32 [%rd10], %f3;\n\n"
"DONE:\n"
"    ret;\n"
"}\n";

static gpu_cuda_ctx_t *g_ctx = NULL;
static int g_init_attempted = 0;
static int g_kernel_loaded = 0;

/* Lazily initializes g_ctx and loads the matmul kernel exactly once,
 * regardless of how many times gpu_matmul()/gpu_matmul_available() are
 * called - subsequent calls reuse the same context. */
static int ensure_ready(void) {
    if (g_init_attempted) return g_ctx != NULL && g_kernel_loaded;
    g_init_attempted = 1;

    g_ctx = gpu_cuda_init();
    if (!g_ctx) return 0;

    if (gpu_cuda_load_kernel(g_ctx, MATMUL_PTX, "matmul_naive") != 0) {
        gpu_cuda_shutdown(g_ctx);
        g_ctx = NULL;
        return 0;
    }

    g_kernel_loaded = 1;
    return 1;
}

int gpu_matmul_available(void) {
    return ensure_ready();
}

int gpu_matmul(const float *A, const float *B, float *C, size_t m, size_t k, size_t n) {
    if (!ensure_ready()) return -1;

    uint64_t d_A = gpu_cuda_alloc(g_ctx, m * k * sizeof(float));
    uint64_t d_B = gpu_cuda_alloc(g_ctx, k * n * sizeof(float));
    uint64_t d_C = gpu_cuda_alloc(g_ctx, m * n * sizeof(float));
    if (!d_A || !d_B || !d_C) {
        gpu_cuda_free(g_ctx, d_A);
        gpu_cuda_free(g_ctx, d_B);
        gpu_cuda_free(g_ctx, d_C);
        return -1;
    }

    int rc = -1;
    if (gpu_cuda_upload(g_ctx, d_A, A, m * k * sizeof(float)) != 0) goto done;
    if (gpu_cuda_upload(g_ctx, d_B, B, k * n * sizeof(float)) != 0) goto done;

    {
        unsigned int m32 = (unsigned int)m, k32 = (unsigned int)k, n32 = (unsigned int)n;
        void *args[] = { &d_A, &d_B, &d_C, &m32, &k32, &n32 };
        const unsigned int BLOCK = 16;
        unsigned int grid_x = ((unsigned int)n + BLOCK - 1) / BLOCK;
        unsigned int grid_y = ((unsigned int)m + BLOCK - 1) / BLOCK;
        if (gpu_cuda_launch_2d(g_ctx, grid_x, grid_y, BLOCK, BLOCK, args) != 0) goto done;
    }

    if (gpu_cuda_download(g_ctx, C, d_C, m * n * sizeof(float)) != 0) goto done;
    rc = 0;

done:
    gpu_cuda_free(g_ctx, d_A);
    gpu_cuda_free(g_ctx, d_B);
    gpu_cuda_free(g_ctx, d_C);
    return rc;
}

void gpu_matmul_shutdown(void) {
    if (g_ctx) gpu_cuda_shutdown(g_ctx);
    g_ctx = NULL;
    g_init_attempted = 0;
    g_kernel_loaded = 0;
}
