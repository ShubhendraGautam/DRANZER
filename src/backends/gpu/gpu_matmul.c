/*
 * GPU matmul via a hand-written PTX kernel (naive, one thread per output
 * element, no tiling/shared-memory - correctness-first, matching this
 * project's existing CPU matmul in spirit before any GPU-side
 * optimization). Validated against tensor_ops.c's matrix_multiply() in
 * tests/test_gpu_matmul.c and tests/test_gpu_model_forward.c.
 *
 * PTX below computes C[row][col] = sum_l A[row][l] * B[l][col] for the
 * thread's (row, col), skipping out-of-bounds threads via a predicated
 * branch. Row/col come from a 2D grid of 2D blocks (ctaid/ntid/tid.x for
 * col, .y for row).
 *
 * Two caches keep repeated calls from paying for the same
 * alloc/upload/download/free cycle every time - see gpu_matmul.h for why
 * this exists (the naive version was 20-80x slower than CPU at this
 * project's typical model sizes) and tests/test_gpu_weight_cache.c +
 * tests/test_gpu_training_step.c for how the trickiest part - not serving
 * stale weight data after training updates them, or after a model is
 * freed and a new one happens to reuse the same heap address - is verified.
 */

#include "backends/gpu/gpu_matmul.h"
#include "backends/gpu/gpu_cuda.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

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
"}\n"

/* C = A @ B, with each 16x16 output tile staged through shared memory.
 *
 * The naive kernel above re-reads both operands from global memory for every
 * output element: a thread computing C[row][col] streams a whole row of A and
 * a whole column of B on its own. Measured on this project's MX450 that runs
 * at roughly 160 GFLOP/s on prefill shapes - about 7% of the 2348 GFLOP/s the
 * same device reaches on a pure FMA microbenchmark (docs/gpu.md), so the
 * ceiling is memory traffic rather than arithmetic.
 *
 * This version walks k in 16-wide steps. On each step the block cooperatively
 * loads one 16x16 tile of A and one of B into shared memory, synchronizes,
 * and then every thread reads its 16 multiply-accumulate operands from shared
 * rather than global. Each loaded element is used by 16 threads instead of 1,
 * which cuts global traffic by the tile width.
 *
 * Both bar.sync calls are reached by every thread in the block: the bounds
 * checks around the two loads write a zero into shared memory rather than
 * branching past the barrier, because a barrier inside divergent control flow
 * is undefined. Zero is the right filler - it contributes nothing to the
 * accumulation - which is also what makes ragged edges correct without a
 * separate remainder kernel.
 *
 * Accumulation order per output element is unchanged from the naive kernel:
 * still k ascending, just fetched differently. */
".visible .entry matmul_tiled(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_K,\n"
"    .param .u32 param_N\n"
")\n{\n"
"    .reg .pred %p<8>;\n"
"    .reg .f32 %f<8>;\n"
"    .reg .b32 %r<32>;\n"
"    .reg .b64 %rd<16>;\n"
"    .shared .align 4 .b8 tileA[1024];\n"
"    .shared .align 4 .b8 tileB[1024];\n\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_C];\n"
"    ld.param.u32 %r1, [param_M];\n"
"    ld.param.u32 %r2, [param_K];\n"
"    ld.param.u32 %r3, [param_N];\n\n"
"    cvta.to.global.u64 %rd4, %rd1;\n"
"    cvta.to.global.u64 %rd5, %rd2;\n"
"    cvta.to.global.u64 %rd6, %rd3;\n\n"
"    mov.u32 %r4, %tid.x;\n"
"    mov.u32 %r5, %tid.y;\n"
"    mov.u32 %r6, %ctaid.x;\n"
"    mov.u32 %r7, %ctaid.y;\n"
"    shl.b32 %r8, %r6, 4;\n"
"    add.s32 %r8, %r8, %r4;\n"          /* col */
"    shl.b32 %r9, %r7, 4;\n"
"    add.s32 %r9, %r9, %r5;\n\n"        /* row */
"    mov.u32 %r10, tileA;\n"
"    mov.u32 %r11, tileB;\n"
"    shl.b32 %r12, %r5, 4;\n"
"    add.s32 %r12, %r12, %r4;\n"
"    shl.b32 %r12, %r12, 2;\n"
"    add.s32 %r13, %r10, %r12;\n"       /* &tileA[ty][tx] */
"    add.s32 %r14, %r11, %r12;\n\n"     /* &tileB[ty][tx] */
"    mov.f32 %f1, 0f00000000;\n"
"    add.s32 %r15, %r2, 15;\n"
"    shr.s32 %r15, %r15, 4;\n"          /* tiles = ceil(K/16) */
"    mov.u32 %r16, 0;\n\n"
"TILE_LOOP:\n"
"    setp.ge.s32 %p1, %r16, %r15;\n"
"    @%p1 bra TILE_DONE;\n\n"
"    shl.b32 %r17, %r16, 4;\n"
"    add.s32 %r18, %r17, %r4;\n"        /* a_col = t*16 + tx */
"    mov.f32 %f2, 0f00000000;\n"
"    setp.lt.s32 %p2, %r9, %r1;\n"
"    setp.lt.s32 %p3, %r18, %r2;\n"
"    and.pred %p4, %p2, %p3;\n"
"    @!%p4 bra SKIP_A;\n"
"    mul.lo.s32 %r19, %r9, %r2;\n"
"    add.s32 %r19, %r19, %r18;\n"
"    mul.wide.s32 %rd7, %r19, 4;\n"
"    add.s64 %rd7, %rd4, %rd7;\n"
"    ld.global.f32 %f2, [%rd7];\n"
"SKIP_A:\n"
"    st.shared.f32 [%r13], %f2;\n\n"
"    add.s32 %r20, %r17, %r5;\n"        /* b_row = t*16 + ty */
"    mov.f32 %f3, 0f00000000;\n"
"    setp.lt.s32 %p5, %r20, %r2;\n"
"    setp.lt.s32 %p6, %r8, %r3;\n"
"    and.pred %p7, %p5, %p6;\n"
"    @!%p7 bra SKIP_B;\n"
"    mul.lo.s32 %r21, %r20, %r3;\n"
"    add.s32 %r21, %r21, %r8;\n"
"    mul.wide.s32 %rd8, %r21, 4;\n"
"    add.s64 %rd8, %rd5, %rd8;\n"
"    ld.global.f32 %f3, [%rd8];\n"
"SKIP_B:\n"
"    st.shared.f32 [%r14], %f3;\n\n"
"    bar.sync 0;\n\n"
"    shl.b32 %r22, %r5, 6;\n"
"    add.s32 %r22, %r10, %r22;\n"       /* &tileA[ty][0] */
"    shl.b32 %r23, %r4, 2;\n"
"    add.s32 %r23, %r11, %r23;\n"       /* &tileB[0][tx] */
"    mov.u32 %r24, 0;\n"
"INNER_LOOP:\n"
"    setp.ge.s32 %p1, %r24, 16;\n"
"    @%p1 bra INNER_DONE;\n"
"    ld.shared.f32 %f4, [%r22];\n"
"    ld.shared.f32 %f5, [%r23];\n"
"    fma.rn.f32 %f1, %f4, %f5, %f1;\n"
"    add.s32 %r22, %r22, 4;\n"
"    add.s32 %r23, %r23, 64;\n"
"    add.s32 %r24, %r24, 1;\n"
"    bra INNER_LOOP;\n"
"INNER_DONE:\n\n"
"    bar.sync 0;\n\n"
"    add.s32 %r16, %r16, 1;\n"
"    bra TILE_LOOP;\n\n"
"TILE_DONE:\n"
"    setp.ge.s32 %p1, %r9, %r1;\n"
"    setp.ge.s32 %p2, %r8, %r3;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra STORE_DONE;\n"
"    mul.lo.s32 %r25, %r9, %r3;\n"
"    add.s32 %r25, %r25, %r8;\n"
"    mul.wide.s32 %rd9, %r25, 4;\n"
"    add.s64 %rd9, %rd6, %rd9;\n"
"    st.global.f32 [%rd9], %f1;\n"
"STORE_DONE:\n"
"    ret;\n"
"}\n"

/* dA (m x k) += dC (m x n) @ B_transposed (n x k), i.e.
 *     dA[i][l] += sum_j dC[i][j] * B[l][j]
 * one thread per (i, l): x indexes l against K, y indexes i against M.
 *
 * B is not transposed in memory - the kernel walks row l of B and row i of
 * dC together, both contiguous in j, so the transpose costs nothing. This is
 * the same access pattern matmul_backward_input() uses on the CPU.
 *
 * The store accumulates. Every backward call site adds into a gradient buffer
 * that a whole minibatch shares (see training.c), so the destination arrives
 * with a meaningful value and must be read before it is written. */
".visible .entry matmul_backward_input_naive(\n"
"    .param .u64 param_dC,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_dA,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_K,\n"
"    .param .u32 param_N\n"
")\n{\n"
"    .reg .pred %p1, %p2, %p3;\n"
"    .reg .f32 %f1, %f2, %f3, %f4;\n"
"    .reg .b32 %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10;\n"
"    .reg .b64 %rd1, %rd2, %rd3, %rd4, %rd5, %rd6, %rd7, %rd8, %rd10;\n\n"
"    ld.param.u64 %rd1, [param_dC];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_dA];\n"
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
"    setp.ge.s32 %p2, %r4, %r7;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra BI_DONE;\n\n"
"    cvta.to.global.u64 %rd4, %rd1;\n"
"    cvta.to.global.u64 %rd5, %rd2;\n"
"    cvta.to.global.u64 %rd6, %rd3;\n\n"
"    mul.lo.s32 %r9, %r5, %r8;\n"
"    mul.wide.s32 %rd7, %r9, 4;\n"
"    add.s64 %rd7, %rd4, %rd7;\n\n"
"    mul.lo.s32 %r9, %r4, %r8;\n"
"    mul.wide.s32 %rd8, %r9, 4;\n"
"    add.s64 %rd8, %rd5, %rd8;\n\n"
"    mov.f32 %f3, 0f00000000;\n"
"    mov.u32 %r10, 0;\n\n"
"BI_LOOP_START:\n"
"    setp.ge.s32 %p1, %r10, %r8;\n"
"    @%p1 bra BI_LOOP_END;\n\n"
"    ld.global.f32 %f1, [%rd7];\n"
"    ld.global.f32 %f2, [%rd8];\n"
"    fma.rn.f32 %f3, %f1, %f2, %f3;\n\n"
"    add.s64 %rd7, %rd7, 4;\n"
"    add.s64 %rd8, %rd8, 4;\n"
"    add.s32 %r10, %r10, 1;\n"
"    bra BI_LOOP_START;\n\n"
"BI_LOOP_END:\n"
"    mul.lo.s32 %r9, %r5, %r7;\n"
"    add.s32 %r9, %r9, %r4;\n"
"    mul.wide.s32 %rd10, %r9, 4;\n"
"    add.s64 %rd10, %rd6, %rd10;\n"
"    ld.global.f32 %f4, [%rd10];\n"
"    add.f32 %f3, %f4, %f3;\n"
"    st.global.f32 [%rd10], %f3;\n\n"
"BI_DONE:\n"
"    ret;\n"
"}\n"

/* dB (k x n) += A_transposed (k x m) @ dC (m x n), i.e.
 *     dB[l][j] += sum_i A[i][l] * dC[i][j]
 * one thread per (l, j): x indexes j against N, y indexes l against K.
 *
 * Unlike the other two kernels this one strides both inputs - A by K and dC
 * by N per step of i - because the reduction runs down the rows. Adjacent
 * threads still read adjacent j, so the dC loads coalesce across a warp;
 * the A load is a broadcast within a row of threads. Accumulates, for the
 * same reason as above. */
".visible .entry matmul_backward_weight_naive(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_dC,\n"
"    .param .u64 param_dB,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_K,\n"
"    .param .u32 param_N\n"
")\n{\n"
"    .reg .pred %p1, %p2, %p3;\n"
"    .reg .f32 %f1, %f2, %f3, %f4;\n"
"    .reg .b32 %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10;\n"
"    .reg .b64 %rd1, %rd2, %rd3, %rd4, %rd5, %rd6, %rd7, %rd8, %rd9, %rd10, %rd11;\n\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_dC];\n"
"    ld.param.u64 %rd3, [param_dB];\n"
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
"    setp.ge.s32 %p1, %r5, %r7;\n"
"    setp.ge.s32 %p2, %r4, %r8;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra BW_DONE;\n\n"
"    cvta.to.global.u64 %rd4, %rd1;\n"
"    cvta.to.global.u64 %rd5, %rd2;\n"
"    cvta.to.global.u64 %rd6, %rd3;\n\n"
"    mul.wide.s32 %rd7, %r5, 4;\n"
"    add.s64 %rd7, %rd4, %rd7;\n\n"
"    mul.wide.s32 %rd8, %r4, 4;\n"
"    add.s64 %rd8, %rd5, %rd8;\n\n"
"    mul.wide.s32 %rd9, %r7, 4;\n"
"    mul.wide.s32 %rd11, %r8, 4;\n\n"
"    mov.f32 %f3, 0f00000000;\n"
"    mov.u32 %r10, 0;\n\n"
"BW_LOOP_START:\n"
"    setp.ge.s32 %p1, %r10, %r6;\n"
"    @%p1 bra BW_LOOP_END;\n\n"
"    ld.global.f32 %f1, [%rd7];\n"
"    ld.global.f32 %f2, [%rd8];\n"
"    fma.rn.f32 %f3, %f1, %f2, %f3;\n\n"
"    add.s64 %rd7, %rd7, %rd9;\n"
"    add.s64 %rd8, %rd8, %rd11;\n"
"    add.s32 %r10, %r10, 1;\n"
"    bra BW_LOOP_START;\n\n"
"BW_LOOP_END:\n"
"    mul.lo.s32 %r9, %r5, %r8;\n"
"    add.s32 %r9, %r9, %r4;\n"
"    mul.wide.s32 %rd10, %r9, 4;\n"
"    add.s64 %rd10, %rd6, %rd10;\n"
"    ld.global.f32 %f4, [%rd10];\n"
"    add.f32 %f3, %f4, %f3;\n"
"    st.global.f32 [%rd10], %f3;\n\n"
"BW_DONE:\n"
"    ret;\n"
"}\n";

static gpu_cuda_ctx_t *g_ctx = NULL;
static int g_init_attempted = 0;
static int g_kernel_loaded = 0;

/* All entry points live in one PTX module, so the driver JITs once and
 * switching between them costs nothing at launch time. */
static gpu_cuda_kernel_t *g_k_forward_naive = NULL;
static gpu_cuda_kernel_t *g_k_forward_tiled = NULL;
static gpu_cuda_kernel_t *g_k_backward_input = NULL;
static gpu_cuda_kernel_t *g_k_backward_weight = NULL;
static int g_environment_checked = 0;
static int g_environment_requests_naive = 0;
static gpu_matmul_config_status_t g_config_status = GPU_MATMUL_CONFIG_OK;
static char g_invalid_environment_value[64];

/* Which forward kernel gpu_matmul() launches. Resolved once, from
 * DRANZER_GPU_MATMUL, so the two can be compared in the same session the way
 * DRANZER_CPU_ISA compares CPU paths - that is how the choice below was
 * measured rather than assumed. Unset means the measured default. */
static gpu_cuda_kernel_t *g_k_forward = NULL;

static void parse_forward_environment(void) {
    if (g_environment_checked) return;
    g_environment_checked = 1;
    const char *requested = getenv("DRANZER_GPU_MATMUL");
    if (requested && strcmp(requested, "naive") == 0) {
        g_environment_requests_naive = 1;
        return;
    }
    if (requested && strcmp(requested, "tiled") != 0 && requested[0] != '\0') {
        g_config_status = GPU_MATMUL_CONFIG_INVALID_ENV;
        strncpy(g_invalid_environment_value, requested,
                sizeof(g_invalid_environment_value) - 1);
        g_invalid_environment_value[sizeof(g_invalid_environment_value) - 1] = '\0';
    }
}

static void select_forward_kernel(void) {
    parse_forward_environment();
    g_k_forward = g_environment_requests_naive
                      ? g_k_forward_naive : g_k_forward_tiled;
}

/* Weight buffer cache: keyed by host pointer, revalidated by generation
 * number rather than re-checking content. gpu_matmul_invalidate_weights()
 * bumps the generation - called after every optimizer step (training.c)
 * AND on every model_new()/model_free() (model.c), the latter because a
 * freed model's weight buffer can be reused by a subsequent malloc for a
 * *different* model at the same address; bumping generation there forces
 * a re-upload (reading whatever is actually at that address now) instead
 * of serving the previous model's cached bytes under a coincidentally
 * matching pointer. */
#define WEIGHT_CACHE_MAX_ENTRIES 256

typedef struct {
    const float *host_ptr;
    uint64_t device_ptr;
    size_t bytes;
    uint64_t generation;
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[WEIGHT_CACHE_MAX_ENTRIES];
static int g_weight_cache_count = 0;
static uint64_t g_weight_generation = 1; /* 0 is never assigned, so a zeroed struct is never mistaken for cached */

/* Persistent scratch for A (activations, uploaded fresh every call) and C
 * (output, downloaded fresh every call) - grown on demand, never shrunk,
 * reused across calls regardless of content (always fully overwritten by
 * the upload/kernel before being read). */
static uint64_t g_scratch_A = 0, g_scratch_C = 0;
static size_t g_scratch_A_bytes = 0, g_scratch_C_bytes = 0;

/* Backward needs a third live buffer: its destination is read-modify-write,
 * so the gradient being accumulated into cannot share scratch with either
 * input. Kept separate from A/C rather than sized into them because the
 * backward shapes and the forward shapes are different. */
static uint64_t g_scratch_D = 0;
static size_t g_scratch_D_bytes = 0;

static int ensure_ready(void) {
    if (g_init_attempted) return g_ctx != NULL && g_kernel_loaded;
    g_init_attempted = 1;

    g_ctx = gpu_cuda_init();
    if (!g_ctx) return 0;

    if (gpu_cuda_load_module(g_ctx, MATMUL_PTX) != 0) {
        gpu_cuda_shutdown(g_ctx);
        g_ctx = NULL;
        return 0;
    }

    g_k_forward_naive = gpu_cuda_resolve_kernel(g_ctx, "matmul_naive");
    g_k_forward_tiled = gpu_cuda_resolve_kernel(g_ctx, "matmul_tiled");
    g_k_backward_input = gpu_cuda_resolve_kernel(g_ctx, "matmul_backward_input_naive");
    g_k_backward_weight = gpu_cuda_resolve_kernel(g_ctx, "matmul_backward_weight_naive");
    if (!g_k_forward_naive || !g_k_forward_tiled ||
        !g_k_backward_input || !g_k_backward_weight) {
        gpu_cuda_shutdown(g_ctx);
        g_ctx = NULL;
        return 0;
    }
    select_forward_kernel();

    g_kernel_loaded = 1;
    return 1;
}

int gpu_matmul_available(void) {
    return ensure_ready();
}

static uint64_t get_scratch_buffer(uint64_t *cached_ptr, size_t *cached_bytes, size_t needed_bytes) {
    if (*cached_ptr && *cached_bytes >= needed_bytes) return *cached_ptr;
    if (*cached_ptr) gpu_cuda_free(g_ctx, *cached_ptr);
    *cached_ptr = gpu_cuda_alloc(g_ctx, needed_bytes);
    *cached_bytes = *cached_ptr ? needed_bytes : 0;
    return *cached_ptr;
}

/* Returns a device buffer holding a fresh copy of host_ptr[0..bytes), by
 * re-using and re-uploading into a previously-allocated buffer for this
 * exact (pointer, size) if its generation is current, or allocating a new
 * one (evicting nothing - the cache is generously sized, see
 * WEIGHT_CACHE_MAX_ENTRIES) otherwise. Returns 0 on failure (cache full,
 * or an alloc/upload error) - caller falls back to CPU for that call. */
static uint64_t get_cached_weight_buffer(const float *host_ptr, size_t bytes) {
    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].host_ptr == host_ptr && g_weight_cache[i].bytes == bytes) {
            if (g_weight_cache[i].generation != g_weight_generation) {
                if (gpu_cuda_upload(g_ctx, g_weight_cache[i].device_ptr, host_ptr, bytes) != 0) return 0;
                g_weight_cache[i].generation = g_weight_generation;
            }
            return g_weight_cache[i].device_ptr;
        }
    }

    if (g_weight_cache_count >= WEIGHT_CACHE_MAX_ENTRIES) return 0;

    uint64_t dptr = gpu_cuda_alloc(g_ctx, bytes);
    if (!dptr) return 0;
    if (gpu_cuda_upload(g_ctx, dptr, host_ptr, bytes) != 0) {
        gpu_cuda_free(g_ctx, dptr);
        return 0;
    }

    g_weight_cache[g_weight_cache_count].host_ptr = host_ptr;
    g_weight_cache[g_weight_cache_count].device_ptr = dptr;
    g_weight_cache[g_weight_cache_count].bytes = bytes;
    g_weight_cache[g_weight_cache_count].generation = g_weight_generation;
    g_weight_cache_count++;
    return dptr;
}

int gpu_matmul_set_forward_kernel(const char *name) {
    if (!name || !ensure_ready()) return -1;
    if (strcmp(name, "tiled") == 0) {
        g_k_forward = g_k_forward_tiled;
        return 0;
    }
    if (strcmp(name, "naive") == 0) {
        g_k_forward = g_k_forward_naive;
        return 0;
    }
    return -1;
}

const char *gpu_matmul_forward_kernel_name(void) {
    if (!ensure_ready()) return "unavailable";
    return g_k_forward == g_k_forward_naive ? "naive" : "tiled";
}

gpu_matmul_config_status_t gpu_matmul_config_status(void) {
    parse_forward_environment();
    return g_config_status;
}

const char *gpu_matmul_invalid_environment_value(void) {
    parse_forward_environment();
    return g_config_status == GPU_MATMUL_CONFIG_INVALID_ENV
               ? g_invalid_environment_value : NULL;
}

void gpu_matmul_invalidate_weights(void) {
    g_weight_generation++;
}

int gpu_matmul(const float *A, const float *B, float *C, size_t m, size_t k, size_t n) {
    if (!ensure_ready()) return -1;

    uint64_t d_B = get_cached_weight_buffer(B, k * n * sizeof(float));
    if (!d_B) return -1;

    uint64_t d_A = get_scratch_buffer(&g_scratch_A, &g_scratch_A_bytes, m * k * sizeof(float));
    uint64_t d_C = get_scratch_buffer(&g_scratch_C, &g_scratch_C_bytes, m * n * sizeof(float));
    if (!d_A || !d_C) return -1;

    if (gpu_cuda_upload(g_ctx, d_A, A, m * k * sizeof(float)) != 0) return -1;

    unsigned int m32 = (unsigned int)m, k32 = (unsigned int)k, n32 = (unsigned int)n;
    void *args[] = { &d_A, &d_B, &d_C, &m32, &k32, &n32 };
    const unsigned int BLOCK = 16;
    unsigned int grid_x = ((unsigned int)n + BLOCK - 1) / BLOCK;
    unsigned int grid_y = ((unsigned int)m + BLOCK - 1) / BLOCK;
    /* Launched without waiting: the download below runs on the same default
     * stream, so it cannot start until this kernel has finished, and it is
     * itself blocking. Waiting here as well cost a full extra device
     * round-trip per matmul for nothing. A fault inside the kernel still
     * fails the call - it surfaces as an error from that download. */
    if (gpu_cuda_launch_2d_async_with(g_ctx, g_k_forward, grid_x, grid_y,
                                      BLOCK, BLOCK, args) != 0) return -1;

    if (gpu_cuda_download(g_ctx, C, d_C, m * n * sizeof(float)) != 0) return -1;

    return 0;
}

/* --------------------------------------------------------- backward pass ---
 *
 * Both entries mirror core/matmul.c's CPU signatures exactly, including the
 * accumulate-into-destination contract, so transformer/training code can
 * substitute one for the other without knowing which ran.
 *
 * They cost more per call than the forward path, structurally: the
 * destination is read-modify-write, so it is uploaded as well as downloaded.
 * A forward matmul moves (A in, C out); a backward one moves (two inputs in,
 * destination in, destination out). Whether that pays is a question about
 * shape, and the answer is measured rather than assumed - see docs/gpu.md and
 * the dispatch thresholds in core/training.c.
 */

int gpu_matmul_backward_input(const float *dC, const float *B, float *dA,
                              size_t m, size_t k, size_t n) {
    if (!ensure_ready()) return -1;

    /* B is a weight matrix here exactly as in the forward pass - the same
     * host pointer, so it hits the same cache entry the forward call filled. */
    uint64_t d_B = get_cached_weight_buffer(B, k * n * sizeof(float));
    if (!d_B) return -1;

    uint64_t d_dC = get_scratch_buffer(&g_scratch_A, &g_scratch_A_bytes, m * n * sizeof(float));
    uint64_t d_dA = get_scratch_buffer(&g_scratch_D, &g_scratch_D_bytes, m * k * sizeof(float));
    if (!d_dC || !d_dA) return -1;

    if (gpu_cuda_upload(g_ctx, d_dC, dC, m * n * sizeof(float)) != 0) return -1;
    /* The destination carries a partial gradient from earlier call sites in
     * this minibatch; the kernel adds to it, so it has to go up first. */
    if (gpu_cuda_upload(g_ctx, d_dA, dA, m * k * sizeof(float)) != 0) return -1;

    unsigned int m32 = (unsigned int)m, k32 = (unsigned int)k, n32 = (unsigned int)n;
    void *args[] = { &d_dC, &d_B, &d_dA, &m32, &k32, &n32 };
    const unsigned int BLOCK = 16;
    /* One thread per element of dA (m x k): x spans k, y spans m. */
    unsigned int grid_x = ((unsigned int)k + BLOCK - 1) / BLOCK;
    unsigned int grid_y = ((unsigned int)m + BLOCK - 1) / BLOCK;
    if (gpu_cuda_launch_2d_async_with(g_ctx, g_k_backward_input, grid_x, grid_y,
                                      BLOCK, BLOCK, args) != 0) return -1;

    if (gpu_cuda_download(g_ctx, dA, d_dA, m * k * sizeof(float)) != 0) return -1;

    return 0;
}

int gpu_matmul_backward_weight(const float *A, const float *dC, float *dB,
                               size_t m, size_t k, size_t n) {
    if (!ensure_ready()) return -1;

    /* Neither input is a weight here: A is an activation and dC a gradient,
     * both of which change every call. Both take scratch buffers. */
    uint64_t d_A = get_scratch_buffer(&g_scratch_A, &g_scratch_A_bytes, m * k * sizeof(float));
    uint64_t d_dC = get_scratch_buffer(&g_scratch_C, &g_scratch_C_bytes, m * n * sizeof(float));
    uint64_t d_dB = get_scratch_buffer(&g_scratch_D, &g_scratch_D_bytes, k * n * sizeof(float));
    if (!d_A || !d_dC || !d_dB) return -1;

    if (gpu_cuda_upload(g_ctx, d_A, A, m * k * sizeof(float)) != 0) return -1;
    if (gpu_cuda_upload(g_ctx, d_dC, dC, m * n * sizeof(float)) != 0) return -1;
    if (gpu_cuda_upload(g_ctx, d_dB, dB, k * n * sizeof(float)) != 0) return -1;

    unsigned int m32 = (unsigned int)m, k32 = (unsigned int)k, n32 = (unsigned int)n;
    void *args[] = { &d_A, &d_dC, &d_dB, &m32, &k32, &n32 };
    const unsigned int BLOCK = 16;
    /* One thread per element of dB (k x n): x spans n, y spans k. */
    unsigned int grid_x = ((unsigned int)n + BLOCK - 1) / BLOCK;
    unsigned int grid_y = ((unsigned int)k + BLOCK - 1) / BLOCK;
    if (gpu_cuda_launch_2d_async_with(g_ctx, g_k_backward_weight, grid_x, grid_y,
                                      BLOCK, BLOCK, args) != 0) return -1;

    if (gpu_cuda_download(g_ctx, dB, d_dB, k * n * sizeof(float)) != 0) return -1;

    return 0;
}

void gpu_matmul_shutdown(void) {
    if (g_ctx) gpu_cuda_shutdown(g_ctx); /* destroying the context reclaims every device allocation under it */
    g_ctx = NULL;
    g_init_attempted = 0;
    g_kernel_loaded = 0;
    g_environment_checked = 0;
    g_environment_requests_naive = 0;
    g_config_status = GPU_MATMUL_CONFIG_OK;
    g_invalid_environment_value[0] = '\0';

    memset(g_weight_cache, 0, sizeof(g_weight_cache));
    g_weight_cache_count = 0;
    g_weight_generation = 1;

    g_k_forward = NULL;
    g_k_forward_naive = NULL;
    g_k_forward_tiled = NULL;
    g_k_backward_input = NULL;
    g_k_backward_weight = NULL;

    g_scratch_A = 0; g_scratch_A_bytes = 0;
    g_scratch_C = 0; g_scratch_C_bytes = 0;
    g_scratch_D = 0; g_scratch_D_bytes = 0;
}
