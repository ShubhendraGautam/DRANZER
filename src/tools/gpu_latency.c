/*
 * GPU per-call cost probe: where does a gpu_matmul() call actually go?
 *
 * Device buffers are already persistent (see gpu_matmul.c's weight cache and
 * scratch buffers), so what remains per call is one activation upload, the
 * kernel launch, and one result download. This tool times those three
 * primitives on their own, then times whole gpu_matmul() calls at the shapes
 * the model issues, so the fixed per-call overhead can be told apart from
 * the arithmetic. It is the tool behind the per-call numbers in docs/gpu.md.
 *
 * Distinct from gpu_probe.out (which asks "is there a usable GPU and what
 * are its specs") and from gpu_microbench.c inside it (which measures peak
 * bandwidth and FMA throughput). This one measures latency - the thing that
 * decides whether GPU offload is worth it at small model sizes.
 *
 * Build:  make gpu-latency
 * Run:    ./gpu_latency.out
 *
 * Exits 0 and explains itself when no CUDA device is usable.
 */

#include "backends/gpu/gpu_cuda.h"
#include "backends/gpu/gpu_matmul.h"
#include "tools/bench_support.h"
#include <stdio.h>
#include <stdlib.h>

/* A kernel that does nothing, so a launch can be timed without any of the
 * work a real kernel would do. One unused parameter keeps the argument
 * marshalling identical to a real launch. */
static const char *NOOP_PTX =
".version 7.0\n.target sm_75\n.address_size 64\n\n"
".visible .entry noop(.param .u64 p)\n{\n    ret;\n}\n";

#define TIMING_ROUNDS 5
#define CALLS_PER_ROUND 200

typedef struct {
    gpu_cuda_ctx_t *ctx;
    uint64_t dptr;
    void *host;
    size_t bytes;
} io_t;

/* Fastest round wins: contention and clock ramping can only ever make a
 * round slower, so the minimum is the closest estimate of the true cost -
 * the same rule the matmul sweep uses (docs/matmul.md). */
static double best_us(double (*fn)(void *, int), void *arg, int calls) {
    double best = -1.0;
    for (int round = 0; round < TIMING_ROUNDS; round++) {
        double seconds = fn(arg, calls) / calls;
        if (best < 0.0 || seconds < best) best = seconds;
    }
    return best * 1e6;
}

static double time_upload(void *arg, int calls) {
    io_t *io = arg;
    double t0 = bench_now_sec();
    for (int i = 0; i < calls; i++) {
        gpu_cuda_upload(io->ctx, io->dptr, io->host, io->bytes);
    }
    return bench_now_sec() - t0;
}

static double time_download(void *arg, int calls) {
    io_t *io = arg;
    double t0 = bench_now_sec();
    for (int i = 0; i < calls; i++) {
        gpu_cuda_download(io->ctx, io->host, io->dptr, io->bytes);
    }
    return bench_now_sec() - t0;
}

static double time_launch(void *arg, int calls) {
    io_t *io = arg;
    void *args[] = { &io->dptr };
    double t0 = bench_now_sec();
    for (int i = 0; i < calls; i++) {
        gpu_cuda_launch_2d(io->ctx, 1, 1, 1, 1, args);
    }
    return bench_now_sec() - t0;
}

typedef struct { float *a, *b, *c; size_t m, k, n; } matmul_t;

static double time_matmul(void *arg, int calls) {
    matmul_t *mm = arg;
    double t0 = bench_now_sec();
    for (int i = 0; i < calls; i++) {
        gpu_matmul(mm->a, mm->b, mm->c, mm->m, mm->k, mm->n);
    }
    return bench_now_sec() - t0;
}

/* Transfers run in this tool's own context, which is destroyed before
 * gpu_matmul.c lazily creates its own: a kernel handle is only valid inside
 * the context that loaded it, so the two must never overlap. */
static int measure_primitives(void) {
    gpu_cuda_ctx_t *ctx = gpu_cuda_init();
    if (!ctx) {
        printf("No usable CUDA context - skipping transfer measurements.\n");
        return -1;
    }
    if (gpu_cuda_load_kernel(ctx, NOOP_PTX, "noop") != 0) {
        printf("Could not load the empty probe kernel - skipping.\n");
        gpu_cuda_shutdown(ctx);
        return -1;
    }

    printf("Per-call driver costs (fastest of %d rounds x %d calls)\n",
           TIMING_ROUNDS, CALLS_PER_ROUND);
    printf("  %-12s %12s %12s %12s\n", "transfer", "upload", "download", "upload GB/s");

    const size_t sizes[] = { 1024, 16384, 262144, 1048576 };
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        io_t io = { ctx, gpu_cuda_alloc(ctx, sizes[i]), malloc(sizes[i]), sizes[i] };
        if (!io.dptr || !io.host) {
            printf("  %-12zu allocation failed\n", sizes[i]);
            gpu_cuda_free(ctx, io.dptr);
            free(io.host);
            continue;
        }
        double up = best_us(time_upload, &io, CALLS_PER_ROUND);
        double down = best_us(time_download, &io, CALLS_PER_ROUND);
        char label[32];
        snprintf(label, sizeof(label), "%zu B", sizes[i]);
        printf("  %-12s %9.1f us %9.1f us %12.2f\n", label, up, down,
               (double)sizes[i] / (up * 1e-6) / 1e9);
        gpu_cuda_free(ctx, io.dptr);
        free(io.host);
    }

    io_t launch = { ctx, 0, NULL, 0 };
    printf("  %-12s %9.1f us\n", "empty launch",
           best_us(time_launch, &launch, CALLS_PER_ROUND));
    printf("\nSmall transfers are latency, not bandwidth: if 1 KB costs about what\n"
           "16 KB costs, every call pays a fixed toll no matter how little data moves.\n\n");

    gpu_cuda_shutdown(ctx);
    return 0;
}

static void measure_shape(const char *label, size_t m, size_t k, size_t n) {
    matmul_t mm = { malloc(m * k * sizeof(float)), malloc(k * n * sizeof(float)),
                    malloc(m * n * sizeof(float)), m, k, n };
    if (!mm.a || !mm.b || !mm.c) {
        printf("  %-30s allocation failed\n", label);
        free(mm.a); free(mm.b); free(mm.c);
        return;
    }
    for (size_t i = 0; i < m * k; i++) mm.a[i] = (float)(i % 7) * 0.1f;
    for (size_t i = 0; i < k * n; i++) mm.b[i] = (float)(i % 5) * 0.1f;

    /* Warm up: also populates the weight cache, so the timed calls measure
     * steady state rather than the one-off upload of B. */
    if (gpu_matmul(mm.a, mm.b, mm.c, m, k, n) != 0) {
        printf("  %-30s call failed\n", label);
        free(mm.a); free(mm.b); free(mm.c);
        return;
    }

    double us = best_us(time_matmul, &mm, 50);
    double operations = 2.0 * (double)m * (double)k * (double)n;
    char shape[32];
    snprintf(shape, sizeof(shape), "%zux%zux%zu", m, k, n);
    printf("  %-30s %-18s %9.1f us %9.2f GFLOP/s\n", label, shape, us,
           operations / (us * 1e-6) / 1e9);
    free(mm.a); free(mm.b); free(mm.c);
}

int main(void) {
    bench_metadata_t metadata;
    bench_collect_metadata(&metadata);
    bench_print_metadata(&metadata);

    if (!gpu_matmul_available()) {
        printf("No usable CUDA GPU detected - nothing to measure.\n"
               "Run ./gpu_probe.out to see which backends were found and why.\n");
        return 0;
    }

    (void)measure_primitives();

    printf("Whole gpu_matmul() calls, weights already resident\n");
    printf("  %-30s %-18s %12s %14s\n", "call site", "m x k x n", "per call", "throughput");
    measure_shape("decode: attn projection", 1, 64, 64);
    measure_shape("decode: output head", 1, 64, 1000);
    measure_shape("decode: attn projection", 1, 256, 256);
    measure_shape("decode: output head", 1, 256, 4000);
    measure_shape("prefill: attn projection", 64, 64, 64);
    measure_shape("prefill: attn projection", 128, 256, 256);
    measure_shape("prefill: ffn up", 128, 256, 1024);
    measure_shape("training: ffn down", 128, 1024, 256);

    printf("\nCompare these against the CPU kernel on the same shapes:\n"
           "  ./bench.out --matmul-only --repeats 5\n");

    gpu_matmul_shutdown();
    return 0;
}
