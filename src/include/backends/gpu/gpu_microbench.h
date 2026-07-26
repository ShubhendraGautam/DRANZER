#ifndef GPU_MICROBENCH_H
#define GPU_MICROBENCH_H

/* Empirically measured (not theoretical) GPU throughput, via two small
 * hand-written PTX kernels loaded through gpu_cuda.h:
 *   - a copy kernel (C[i] = A[i]) sized to move a large buffer, to measure
 *     real achieved memory bandwidth
 *   - a compute-bound kernel (a long dependent FMA chain per thread, with
 *     runtime-supplied - not compile-time-constant - operands so the
 *     driver's JIT can't optimize the loop away) to measure real achieved
 *     FP32 throughput
 * This is what tells you how close a real kernel gets to
 * gpu_theoretical_perf.h's ceiling, and is the actual decision-relevant
 * number for planning whether/how to offload a given operation: transfer
 * overhead and real achieved throughput matter far more than a
 * theoretical peak. */
typedef struct {
    int available;
    double bandwidth_gbps;   /* measured, from the copy kernel */
    double fma_gflops;       /* measured, from the compute kernel */
} gpu_microbench_result_t;

void gpu_microbench_run(gpu_microbench_result_t *out);
void gpu_microbench_print(const gpu_microbench_result_t *r);

#endif // GPU_MICROBENCH_H
