#ifndef GPU_THEORETICAL_PERF_H
#define GPU_THEORETICAL_PERF_H

/* Combines this project's own live measurements (SM count and clock rate,
 * both from the cuDeviceGetAttribute sweep) with gpu_arch_specs.h's public
 * cores-per-SM table into a theoretical peak FP32 FLOPS estimate -
 * exactly the number that matters before deciding how a matmul/attention
 * kernel should be tiled: how much compute is even available to target.
 * "Theoretical" because it assumes every core does one FMA (2 FLOPs) per
 * cycle with no stalls - gpu_microbench.h's empirical throughput test is
 * what tells you how close a real kernel gets to this ceiling. */
typedef struct {
    int available; /* 0 if this compute capability isn't in gpu_arch_specs.h's table */
    const char *arch_name;
    int cores_per_sm;
    int total_cores;
    double peak_flops_fp32; /* FMA counted as 2 FLOPs */
} gpu_theoretical_perf_t;

void gpu_theoretical_perf_compute(int sm_count, int clock_rate_khz, int cc_major, int cc_minor,
                                  gpu_theoretical_perf_t *out);

void gpu_theoretical_perf_print(const gpu_theoretical_perf_t *p);

#endif // GPU_THEORETICAL_PERF_H
