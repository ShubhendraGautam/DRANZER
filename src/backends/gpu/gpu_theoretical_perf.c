#include "backends/gpu/gpu_theoretical_perf.h"
#include "backends/gpu/gpu_arch_specs.h"
#include <stdio.h>
#include <string.h>

void gpu_theoretical_perf_compute(int sm_count, int clock_rate_khz, int cc_major, int cc_minor,
                                  gpu_theoretical_perf_t *out) {
    memset(out, 0, sizeof(*out));

    int cores_per_sm = gpu_arch_cores_per_sm(cc_major, cc_minor);
    if (cores_per_sm == 0 || sm_count <= 0 || clock_rate_khz <= 0) return;

    out->available = 1;
    out->arch_name = gpu_arch_name(cc_major, cc_minor);
    out->cores_per_sm = cores_per_sm;
    out->total_cores = sm_count * cores_per_sm;

    double clock_hz = (double)clock_rate_khz * 1000.0;
    out->peak_flops_fp32 = (double)out->total_cores * clock_hz * 2.0; /* FMA = 2 FLOPs */
}

void gpu_theoretical_perf_print(const gpu_theoretical_perf_t *p) {
    if (!p->available) {
        printf("    Not available - this compute capability isn't in gpu_arch_specs.c's table.\n");
        return;
    }
    printf("    Architecture: %s (%d cores/SM, %d total FP32 cores)\n",
           p->arch_name ? p->arch_name : "unknown", p->cores_per_sm, p->total_cores);
    printf("    Theoretical peak FP32: %.2f TFLOPS (SM_count * cores/SM * clock_rate * 2 for FMA)\n",
           p->peak_flops_fp32 / 1e12);
}
