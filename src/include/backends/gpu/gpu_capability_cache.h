#ifndef GPU_CAPABILITY_CACHE_H
#define GPU_CAPABILITY_CACHE_H

#include "backends/gpu/gpu_probe.h"
#include <time.h>

/* A GPU's full capability profile, gathered from the public NVML/CUDA
 * Driver API/OpenCL probes in gpu_backend_*.c (nothing undocumented - see
 * those files), persisted to disk so it can be reused across runs and
 * shared between researchers on similar hardware/driver combinations
 * instead of every one of them re-probing from scratch. */
typedef struct {
    gpu_nvml_facts_t nvml;
    int has_nvml;
    gpu_cuda_facts_t cuda;
    int has_cuda;
    gpu_opencl_facts_t opencl;
    int has_opencl;
    time_t probed_at;

    /* Derived/measured performance data (gpu_theoretical_perf.h,
     * gpu_microbench.h) - the microbenchmark is the only part of a probe
     * that takes real wall-clock time, so caching it is what lets a
     * researcher look up "what does this GPU actually achieve" without
     * re-running it. */
    int has_perf;
    int sm_count;
    int clock_rate_khz;
    double theoretical_peak_flops_fp32;
    double measured_bandwidth_gbps;
    double measured_fma_gflops;
} gpu_capability_t;

/* Builds the path this capability record would be saved to/loaded from:
 * dir/<sanitized gpu_name>_<sanitized driver_version>.cache. Returns 0 on
 * success, -1 if there isn't enough information (no NVML GPU name) to
 * build a meaningful key. */
int gpu_capability_cache_filename(const gpu_capability_t *cap, const char *dir,
                                  char *out_path, size_t out_path_size);

/* Writes cap to dir/<key>.cache as simple key=value text (consistent with
 * this project's config.c format). Creates dir if it doesn't exist.
 * Returns 0 on success. */
int gpu_capability_cache_save(const gpu_capability_t *cap, const char *dir);

/* Reads a cache file (exact path) into *out. This is the reuse path: a
 * researcher with a matching GPU model + driver version can load a cache
 * file someone else already produced instead of re-probing. Returns 0 on
 * success, -1 if the file doesn't exist or can't be parsed. */
int gpu_capability_cache_load(const char *path, gpu_capability_t *out);

/* Human-readable dump of a capability record (same shape whether it was
 * just probed or loaded from a cache file). */
void gpu_capability_print(const gpu_capability_t *cap);

#endif // GPU_CAPABILITY_CACHE_H
