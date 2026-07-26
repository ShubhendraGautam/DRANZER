#include "backends/gpu/gpu_cuda_attrs.h"
#include <stdio.h>

/* Verified names only - see the header. Names here have all been
 * cross-checked against an independent source before being trusted:
 * WARP_SIZE/MAX_THREADS_PER_BLOCK against universal NVIDIA constants (32
 * and 1024 on any modern GPU), COMPUTE_CAPABILITY_MAJOR/MINOR against
 * NVML's independently-reported value - both matched exactly. Expanding
 * this table further should be done the same way: verify against
 * NVIDIA's published cuda.h/Driver API reference or another independent
 * source, not from memory alone. */
const gpu_cuda_attr_entry_t GPU_CUDA_ATTR_TABLE[] = {
    {"MAX_THREADS_PER_BLOCK", 1},
    {"MAX_BLOCK_DIM_X", 2},
    {"MAX_BLOCK_DIM_Y", 3},
    {"MAX_BLOCK_DIM_Z", 4},
    {"MAX_GRID_DIM_X", 5},
    {"MAX_GRID_DIM_Y", 6},
    {"MAX_GRID_DIM_Z", 7},
    {"MAX_SHARED_MEMORY_PER_BLOCK", 8},
    {"TOTAL_CONSTANT_MEMORY", 9},
    {"WARP_SIZE", 10},
    {"MAX_REGISTERS_PER_BLOCK", 12},
    {"CLOCK_RATE_KHZ", 13},
    {"MULTIPROCESSOR_COUNT", 16},
    {"CAN_MAP_HOST_MEMORY", 19},
    {"COMPUTE_MODE", 20},
    {"COMPUTE_CAPABILITY_MAJOR", 75},
    {"COMPUTE_CAPABILITY_MINOR", 76},
    /* Cross-checked against gpu_nvml_telemetry.c's max_clock_mem_mhz,
     * which itself matches nvidia-smi exactly (3501 MHz on the machine
     * this was verified on) - code 36 * 1000 == that value precisely. */
    {"MEMORY_CLOCK_RATE_KHZ", 36},
};
const int GPU_CUDA_ATTR_TABLE_COUNT = sizeof(GPU_CUDA_ATTR_TABLE) / sizeof(GPU_CUDA_ATTR_TABLE[0]);

static const char *lookup_name(int code) {
    for (int i = 0; i < GPU_CUDA_ATTR_TABLE_COUNT; i++) {
        if (GPU_CUDA_ATTR_TABLE[i].code == code) return GPU_CUDA_ATTR_TABLE[i].name;
    }
    return NULL;
}

int gpu_cuda_attrs_sweep(gpu_cuda_ctx_t *ctx, gpu_cuda_attr_result_t *results) {
    int count = 0;
    for (int code = 1; code <= GPU_CUDA_ATTR_SWEEP_MAX; code++) {
        int value;
        if (gpu_cuda_device_attribute(ctx, code, &value) != 0) continue; /* not supported - expected for many codes */
        results[count].code = code;
        results[count].value = value;
        results[count].name = lookup_name(code);
        count++;
    }
    return count;
}

void gpu_cuda_attrs_print_all(gpu_cuda_ctx_t *ctx) {
    gpu_cuda_attr_result_t results[GPU_CUDA_ATTR_SWEEP_MAX];
    int count = gpu_cuda_attrs_sweep(ctx, results);

    int named = 0;
    for (int i = 0; i < count; i++) {
        if (results[i].name) {
            printf("    [%3d] %-30s = %d\n", results[i].code, results[i].name, results[i].value);
            named++;
        } else {
            printf("    [%3d] (unlabeled)                     = %d\n", results[i].code, results[i].value);
        }
    }
    printf("    (%d/%d attribute codes 1..%d resolved on this device, %d labeled with a verified name)\n",
           count, GPU_CUDA_ATTR_SWEEP_MAX, GPU_CUDA_ATTR_SWEEP_MAX, named);
}
