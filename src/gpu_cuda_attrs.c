#include "include/gpu_cuda_attrs.h"
#include <stdio.h>

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
};
const int GPU_CUDA_ATTR_TABLE_COUNT = sizeof(GPU_CUDA_ATTR_TABLE) / sizeof(GPU_CUDA_ATTR_TABLE[0]);

void gpu_cuda_attrs_print_all(gpu_cuda_ctx_t *ctx) {
    int resolved = 0;
    for (int i = 0; i < GPU_CUDA_ATTR_TABLE_COUNT; i++) {
        int value;
        if (gpu_cuda_device_attribute(ctx, GPU_CUDA_ATTR_TABLE[i].code, &value) == 0) {
            printf("    %-30s = %d\n", GPU_CUDA_ATTR_TABLE[i].name, value);
            resolved++;
        }
    }
    printf("    (%d/%d attributes resolved)\n", resolved, GPU_CUDA_ATTR_TABLE_COUNT);
}
