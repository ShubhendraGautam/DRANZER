#include "backends/gpu/gpu_arch_specs.h"
#include <stddef.h>

typedef struct {
    int major, minor;
    int cores_per_sm;
    const char *name;
} gpu_arch_entry_t;

/* Source: NVIDIA's publicly published CUDA C Programming Guide, Appendix
 * "Compute Capabilities" (cores-per-SM figures are for FP32 CUDA cores). */
static const gpu_arch_entry_t GPU_ARCH_TABLE[] = {
    {3, 0, 192, "Kepler"},
    {3, 5, 192, "Kepler"},
    {3, 7, 192, "Kepler"},
    {5, 0, 128, "Maxwell"},
    {5, 2, 128, "Maxwell"},
    {5, 3, 128, "Maxwell"},
    {6, 0, 64,  "Pascal (GP100)"},
    {6, 1, 128, "Pascal"},
    {6, 2, 128, "Pascal"},
    {7, 0, 64,  "Volta"},
    {7, 2, 64,  "Volta"},
    {7, 5, 64,  "Turing"},
    {8, 0, 64,  "Ampere (A100)"},
    {8, 6, 128, "Ampere"},
    {8, 7, 128, "Ampere (Orin)"},
    {8, 9, 128, "Ada Lovelace"},
    {9, 0, 128, "Hopper"},
};
static const int GPU_ARCH_TABLE_COUNT = sizeof(GPU_ARCH_TABLE) / sizeof(GPU_ARCH_TABLE[0]);

static const gpu_arch_entry_t *lookup(int major, int minor) {
    for (int i = 0; i < GPU_ARCH_TABLE_COUNT; i++) {
        if (GPU_ARCH_TABLE[i].major == major && GPU_ARCH_TABLE[i].minor == minor) return &GPU_ARCH_TABLE[i];
    }
    return NULL;
}

int gpu_arch_cores_per_sm(int cc_major, int cc_minor) {
    const gpu_arch_entry_t *e = lookup(cc_major, cc_minor);
    return e ? e->cores_per_sm : 0;
}

const char *gpu_arch_name(int cc_major, int cc_minor) {
    const gpu_arch_entry_t *e = lookup(cc_major, cc_minor);
    return e ? e->name : NULL;
}
