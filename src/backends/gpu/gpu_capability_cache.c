/*
 * Persists a gpu_capability_t (built entirely from the public NVML/CUDA
 * Driver API/OpenCL probes in gpu_backend_*.c) to a simple key=value text
 * file, and reads it back. Same on-disk style as config.c, deliberately -
 * no JSON library, no new dependency, just a format this project already
 * uses.
 */

#include "include/gpu_capability_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int gpu_capability_cache_filename(const gpu_capability_t *cap, const char *dir,
                                  char *out_path, size_t out_path_size) {
    if (!cap->has_nvml || cap->nvml.gpu_name[0] == '\0') return -1;

    char key[256];
    size_t j = 0;
    for (size_t i = 0; cap->nvml.gpu_name[i] != '\0' && j < sizeof(key) - 2; i++) {
        char c = cap->nvml.gpu_name[i];
        key[j++] = (c == ' ' || c == '/' || c == '\\') ? '_' : c;
    }
    key[j++] = '_';
    for (size_t i = 0; cap->nvml.driver_version[i] != '\0' && j < sizeof(key) - 1; i++) {
        char c = cap->nvml.driver_version[i];
        key[j++] = (c == ' ' || c == '/' || c == '\\') ? '_' : c;
    }
    key[j] = '\0';

    snprintf(out_path, out_path_size, "%s/%s.cache", dir, key);
    return 0;
}

int gpu_capability_cache_save(const gpu_capability_t *cap, const char *dir) {
    char path[1024];
    if (gpu_capability_cache_filename(cap, dir, path, sizeof(path)) != 0) return -1;

    mkdir(dir, 0755); /* ignore EEXIST - that's the common case */

    FILE *f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "# GPU capability profile - probed by gpu_probe.out\n");
    fprintf(f, "# Every field below comes from a publicly documented NVML/CUDA Driver\n");
    fprintf(f, "# API/OpenCL call (see gpu_backend_nvml.c/gpu_backend_cuda.c/gpu_backend_opencl.c) -\n");
    fprintf(f, "# nothing here is derived from an undocumented interface.\n\n");

    fprintf(f, "probed_unix_time = %ld\n\n", (long)cap->probed_at);

    fprintf(f, "has_nvml = %d\n", cap->has_nvml);
    if (cap->has_nvml) {
        fprintf(f, "gpu_name = %s\n", cap->nvml.gpu_name);
        fprintf(f, "driver_version = %s\n", cap->nvml.driver_version);
        fprintf(f, "device_count = %d\n", cap->nvml.device_count);
        fprintf(f, "vram_total_mb = %llu\n", cap->nvml.vram_total_mb);
        fprintf(f, "compute_capability_major = %d\n", cap->nvml.cc_major);
        fprintf(f, "compute_capability_minor = %d\n", cap->nvml.cc_minor);
    }
    fprintf(f, "\n");

    fprintf(f, "has_cuda = %d\n", cap->has_cuda);
    if (cap->has_cuda) {
        fprintf(f, "cuda_driver_functional = %d\n", cap->cuda.driver_functional);
        fprintf(f, "cuda_driver_version = %d\n", cap->cuda.driver_version);
        fprintf(f, "cuda_device_count = %d\n", cap->cuda.device_count);
        fprintf(f, "nvcc_available = %d\n", cap->cuda.nvcc_available);
        fprintf(f, "nvrtc_available = %d\n", cap->cuda.nvrtc_available);
    }
    fprintf(f, "\n");

    fprintf(f, "has_opencl = %d\n", cap->has_opencl);
    if (cap->has_opencl) {
        fprintf(f, "opencl_loader_functional = %d\n", cap->opencl.loader_functional);
        fprintf(f, "opencl_platform_count = %d\n", cap->opencl.platform_count);
        fprintf(f, "opencl_platform_names = %s\n", cap->opencl.platform_names);
        fprintf(f, "opencl_device_count = %d\n", cap->opencl.total_device_count);
    }

    fclose(f);
    return 0;
}

int gpu_capability_cache_load(const char *path, gpu_capability_t *out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    memset(out, 0, sizeof(*out));

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        long t;
        if (sscanf(line, "probed_unix_time = %ld", &t) == 1) { out->probed_at = (time_t)t; continue; }

        if (sscanf(line, "has_nvml = %d", &out->has_nvml) == 1) continue;
        if (sscanf(line, "gpu_name = %127[^\n]", out->nvml.gpu_name) == 1) continue;
        if (sscanf(line, "driver_version = %63[^\n]", out->nvml.driver_version) == 1) continue;
        if (sscanf(line, "device_count = %d", &out->nvml.device_count) == 1) continue;
        if (sscanf(line, "vram_total_mb = %llu", &out->nvml.vram_total_mb) == 1) continue;
        if (sscanf(line, "compute_capability_major = %d", &out->nvml.cc_major) == 1) continue;
        if (sscanf(line, "compute_capability_minor = %d", &out->nvml.cc_minor) == 1) continue;

        if (sscanf(line, "has_cuda = %d", &out->has_cuda) == 1) continue;
        if (sscanf(line, "cuda_driver_functional = %d", &out->cuda.driver_functional) == 1) continue;
        if (sscanf(line, "cuda_driver_version = %d", &out->cuda.driver_version) == 1) continue;
        if (sscanf(line, "cuda_device_count = %d", &out->cuda.device_count) == 1) continue;
        if (sscanf(line, "nvcc_available = %d", &out->cuda.nvcc_available) == 1) continue;
        if (sscanf(line, "nvrtc_available = %d", &out->cuda.nvrtc_available) == 1) continue;

        if (sscanf(line, "has_opencl = %d", &out->has_opencl) == 1) continue;
        if (sscanf(line, "opencl_loader_functional = %d", &out->opencl.loader_functional) == 1) continue;
        if (sscanf(line, "opencl_platform_count = %d", &out->opencl.platform_count) == 1) continue;
        if (sscanf(line, "opencl_platform_names = %255[^\n]", out->opencl.platform_names) == 1) continue;
        if (sscanf(line, "opencl_device_count = %d", &out->opencl.total_device_count) == 1) continue;
    }

    fclose(f);
    return 0;
}

void gpu_capability_print(const gpu_capability_t *cap) {
    printf("Probed at: %s", ctime(&cap->probed_at));

    if (cap->has_nvml) {
        printf("GPU: %s\n", cap->nvml.gpu_name);
        printf("Driver version: %s\n", cap->nvml.driver_version);
        printf("VRAM: %llu MB\n", cap->nvml.vram_total_mb);
        if (cap->nvml.cc_major >= 0) {
            printf("Compute capability: %d.%d\n", cap->nvml.cc_major, cap->nvml.cc_minor);
        }
    } else {
        printf("NVML: not available\n");
    }

    if (cap->has_cuda) {
        printf("CUDA driver: %s (version %d.%d)\n",
               cap->cuda.driver_functional ? "functional" : "not functional",
               cap->cuda.driver_version / 1000, (cap->cuda.driver_version % 1000) / 10);
        printf("  nvcc: %s, NVRTC: %s (neither needed for this project's hand-written PTX path)\n",
               cap->cuda.nvcc_available ? "available" : "not available",
               cap->cuda.nvrtc_available ? "available" : "not available");
    } else {
        printf("CUDA: not available\n");
    }

    if (cap->has_opencl) {
        printf("OpenCL loader: %s, %d platform(s)%s%s\n",
               cap->opencl.loader_functional ? "functional" : "not functional",
               cap->opencl.platform_count,
               cap->opencl.platform_count > 0 ? ": " : "",
               cap->opencl.platform_count > 0 ? cap->opencl.platform_names : "");
    } else {
        printf("OpenCL: not available\n");
    }
}
