/* CUDA probe. Deliberately checks three DIFFERENT things that are easy to
 * conflate into a single "is CUDA installed?" question:
 *
 *   1. The driver API (libcuda.so) - ships with the NVIDIA driver itself.
 *      cuInit() succeeding here means the GPU is visible to CUDA, but says
 *      NOTHING about whether you can get a kernel onto it.
 *   2. nvcc (the CUDA toolkit's compiler) - lets you compile .cu files at
 *      BUILD time.
 *   3. NVRTC (libnvrtc.so, "NVIDIA Runtime Compilation") - lets you compile
 *      CUDA C source at RUN time instead, the same model OpenCL uses. This
 *      is what would let a CUDA backend stay "pure C" (no nvcc needed to
 *      build this project) the same way the OpenCL backend does.
 *
 * A machine can have (1) without either (2) or (3) - that's this exact
 * sandbox's state: cuInit() succeeds, but there is currently no way to
 * compile a kernel at all. Reporting only "CUDA: found" in that case would
 * be actively misleading. */

#include "include/gpu_probe.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int CUresult;
typedef CUresult (*fn_cuInit)(unsigned int);
typedef CUresult (*fn_cuDriverGetVersion)(int *);
typedef CUresult (*fn_cuDeviceGetCount)(int *);

static int nvcc_in_path(char *out_path, size_t out_path_size) {
    const char *path_env = getenv("PATH");
    if (!path_env) return 0;

    char path_copy[4096];
    strncpy(path_copy, path_env, sizeof(path_copy) - 1);
    path_copy[sizeof(path_copy) - 1] = '\0';

    char *saveptr = NULL;
    for (char *dir = strtok_r(path_copy, ":", &saveptr); dir; dir = strtok_r(NULL, ":", &saveptr)) {
        char candidate[1024];
        snprintf(candidate, sizeof(candidate), "%s/nvcc", dir);
        if (access(candidate, X_OK) == 0) {
            strncpy(out_path, candidate, out_path_size - 1);
            out_path[out_path_size - 1] = '\0';
            return 1;
        }
    }
    return 0;
}

void gpu_backend_cuda_probe(gpu_backend_report_t *out, gpu_cuda_facts_t *facts) {
    out->status = GPU_BACKEND_UNAVAILABLE;
    out->message[0] = '\0';
    size_t len = 0;
    if (facts) memset(facts, 0, sizeof(*facts));

    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libcuda.so", RTLD_NOW);
    if (!lib) {
        snprintf(out->message, sizeof(out->message),
                 "libcuda.so not found - no CUDA driver present.");
        return;
    }

    fn_cuInit cuInit = (fn_cuInit)dlsym(lib, "cuInit");
    fn_cuDriverGetVersion cuDriverGetVersion = (fn_cuDriverGetVersion)dlsym(lib, "cuDriverGetVersion");
    fn_cuDeviceGetCount cuDeviceGetCount = (fn_cuDeviceGetCount)dlsym(lib, "cuDeviceGetCount");

    CUresult rc = cuInit ? cuInit(0) : -1;
    if (!cuInit || rc != 0) {
        snprintf(out->message, sizeof(out->message),
                 "libcuda.so loaded, but cuInit() failed (code %d) - driver library present but not functional.", rc);
        dlclose(lib);
        return;
    }

    int version = 0, device_count = 0;
    if (cuDriverGetVersion) cuDriverGetVersion(&version);
    if (cuDeviceGetCount) cuDeviceGetCount(&device_count);

    len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                             "CUDA driver API: functional (version %d.%d, %d device(s) visible)\n",
                             version / 1000, (version % 1000) / 10, device_count);
    if (facts) {
        facts->driver_functional = 1;
        facts->driver_version = version;
        facts->device_count = device_count;
    }
    dlclose(lib);

    /* The driver API alone (cuModuleLoadData + hand-written PTX text, no
     * nvcc/NVRTC involved) is a complete, working GPU compute path - see
     * gpu_matmul.c. nvcc/NVRTC are a separate, optional convenience for
     * authoring kernels in CUDA C instead of PTX directly; their absence
     * does not mean "no GPU compute available" for this project. */
    out->status = GPU_BACKEND_READY;

    char nvcc_path[1024];
    int has_nvcc = nvcc_in_path(nvcc_path, sizeof(nvcc_path));

    void *nvrtc = dlopen("libnvrtc.so", RTLD_NOW);
    if (!nvrtc) nvrtc = dlopen("libnvrtc.so.1", RTLD_NOW);
    int has_nvrtc = (nvrtc != NULL);
    if (nvrtc) dlclose(nvrtc);

    if (facts) { facts->nvcc_available = has_nvcc; facts->nvrtc_available = has_nvrtc; }

    if (has_nvcc) {
        len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                                 "    nvcc found at %s -> CUDA C kernels can also be authored and compiled at build time.\n", nvcc_path);
    } else {
        len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                                 "    nvcc NOT found in PATH -> no build-time CUDA C compilation (hand-written PTX still works - see gpu_matmul.c).\n");
    }

    if (has_nvrtc) {
        len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                                 "    libnvrtc.so found -> CUDA C kernels can also be compiled at runtime, no nvcc needed.\n");
    } else {
        len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                                 "    libnvrtc.so NOT found -> no runtime CUDA C compilation either (hand-written PTX still works).\n");
    }
}
