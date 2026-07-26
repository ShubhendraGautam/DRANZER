/* NVIDIA Management Library probe. libnvidia-ml.so ships as part of the
 * NVIDIA kernel driver install itself (not the CUDA toolkit), so this
 * works even on a machine that has never had CUDA installed - it's the
 * same library `nvidia-smi` calls internally. NVML's C ABI is stable and
 * public; the structs/prototypes below are declared directly rather than
 * requiring the CUDA toolkit's nvml.h to be present. */

#include "backends/gpu/gpu_probe.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int nvmlReturn_t;
typedef struct nvmlDevice_st *nvmlDevice_t;
typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;
typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef nvmlReturn_t (*fn_nvmlInit)(void);
typedef nvmlReturn_t (*fn_nvmlShutdown)(void);
typedef nvmlReturn_t (*fn_nvmlDeviceGetCount)(unsigned int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetName)(nvmlDevice_t, char *, unsigned int);
typedef nvmlReturn_t (*fn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetCudaComputeCapability)(nvmlDevice_t, int *, int *);
typedef nvmlReturn_t (*fn_nvmlSystemGetDriverVersion)(char *, unsigned int);
typedef nvmlReturn_t (*fn_nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t *);
typedef const char *(*fn_nvmlErrorString)(nvmlReturn_t);

void gpu_backend_nvml_probe(gpu_backend_report_t *out, gpu_nvml_facts_t *facts) {
    out->status = GPU_BACKEND_UNAVAILABLE;
    out->message[0] = '\0';
    size_t len = 0;
    if (facts) {
        memset(facts, 0, sizeof(*facts));
        facts->cc_major = -1;
        facts->cc_minor = -1;
    }

    void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (!lib) {
        snprintf(out->message, sizeof(out->message),
                 "libnvidia-ml.so not found - no NVIDIA driver installed (or not visible to this process).");
        return;
    }

#define LOAD(name) fn_##name name = (fn_##name)dlsym(lib, #name); \
    if (!name) { \
        snprintf(out->message, sizeof(out->message), "libnvidia-ml.so loaded, but symbol %s missing (unexpected driver ABI?).", #name); \
        dlclose(lib); \
        return; \
    }

    LOAD(nvmlInit)
    LOAD(nvmlShutdown)
    LOAD(nvmlDeviceGetCount)
    LOAD(nvmlDeviceGetHandleByIndex)
    LOAD(nvmlDeviceGetName)
    LOAD(nvmlDeviceGetMemoryInfo)
    LOAD(nvmlDeviceGetCudaComputeCapability)
    LOAD(nvmlSystemGetDriverVersion)
    LOAD(nvmlDeviceGetUtilizationRates)
    LOAD(nvmlErrorString)
#undef LOAD

    nvmlReturn_t rc = nvmlInit();
    if (rc != 0) {
        snprintf(out->message, sizeof(out->message),
                 "libnvidia-ml.so loaded, but nvmlInit() failed (%s) - driver library present but not functional.",
                 nvmlErrorString(rc));
        dlclose(lib);
        return;
    }

    char driver_version[128] = {0};
    nvmlSystemGetDriverVersion(driver_version, sizeof(driver_version));
    len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                             "NVIDIA driver version: %s\n", driver_version);
    if (facts) strncpy(facts->driver_version, driver_version, sizeof(facts->driver_version) - 1);

    unsigned int count = 0;
    nvmlDeviceGetCount(&count);
    len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                             "    GPU count: %u\n", count);
    if (facts) facts->device_count = (int)count;

    for (unsigned int i = 0; i < count && len < sizeof(out->message); i++) {
        nvmlDevice_t dev;
        if (nvmlDeviceGetHandleByIndex(i, &dev) != 0) continue;

        char name[128] = {0};
        nvmlDeviceGetName(dev, name, sizeof(name));

        nvmlMemory_t mem = {0};
        nvmlDeviceGetMemoryInfo(dev, &mem);

        int major = -1, minor = -1;
        char cc_buf[16] = "unknown";
        int have_cc = (nvmlDeviceGetCudaComputeCapability(dev, &major, &minor) == 0);
        if (have_cc) snprintf(cc_buf, sizeof(cc_buf), "%d.%d", major, minor);

        nvmlUtilization_t util = {0};
        nvmlDeviceGetUtilizationRates(dev, &util);

        len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                                 "    [%u] %s - %.0f MB total / %.0f MB free VRAM, compute capability %s, utilization %u%%\n",
                                 i, name, mem.total / (1024.0 * 1024.0), mem.free / (1024.0 * 1024.0),
                                 cc_buf, util.gpu);

        /* Facts capture device 0 as the representative device - good
         * enough for a capability cache keyed by "this GPU model"; a
         * multi-GPU machine with mixed models isn't this project's
         * concern (nor is it common). */
        if (facts && i == 0) {
            strncpy(facts->gpu_name, name, sizeof(facts->gpu_name) - 1);
            facts->vram_total_mb = (unsigned long long)(mem.total / (1024 * 1024));
            if (have_cc) { facts->cc_major = major; facts->cc_minor = minor; }
        }
    }

    nvmlShutdown();
    dlclose(lib);
    out->status = GPU_BACKEND_READY;
}
