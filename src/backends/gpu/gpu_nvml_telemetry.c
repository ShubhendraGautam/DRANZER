/* Live NVML telemetry - same dlopen("libnvidia-ml.so") approach as
 * gpu_backend_nvml.c, extended with documented queries nvidia-smi itself
 * uses: temperature, power, clocks, ECC mode, performance state,
 * persistence mode. Each query is independent and many GPUs (especially
 * laptop parts) don't support all of them - see gpu_nvml_telemetry.h. */

#include "backends/gpu/gpu_nvml_telemetry.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int nvmlReturn_t;
typedef struct nvmlDevice_st *nvmlDevice_t;

typedef nvmlReturn_t (*fn_nvmlInit)(void);
typedef nvmlReturn_t (*fn_nvmlShutdown)(void);
typedef nvmlReturn_t (*fn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetPowerManagementLimit)(nvmlDevice_t, unsigned int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetClockInfo)(nvmlDevice_t, int, unsigned int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetMaxClockInfo)(nvmlDevice_t, int, unsigned int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetEccMode)(nvmlDevice_t, int *, int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetPerformanceState)(nvmlDevice_t, int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetPersistenceMode)(nvmlDevice_t, int *);

#define NVML_TEMPERATURE_GPU 0
#define NVML_CLOCK_GRAPHICS 0
#define NVML_CLOCK_SM 1
#define NVML_CLOCK_MEM 2

void gpu_nvml_telemetry_probe(gpu_nvml_telemetry_t *out) {
    memset(out, 0, sizeof(*out));

    void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (!lib) return;

    fn_nvmlInit nvmlInit = (fn_nvmlInit)dlsym(lib, "nvmlInit");
    fn_nvmlShutdown nvmlShutdown = (fn_nvmlShutdown)dlsym(lib, "nvmlShutdown");
    fn_nvmlDeviceGetHandleByIndex nvmlDeviceGetHandleByIndex =
        (fn_nvmlDeviceGetHandleByIndex)dlsym(lib, "nvmlDeviceGetHandleByIndex");
    if (!nvmlInit || !nvmlShutdown || !nvmlDeviceGetHandleByIndex) { dlclose(lib); return; }

    if (nvmlInit() != 0) { dlclose(lib); return; }

    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex(0, &dev) != 0) { nvmlShutdown(); dlclose(lib); return; }

#define TRY(fn_type, fn_name, call) do { \
    fn_type fn_name = (fn_type)dlsym(lib, #fn_name); \
    if (fn_name) { call; } \
} while (0)

    TRY(fn_nvmlDeviceGetTemperature, nvmlDeviceGetTemperature, {
        if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &out->temperature_c) == 0) out->has_temperature_c = 1;
    });
    TRY(fn_nvmlDeviceGetPowerUsage, nvmlDeviceGetPowerUsage, {
        if (nvmlDeviceGetPowerUsage(dev, &out->power_mw) == 0) out->has_power_mw = 1;
    });
    TRY(fn_nvmlDeviceGetPowerManagementLimit, nvmlDeviceGetPowerManagementLimit, {
        if (nvmlDeviceGetPowerManagementLimit(dev, &out->power_limit_mw) == 0) out->has_power_limit_mw = 1;
    });
    TRY(fn_nvmlDeviceGetClockInfo, nvmlDeviceGetClockInfo, {
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &out->clock_graphics_mhz) == 0) out->has_clock_graphics_mhz = 1;
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &out->clock_sm_mhz) == 0) out->has_clock_sm_mhz = 1;
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &out->clock_mem_mhz) == 0) out->has_clock_mem_mhz = 1;
    });
    TRY(fn_nvmlDeviceGetMaxClockInfo, nvmlDeviceGetMaxClockInfo, {
        if (nvmlDeviceGetMaxClockInfo(dev, NVML_CLOCK_GRAPHICS, &out->max_clock_graphics_mhz) == 0) out->has_max_clock_graphics_mhz = 1;
        if (nvmlDeviceGetMaxClockInfo(dev, NVML_CLOCK_MEM, &out->max_clock_mem_mhz) == 0) out->has_max_clock_mem_mhz = 1;
    });
    TRY(fn_nvmlDeviceGetEccMode, nvmlDeviceGetEccMode, {
        int pending;
        if (nvmlDeviceGetEccMode(dev, &out->ecc_enabled, &pending) == 0) out->has_ecc_mode = 1;
    });
    TRY(fn_nvmlDeviceGetPerformanceState, nvmlDeviceGetPerformanceState, {
        if (nvmlDeviceGetPerformanceState(dev, &out->performance_state) == 0) out->has_performance_state = 1;
    });
    TRY(fn_nvmlDeviceGetPersistenceMode, nvmlDeviceGetPersistenceMode, {
        if (nvmlDeviceGetPersistenceMode(dev, &out->persistence_mode) == 0) out->has_persistence_mode = 1;
    });
#undef TRY

    nvmlShutdown();
    dlclose(lib);
}

void gpu_nvml_telemetry_print(const gpu_nvml_telemetry_t *t) {
    if (t->has_temperature_c) printf("    Temperature: %u C\n", t->temperature_c);
    if (t->has_power_mw) printf("    Power draw: %.1f W\n", t->power_mw / 1000.0);
    if (t->has_power_limit_mw) printf("    Power limit: %.1f W\n", t->power_limit_mw / 1000.0);
    if (t->has_clock_graphics_mhz) printf("    Graphics clock: %u MHz (max %u MHz)\n",
                                           t->clock_graphics_mhz, t->max_clock_graphics_mhz);
    if (t->has_clock_sm_mhz) printf("    SM clock: %u MHz\n", t->clock_sm_mhz);
    if (t->has_clock_mem_mhz) printf("    Memory clock: %u MHz (max %u MHz)\n",
                                      t->clock_mem_mhz, t->max_clock_mem_mhz);
    if (t->has_ecc_mode) printf("    ECC: %s\n", t->ecc_enabled ? "enabled" : "disabled");
    if (t->has_performance_state) printf("    Performance state: P%d (P0 = max performance)\n", t->performance_state);
    if (t->has_persistence_mode) printf("    Persistence mode: %s\n", t->persistence_mode ? "enabled" : "disabled");
}
