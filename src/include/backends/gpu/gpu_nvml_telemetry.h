#ifndef GPU_NVML_TELEMETRY_H
#define GPU_NVML_TELEMETRY_H

/* Live NVML telemetry beyond gpu_backend_nvml.c's basic identity facts
 * (name/VRAM/compute capability) - temperature, power, clocks, ECC,
 * performance state. Every field is independently marked present/absent
 * since not all of these are supported on every GPU (e.g. laptop GPUs
 * often don't expose a controllable fan speed, some don't support ECC).
 * All from documented NVML calls - see gpu_nvml_telemetry.c. */
typedef struct {
    int has_temperature_c; unsigned int temperature_c;
    int has_power_mw; unsigned int power_mw;
    int has_power_limit_mw; unsigned int power_limit_mw;
    int has_clock_graphics_mhz; unsigned int clock_graphics_mhz;
    int has_clock_sm_mhz; unsigned int clock_sm_mhz;
    int has_clock_mem_mhz; unsigned int clock_mem_mhz;
    int has_max_clock_graphics_mhz; unsigned int max_clock_graphics_mhz;
    int has_max_clock_mem_mhz; unsigned int max_clock_mem_mhz;
    int has_ecc_mode; int ecc_enabled;
    int has_performance_state; int performance_state; /* P-state: 0 = max performance */
    int has_persistence_mode; int persistence_mode;
} gpu_nvml_telemetry_t;

/* Probes device 0. Leaves has_* fields at 0 (and their value fields
 * untouched) for anything unavailable/unsupported - not an error, just
 * this GPU/driver not exposing that particular query. */
void gpu_nvml_telemetry_probe(gpu_nvml_telemetry_t *out);

void gpu_nvml_telemetry_print(const gpu_nvml_telemetry_t *t);

#endif // GPU_NVML_TELEMETRY_H
