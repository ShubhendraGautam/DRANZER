#ifndef GPU_PROBE_H
#define GPU_PROBE_H

/* Shared report type for every gpu_backend_*.c probe. Each backend module
 * (drm, nvml, cuda, opencl) is self-contained and knows nothing about the
 * others - gpu_probe.c (the only file with main()) calls each in turn and
 * prints a combined report + recommendation.
 *
 * Every probe here uses dlopen()/dlsym() against whatever runtime
 * libraries are already present on the system, declaring the handful of
 * function prototypes/structs it needs itself rather than requiring the
 * vendor SDK's headers to be installed. That's deliberate: it lets this
 * tool answer "what can this machine do right now" even when no GPU SDK
 * (CUDA toolkit, OpenCL dev headers) is installed at all - which is
 * exactly the gap between "the driver is there" and "you can compile GPU
 * code" that this tool exists to surface.
 *
 * Linux-only (relies on <dlfcn.h> and /dev, /proc, /etc paths) - no
 * Windows/macOS support attempted here.
 */

typedef enum {
    GPU_BACKEND_UNAVAILABLE, /* nothing found - the runtime library itself isn't present */
    GPU_BACKEND_PARTIAL,     /* something is present (driver, loader) but compute isn't fully usable yet */
    GPU_BACKEND_READY,       /* usable for compute right now, as far as this probe can tell */
} gpu_backend_status_t;

typedef struct {
    gpu_backend_status_t status;
    char message[2048]; /* multi-line, printf-ready summary of what was found */
} gpu_backend_report_t;

/* Structured facts alongside each backend's human-readable report - same
 * probe pass, machine-readable output, used by gpu_capability_cache.c to
 * persist a reusable profile. All fields come from the same public,
 * documented APIs as the report message; nothing here is derived from any
 * undocumented interface. */
typedef struct {
    int device_count;
    char driver_version[64];
    char gpu_name[128];
    unsigned long long vram_total_mb;
    int cc_major, cc_minor; /* -1 if unknown */
} gpu_nvml_facts_t;

typedef struct {
    int driver_functional;
    int driver_version;   /* e.g. 13000 for CUDA 13.0 */
    int device_count;
    int nvcc_available;
    int nvrtc_available;
} gpu_cuda_facts_t;

typedef struct {
    int loader_functional;
    int platform_count;
    char platform_names[256]; /* comma-separated */
    int total_device_count;
} gpu_opencl_facts_t;

/* Generic GPU device node presence (/dev/dri) - vendor-agnostic, works
 * even when no vendor driver userspace library is installed. */
void gpu_backend_drm_probe(gpu_backend_report_t *out);

/* NVIDIA Management Library (libnvidia-ml.so) - ships with the NVIDIA
 * driver itself, no CUDA toolkit required. Gives real device name, VRAM,
 * CUDA compute capability, driver version, live utilization. `facts` may
 * be NULL if the caller only wants the human-readable report. */
void gpu_backend_nvml_probe(gpu_backend_report_t *out, gpu_nvml_facts_t *facts);

/* CUDA: distinguishes the driver API (libcuda.so, ships with the driver)
 * from the ability to actually compile a kernel (nvcc at build time, or
 * NVRTC/libnvrtc.so for compiling CUDA C at runtime) - having the former
 * without the latter is a common, easy-to-miss state (it's this sandbox's
 * exact state) where cuInit() succeeds but there's still no way to get a
 * kernel onto the GPU. `facts` may be NULL. */
void gpu_backend_cuda_probe(gpu_backend_report_t *out, gpu_cuda_facts_t *facts);

/* OpenCL: distinguishes the ICD loader (libOpenCL.so) from vendor ICDs
 * actually being registered (/etc/OpenCL/vendors/ (*.icd files)) - the loader can
 * be present and fully functional while reporting zero platforms if no
 * vendor has registered one, which looks confusingly like "nothing is
 * installed" unless the two are checked separately. `facts` may be NULL. */
void gpu_backend_opencl_probe(gpu_backend_report_t *out, gpu_opencl_facts_t *facts);

#endif // GPU_PROBE_H
