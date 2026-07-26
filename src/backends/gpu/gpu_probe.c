/*
 * GPU capability probe: reports what this machine can actually do for GPU
 * compute right now - not just "is a GPU present" but "which of
 * DRM/NVML/CUDA/OpenCL are usable, and what's missing for the rest" - then
 * gives a concrete, actionable recommendation. Also saves what it found to
 * a reusable cache file (gpu_capability_cache.c) keyed by GPU model +
 * driver version, so a researcher on matching hardware can skip
 * re-probing - or just `--load` a cache file someone else already shared.
 *
 * Every check here works via dlopen()/dlsym() against whatever runtime
 * libraries already exist on the system, so this tool itself never needs
 * the CUDA toolkit or OpenCL dev headers to build or run - only libdl
 * (`-ldl`), which is always available. See gpu_probe.h for why each
 * backend is checked the way it is, and gpu_matmul.c for this project's
 * actual (hand-written-PTX, zero-toolkit) GPU compute path.
 *
 * Its own file/binary (gpu_probe.out) on purpose, consistent with bench.c:
 * it has nothing to do with the model or the CLI.
 */

#include "include/gpu_probe.h"
#include "include/gpu_capability_cache.h"
#include "include/gpu_cuda.h"
#include "include/gpu_cuda_attrs.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#define DEFAULT_CACHE_DIR "gpu_capability_cache"

static void print_report(const char *label, const gpu_backend_report_t *r) {
    const char *status_str =
        r->status == GPU_BACKEND_READY ? "READY" :
        r->status == GPU_BACKEND_PARTIAL ? "PARTIAL" : "UNAVAILABLE";
    printf("--- %s: %s ---\n", label, status_str);
    printf("    %s\n\n", r->message);
}

static int run_probe(void) {
    printf("=== GPU Capability Probe ===\n\n");

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    printf("CPU: %ld logical core(s) online (CPU-only fallback is always available - see `make bench`)\n\n", ncpu);

    gpu_backend_report_t drm, nvml, cuda, opencl;
    gpu_capability_t cap;
    memset(&cap, 0, sizeof(cap));
    cap.probed_at = time(NULL);

    gpu_backend_drm_probe(&drm);
    gpu_backend_nvml_probe(&nvml, &cap.nvml);
    gpu_backend_cuda_probe(&cuda, &cap.cuda);
    gpu_backend_opencl_probe(&opencl, &cap.opencl);
    cap.has_nvml = (nvml.status != GPU_BACKEND_UNAVAILABLE);
    cap.has_cuda = (cuda.status != GPU_BACKEND_UNAVAILABLE);
    cap.has_opencl = (opencl.status != GPU_BACKEND_UNAVAILABLE);

    print_report("DRM device nodes", &drm);
    print_report("NVML (driver-level GPU profiling)", &nvml);
    print_report("CUDA", &cuda);
    print_report("OpenCL", &opencl);

    if (cuda.status == GPU_BACKEND_READY) {
        gpu_cuda_ctx_t *attr_ctx = gpu_cuda_init();
        if (attr_ctx) {
            printf("--- CUDA device attributes (table-driven cuDeviceGetAttribute sweep) ---\n");
            gpu_cuda_attrs_print_all(attr_ctx);
            printf("\n");
            gpu_cuda_shutdown(attr_ctx);
        }
    }

    printf("=== Recommendation ===\n");
    if (cuda.status == GPU_BACKEND_READY) {
        printf("CUDA driver is functional - gpu_matmul() (hand-written PTX, no toolkit needed) can run\n"
               "on this GPU right now. Try `make test` (test_gpu_matmul.c exercises it against the CPU\n"
               "reference) or call gpu_matmul_available()/gpu_matmul() directly from gpu_matmul.h.\n");
    } else if (opencl.status == GPU_BACKEND_READY) {
        printf("OpenCL has a usable platform+device - this project's GPU compute path currently targets\n"
               "CUDA only (gpu_matmul.c); an OpenCL kernel backend would need to be written to use this.\n");
    } else if (cuda.status == GPU_BACKEND_PARTIAL || opencl.status == GPU_BACKEND_PARTIAL) {
        printf("A GPU is visible to this machine, but no compute backend is fully usable yet:\n");
        if (opencl.status == GPU_BACKEND_PARTIAL) {
            printf("  - OpenCL: loader present, no vendor platform registered (see above for how to fix).\n");
        }
        if (cuda.status == GPU_BACKEND_PARTIAL) {
            printf("  - CUDA: driver present but not functional (see above).\n");
        }
        printf("Until one of those is resolved, use the CPU path (`make` / `make OMP=1 CC=gcc`) - see\n"
               "`make bench` for what this machine's CPU can actually deliver.\n");
    } else {
        printf("No GPU compute backend detected at all. This is a CPU-only machine as far as this\n"
               "probe can tell - use `make` (optionally `make OMP=1 CC=gcc` for multi-core) and size\n"
               "models per `make bench`'s guidance.\n");
    }

    printf("\n=== Capability Cache ===\n");
    if (gpu_capability_cache_save(&cap, DEFAULT_CACHE_DIR) == 0) {
        char path[1024];
        gpu_capability_cache_filename(&cap, DEFAULT_CACHE_DIR, path, sizeof(path));
        printf("Saved to %s - reuse it with `./gpu_probe.out --load %s`,\n"
               "or share the file with another researcher on the same GPU model + driver version.\n", path, path);
    } else {
        printf("Not saved (no NVML GPU name available to key the cache by - nothing to reuse yet).\n");
    }

    return 0;
}

static int run_load(const char *path) {
    gpu_capability_t cap;
    if (gpu_capability_cache_load(path, &cap) != 0) {
        fprintf(stderr, "Could not read capability cache file: %s\n", path);
        return 1;
    }
    printf("=== Loaded GPU Capability Profile (%s) ===\n\n", path);
    gpu_capability_print(&cap);
    printf("\n(This came from a cache file, not a fresh probe - run gpu_probe.out with no arguments\n"
           "to re-probe this machine directly.)\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--load") == 0) {
        return run_load(argv[2]);
    }
    return run_probe();
}
