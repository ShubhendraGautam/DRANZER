#ifndef GPU_CUDA_H
#define GPU_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* Minimal CUDA Driver API wrapper - dlopen("libcuda.so.1") plus the dozen
 * or so entry points needed to load a PTX module (as a plain string,
 * JIT-compiled by the driver itself) and run a kernel on it. No CUDA
 * toolkit, no NVRTC, no nvcc: the ONLY runtime dependency is the driver
 * library that ships with any NVIDIA driver install - see gpu_backend_cuda.c
 * for how that's confirmed available before this module is ever used.
 *
 * This is deliberately the ONLY file in the project that knows about CUDA
 * Driver API types/calls - gpu_matmul.c (and any future GPU kernel module)
 * goes through this, never dlopen's libcuda.so.1 itself.
 *
 * NVIDIA-only, Linux-only. Not built by default - opt in with `make GPU=1`.
 */

typedef struct gpu_cuda_ctx gpu_cuda_ctx_t; /* opaque */

/* Loads libcuda.so.1, initializes the driver, creates a context on device
 * 0. Returns NULL if the driver library isn't present or isn't functional
 * (e.g. no NVIDIA GPU) - callers should treat that as "GPU path
 * unavailable, fall back to CPU", not as an error to propagate loudly. */
gpu_cuda_ctx_t *gpu_cuda_init(void);

void gpu_cuda_shutdown(gpu_cuda_ctx_t *ctx);

/* Loads PTX source (a NUL-terminated string, JIT-compiled by the driver on
 * the spot) as a module and resolves `kernel_name` from it. Returns 0 on
 * success. The most recently loaded kernel is what gpu_cuda_launch_2d runs. */
int gpu_cuda_load_kernel(gpu_cuda_ctx_t *ctx, const char *ptx_src, const char *kernel_name);

/* Device memory. gpu_cuda_alloc returns 0 on failure (0 is never a valid
 * device pointer). */
uint64_t gpu_cuda_alloc(gpu_cuda_ctx_t *ctx, size_t bytes);
void gpu_cuda_free(gpu_cuda_ctx_t *ctx, uint64_t dptr);
int gpu_cuda_upload(gpu_cuda_ctx_t *ctx, uint64_t dptr, const void *host, size_t bytes);
int gpu_cuda_download(gpu_cuda_ctx_t *ctx, void *host, uint64_t dptr, size_t bytes);

/* Launches the kernel loaded by the most recent gpu_cuda_load_kernel call,
 * with a 2D grid of 2D blocks (z dimension fixed at 1 - nothing in this
 * project needs a 3rd dimension). `args` follows the CUDA Driver API's own
 * convention: an array of pointers, each pointing to one kernel argument's
 * value, in the same order as the kernel's .param declarations. Blocks
 * until the kernel completes and returns 0 on success (nonzero means the
 * launch OR the kernel itself failed - e.g. an illegal memory access). */
int gpu_cuda_launch_2d(gpu_cuda_ctx_t *ctx,
                        unsigned int grid_x, unsigned int grid_y,
                        unsigned int block_x, unsigned int block_y,
                        void **args);

/* Wraps cuDeviceGetAttribute(), a single documented CUDA Driver API call
 * that accepts ~100 different documented CUdevice_attribute codes (max
 * threads/block, clock rate, memory bus width, ECC status, cooperative
 * launch support, ...) - see gpu_cuda_attrs.h for the table-driven sweep
 * built on top of this instead of hand-writing a probe per attribute.
 * Returns 0 and fills *out on success; nonzero (and leaves *out
 * untouched) if `attrib` isn't a recognized/supported code on this
 * device - that's an expected, non-fatal outcome for a sweep, not an error
 * to log loudly. */
int gpu_cuda_device_attribute(gpu_cuda_ctx_t *ctx, int attrib, int *out);

#endif // GPU_CUDA_H
