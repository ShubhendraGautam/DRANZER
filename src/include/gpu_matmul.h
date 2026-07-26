#ifndef GPU_MATMUL_H
#define GPU_MATMUL_H

#include <stddef.h>

/* GPU matrix multiply: C (m x n) = A (m x k) @ B (k x n), matching
 * tensor_ops.h's matrix_multiply() signature/semantics exactly so the two
 * are drop-in comparable (see tests/test_gpu_matmul.c). Backed by a
 * hand-written PTX kernel (gpu_matmul.c) loaded through gpu_cuda.h - no
 * CUDA toolkit, no NVRTC, just libcuda.so.1.
 *
 * Lazily initializes a CUDA context on first call and keeps it alive for
 * reuse; NVIDIA-only, and does nothing useful on a machine without a
 * working CUDA driver - always check gpu_matmul_available() first. */

/* Returns 1 if a CUDA GPU is usable right now, 0 otherwise. Safe to call
 * repeatedly; only actually probes hardware once. */
int gpu_matmul_available(void);

/* Returns 0 on success, -1 if the GPU path is unavailable or the kernel
 * launch failed - callers should fall back to matrix_multiply() (CPU) in
 * either case, not treat this as fatal. */
int gpu_matmul(const float *A, const float *B, float *C, size_t m, size_t k, size_t n);

/* Releases the lazily-created CUDA context, if any. Optional - not
 * calling this just means the context lives until process exit. */
void gpu_matmul_shutdown(void);

#endif // GPU_MATMUL_H
