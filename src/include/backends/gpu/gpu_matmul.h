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
 * working CUDA driver - always check gpu_matmul_available() first.
 *
 * Two caches keep repeated calls from paying for the same work twice,
 * which is what made the first (naive) version of this file 20-80x
 * *slower* than CPU at this project's typical model sizes:
 *   - A and C get persistent scratch device buffers, grown on demand and
 *     reused across calls, instead of a fresh cuMemAlloc/cuMemFree every
 *     call (the dominant cost - allocation/deallocation on the driver is
 *     far more expensive than the transfer itself for small buffers).
 *   - B (the weight matrix in every call site this project actually has -
 *     see transformer.c) gets a persistent device buffer keyed by its
 *     host pointer, uploaded once and re-used until
 *     gpu_matmul_invalidate_weights() is called - the model's weights
 *     don't change between calls except when the optimizer updates them. */

/* Returns 1 if a CUDA GPU is usable right now, 0 otherwise. Safe to call
 * repeatedly; only actually probes hardware once. */
int gpu_matmul_available(void);

/* Returns 0 on success, -1 if the GPU path is unavailable or the kernel
 * launch failed - callers should fall back to matrix_multiply() (CPU) in
 * either case, not treat this as fatal. */
int gpu_matmul(const float *A, const float *B, float *C, size_t m, size_t k, size_t n);

/* Backward-pass counterparts of core/matmul.h's matmul_backward_input() and
 * matmul_backward_weight(), with identical shapes and identical semantics
 * including the accumulate-into-destination contract:
 *
 *   backward_input:   dA (m x k) += dC (m x n) @ B_transposed
 *   backward_weight:  dB (k x n) += A_transposed @ dC (m x n)
 *
 * Same return convention as gpu_matmul(): 0 on success, -1 means "use the CPU
 * for this one", never fatal.
 *
 * These are more transfer-heavy per call than gpu_matmul(). Because the
 * destination accumulates, it is uploaded as well as downloaded - four
 * transfers against the forward path's two, and for backward_weight the
 * destination is a full weight-sized matrix. At small shapes that overhead
 * exceeds the arithmetic saved, which is why core/training.c dispatches these
 * on a measured shape threshold rather than whenever a GPU exists. See
 * docs/gpu.md for the numbers behind that threshold. */
int gpu_matmul_backward_input(const float *dC, const float *B, float *dA,
                              size_t m, size_t k, size_t n);
int gpu_matmul_backward_weight(const float *A, const float *dC, float *dB,
                               size_t m, size_t k, size_t n);

/* Marks every cached GPU-resident weight buffer stale. Must be called
 * once after anything changes ANY model weight (training.c calls this
 * once per model_optimizer_step) - without it, gpu_matmul() would keep
 * using pre-update weight values forever, which is a silent correctness
 * bug, not a crash. Safe/cheap to call even if nothing is cached yet or
 * no GPU is in use. */
void gpu_matmul_invalidate_weights(void);

/* Releases the lazily-created CUDA context and every cached buffer, if
 * any. Optional - not calling this just means they live until process exit. */
void gpu_matmul_shutdown(void);

#endif // GPU_MATMUL_H
