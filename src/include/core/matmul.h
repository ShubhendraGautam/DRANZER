#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

/* CPU matrix-multiplication kernels and their selection policy.
 *
 * Split out of tensor_ops.* because matmul is the only primitive with more
 * than one implementation: everything here exists so a kernel can be chosen
 * from measurements (see tools/bench_matmul.c and docs/matmul.md) instead of
 * being hard-coded, and so every candidate stays independently selectable
 * and comparable against the portable scalar reference.
 *
 * All kernels compute C (m x n) = A (m x k) @ B (k x n) over row-major
 * buffers and fully define C (no accumulation into prior contents). A, B,
 * and C must not overlap: they are `restrict`-qualified because the
 * difference is not cosmetic - without it a compiler must assume every
 * accumulator store can alias the next B load, which measured 3-4x slower
 * under clang on the multi-row kernels (docs/matmul.md).
 * They differ only in traversal order, so they agree with the scalar
 * reference up to float summation order - never structurally. */

typedef enum {
    /* The measured default: matmul_select() resolves this to the kernel the
     * sweep in docs/matmul.md picked. */
    MATMUL_KERNEL_AUTO = 0,
    /* Portable unblocked ijk reference. Slow but the definition of correct:
     * every other kernel is measured and validated against it. */
    MATMUL_KERNEL_SCALAR,
    /* Unblocked i/l/j. No tiling bookkeeping, contiguous stores, one pass
     * over B per row of A. Measured fastest on Clang for single-token decode
     * shapes and slower than the blocked kernels on GCC for the same shapes,
     * which is why it is selectable but not the default. */
    MATMUL_KERNEL_ROWWISE,
    /* Cache-blocked i/l/j over `matmul_tile_size()`-sized tiles. */
    MATMUL_KERNEL_TILED,
    /* Cache-blocked, plus four rows of A per pass so each loaded B element
     * feeds four independent accumulator rows. Raises arithmetic intensity
     * on the multi-row prefill/training shapes. */
    MATMUL_KERNEL_TILED_MR4,
    MATMUL_KERNEL_COUNT
} matmul_kernel_t;

/* Compile-time default tile, overridable with -DDRANZER_MATMUL_BLOCK_SIZE.
 * The runtime setter below is what the benchmark sweeps. 256 was measured
 * against 16, 32, 64, and 128 across both compilers and every model tier
 * (docs/matmul.md); at this project's shapes it mostly means blocking applies
 * to the output width alone, which is where it earned its keep. */
#ifndef DRANZER_MATMUL_BLOCK_SIZE
#define DRANZER_MATMUL_BLOCK_SIZE 256
#endif

/* Run one specific kernel. MATMUL_KERNEL_AUTO applies the selection policy;
 * any other value runs exactly that kernel regardless of shape. Used by the
 * benchmark and the equivalence tests to compare candidates directly. */
void matmul_run(matmul_kernel_t kernel, const float *restrict A,
                const float *restrict B, float *restrict C,
                size_t m, size_t k, size_t n);

/* The kernel MATMUL_KERNEL_AUTO resolves to for this shape. Deterministic -
 * the same shape always selects the same kernel within one binary, which is
 * what keeps training runs reproducible. It currently answers the same for
 * every shape: the measurements did not support a split (see matmul.c and
 * docs/matmul.md). It stays shape-aware in the interface because that is
 * where runtime SIMD dispatch will hook in. */
matmul_kernel_t matmul_select(size_t m, size_t k, size_t n);

/* Process-wide kernel override. MATMUL_KERNEL_AUTO (the default) restores
 * shape-directed selection. Returns 0, or -1 for an out-of-range kernel,
 * leaving the previous setting untouched. Not synchronized: set it before
 * starting compute, not from inside a parallel region. */
int matmul_set_kernel(matmul_kernel_t kernel);
matmul_kernel_t matmul_get_kernel(void);

/* Tile edge used by the blocked kernels. Returns 0, or -1 for a zero tile,
 * leaving the previous setting untouched. */
int matmul_set_tile_size(size_t tile);
size_t matmul_tile_size(void);

/* Stable lowercase kernel names ("auto", "scalar", "rowwise", "tiled",
 * "tiled_mr4") for CLI flags, CSV rows, and documentation.
 * matmul_kernel_from_name() returns -1 for an unknown name. */
const char *matmul_kernel_name(matmul_kernel_t kernel);
int matmul_kernel_from_name(const char *name, matmul_kernel_t *kernel_out);

/* The project's general-purpose matmul: dispatches through the current
 * kernel setting. Zeroes C first.
 *
 * Parallelized with OpenMP inside the blocked kernels when built with OMP=1:
 * blocks own disjoint regions of C, so threads never write the same output
 * and there is no cross-thread reduction - results are bit-identical to a
 * serial build. */
void matrix_multiply(float *restrict A, float *restrict B, float *restrict C,
                     size_t m, size_t k, size_t n);

/* Portable unblocked reference, always independently selectable regardless
 * of the current kernel setting (this is what model->use_scalar_matmul and
 * `bench.out --scalar` reach for). */
void matrix_multiply_scalar(const float *restrict A, const float *restrict B,
                            float *restrict C, size_t m, size_t k, size_t n);

/* Backprop through C = A @ B (A: m x k, B: k x n, C: m x n).
 * Accumulates (+=) into dA; caller must zero dA first for a fresh gradient.
 * Parallel over i: each iteration owns a disjoint row of dA. */
void matmul_backward_input(float *restrict dC, float *restrict B, float *restrict dA,
                            size_t m, size_t k, size_t n);

/* Backprop through C = A @ B (A: m x k, B: k x n, C: m x n).
 * Accumulates (+=) into dB; caller must zero dB first for a fresh gradient.
 * Parallel over l: each iteration owns a disjoint row of dB. */
void matmul_backward_weight(float *restrict A, float *restrict dC, float *restrict dB,
                             size_t m, size_t k, size_t n);

#endif // MATMUL_H
