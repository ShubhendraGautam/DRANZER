#ifndef GPU_CUDA_ATTRS_H
#define GPU_CUDA_ATTRS_H

#include "backends/gpu/gpu_cuda.h"

/* Table-driven sweep over cuDeviceGetAttribute()'s documented attribute
 * codes - one stable CUDA Driver API function, looped over many known
 * codes, instead of hand-writing a separate probe per device property.
 *
 * Deliberately conservative: only attribute codes from the long-stable,
 * widely-referenced core of the CUdevice_attribute enum are included here
 * (the ones present since CUDA's earliest driver API versions). Getting a
 * numeric code wrong wouldn't crash anything (cuDeviceGetAttribute
 * validates the code is in range and just errors on an unrecognized one),
 * but it COULD silently report a real value under the wrong label if the
 * wrong code happens to be valid for a different attribute - a real
 * correctness risk for a project built on verifying its own output rather
 * than trusting it. Expanding this table should be done by checking the
 * attribute code against NVIDIA's published cuda.h or Driver API
 * reference, not from memory. */

typedef struct {
    const char *name;
    int code;
} gpu_cuda_attr_entry_t;

/* Verified names only - used purely for labeling, never for deciding
 * which codes to probe (see gpu_cuda_attrs_sweep below for that). */
extern const gpu_cuda_attr_entry_t GPU_CUDA_ATTR_TABLE[];
extern const int GPU_CUDA_ATTR_TABLE_COUNT;

/* Highest attribute code this sweep probes. CUDA's documented
 * CUdevice_attribute enum has grown over time (new architectures add new
 * attributes); bump this if a future CUDA version defines codes beyond
 * it - probing a code the driver doesn't recognize just fails cleanly, so
 * there's no harm in this being a bit generous. */
#define GPU_CUDA_ATTR_SWEEP_MAX 128

/* One entry per attribute code that resolved successfully during a sweep. */
typedef struct {
    int code;
    int value;
    const char *name; /* NULL if this code isn't in GPU_CUDA_ATTR_TABLE */
} gpu_cuda_attr_result_t;

/* Probes every attribute code from 1 to GPU_CUDA_ATTR_SWEEP_MAX against
 * ctx's device - not a hand-picked list, a full numeric sweep, so it
 * discovers whatever a given driver/GPU/CUDA-version combination actually
 * supports rather than only what this file's author already knew to ask
 * for. Fills `results` (caller-provided, must hold at least
 * GPU_CUDA_ATTR_SWEEP_MAX entries) with every code that resolved and
 * returns how many did. Codes not present in GPU_CUDA_ATTR_TABLE still
 * appear in the results with result->name == NULL ("unlabeled") - this
 * project only ever prints a name for a code once cross-validated against
 * an independent source (see gpu_cuda_attrs.c), so an unlabeled result is
 * not a bug, just an honest "value exists, name not yet verified here". */
int gpu_cuda_attrs_sweep(gpu_cuda_ctx_t *ctx, gpu_cuda_attr_result_t *results);

/* Convenience: runs the sweep and prints it - named codes as "NAME = value",
 * everything else as "[code N] (unlabeled) = value". */
void gpu_cuda_attrs_print_all(gpu_cuda_ctx_t *ctx);

#endif // GPU_CUDA_ATTRS_H
