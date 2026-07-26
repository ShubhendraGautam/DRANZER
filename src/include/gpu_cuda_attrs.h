#ifndef GPU_CUDA_ATTRS_H
#define GPU_CUDA_ATTRS_H

#include "gpu_cuda.h"

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

extern const gpu_cuda_attr_entry_t GPU_CUDA_ATTR_TABLE[];
extern const int GPU_CUDA_ATTR_TABLE_COUNT;

/* Runs the whole table against ctx's device, printing "name = value" for
 * every attribute that resolved successfully (silently skipping ones the
 * driver doesn't recognize - expected for some, not an error). */
void gpu_cuda_attrs_print_all(gpu_cuda_ctx_t *ctx);

#endif // GPU_CUDA_ATTRS_H
