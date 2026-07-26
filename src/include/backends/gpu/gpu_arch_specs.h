#ifndef GPU_ARCH_SPECS_H
#define GPU_ARCH_SPECS_H

/* Static lookup table: CUDA compute capability -> architecture codename
 * and CUDA cores per SM, sourced from NVIDIA's publicly published
 * "Compute Capabilities" reference (CUDA C Programming Guide appendix).
 * Used to turn this project's own live measurements (SM count via
 * cuDeviceGetAttribute, clock rate via NVML/CUDA) into a theoretical peak
 * FLOPS estimate - see gpu_theoretical_perf.h.
 *
 * Deliberately conservative: returns 0/NULL for any compute capability
 * not in this table rather than guessing, since an architecture this
 * project hasn't seen yet could have a different cores-per-SM ratio. */

/* Cores per SM for (cc_major, cc_minor), or 0 if unknown. */
int gpu_arch_cores_per_sm(int cc_major, int cc_minor);

/* Architecture codename (e.g. "Turing"), or NULL if unknown. */
const char *gpu_arch_name(int cc_major, int cc_minor);

#endif // GPU_ARCH_SPECS_H
