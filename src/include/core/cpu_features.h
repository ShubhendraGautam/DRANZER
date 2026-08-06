#ifndef CPU_FEATURES_H
#define CPU_FEATURES_H

/* Runtime CPU instruction-set detection.
 *
 * This exists so one binary can carry SIMD kernels it is not allowed to run
 * unconditionally. The project deliberately does not build with -march=native
 * (see the Makefile): a binary tuned for the build machine raises SIGILL on
 * anything older. The SIMD matmul kernels are therefore compiled with
 * per-function target attributes and gated behind the queries below, so the
 * same executable uses AVX-512 where it exists, AVX2 where it does not, and
 * portable C on everything else.
 *
 * Detection happens once, on first query, and the answer is cached. It is safe
 * to call from anywhere including inside an OpenMP region, but the first call
 * should happen before compute starts - see cpu_features_detect().
 */

typedef enum {
    /* Portable C. Always available, by definition. */
    CPU_ISA_BASELINE = 0,
    /* x86-64: AVX2 and FMA together. The kernels use both, so both are
     * required before this reports available - FMA is a separate CPUID bit
     * and there are real CPUs with one and not the other. */
    CPU_ISA_AVX2,
    /* x86-64: AVX-512F and AVX-512VL. F alone would do for the 512-bit body,
     * but VL is what makes the masked tails usable on the short rows this
     * project's shapes produce. */
    CPU_ISA_AVX512,
    /* AArch64 Advanced SIMD. Mandatory in ARMv8-A, so on aarch64 this is a
     * compile-time yes; it is still routed through the same query so callers
     * never special-case the architecture. */
    CPU_ISA_NEON,
    CPU_ISA_COUNT
} cpu_isa_t;

/* Whether this CPU can execute code compiled for `isa`. CPU_ISA_BASELINE is
 * always 1; an ISA belonging to another architecture is always 0. */
int cpu_isa_available(cpu_isa_t isa);

/* The widest available ISA for this architecture, or CPU_ISA_BASELINE when
 * there is none. Ordering is AVX-512 > AVX2 on x86-64 and NEON on aarch64;
 * this answers "what is supported", not "what is fastest" - that is
 * matmul_select()'s job, and it is a measured question. */
cpu_isa_t cpu_isa_best(void);

/* Stable lowercase names ("baseline", "avx2", "avx512", "neon") for CLI flags,
 * CSV provenance columns, and documentation.
 * cpu_isa_from_name() returns -1 for an unknown name. */
const char *cpu_isa_name(cpu_isa_t isa);
int cpu_isa_from_name(const char *name, cpu_isa_t *isa_out);

/* One-line summary of what was detected, e.g. "avx512 (avx2, avx512)" or
 * "baseline (capped by DRANZER_CPU_ISA=baseline)". Points into static storage;
 * valid until the next cpu_features_set_max_isa() call. */
const char *cpu_features_summary(void);

/* Force detection now rather than on first query. Optional: every query
 * detects on demand. Worth calling during startup so the one-time CPUID cost
 * and the cache write never land inside a parallel region. */
void cpu_features_detect(void);

/* Cap detection at `isa`: anything wider reports unavailable from here on.
 * This can only ever remove ISAs. Raising the cap can never enable something
 * the CPU lacks, because the cap is applied on top of real detection - asking
 * for AVX-512 on a machine without it still reports unavailable.
 *
 * This is how the "runs on a machine lacking the instruction set" path is
 * tested without owning that machine: cap to baseline and the dispatch, the
 * selection policy, and every caller must still produce correct results.
 * Returns 0, or -1 for an out-of-range ISA, leaving the cap untouched.
 *
 * Not synchronized, and it invalidates nothing that is already running: set it
 * before compute starts, not from inside a parallel region.
 *
 * The environment variable DRANZER_CPU_ISA applies the same cap at detection
 * time, which is what lets a benchmark compare ISA paths without recompiling. */
int cpu_features_set_max_isa(cpu_isa_t isa);
cpu_isa_t cpu_features_max_isa(void);

/* Remove the cap, restoring whatever the hardware actually supports. This is
 * how a test that capped to baseline puts the process back; there is no
 * "highest ISA" value to pass to the setter that means the same thing on
 * every architecture. Does not re-read DRANZER_CPU_ISA - an explicit call
 * outranks the environment. */
void cpu_features_clear_max_isa(void);

#endif // CPU_FEATURES_H
