/*
 * Runtime CPU instruction-set detection - see core/cpu_features.h for the
 * contract and docs/matmul.md for what the answer is used for.
 *
 * The x86 path is deliberately paranoid about one thing: CPUID reporting a
 * feature is not permission to use it. AVX and AVX-512 add register state the
 * operating system has to agree to save across context switches, and an OS
 * that has not enabled that state leaves CPUID's feature bit set while the
 * first instruction faults. XCR0 is what actually answers the question, so
 * every wide-register ISA below is gated on it as well as on its feature bit.
 */

#include "core/cpu_features.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#define DRANZER_ARCH_X86_64 1
#include <cpuid.h>
#elif defined(__aarch64__)
#define DRANZER_ARCH_AARCH64 1
#endif

static const char *const isa_names[CPU_ISA_COUNT] = {
    "baseline", "avx2", "avx512", "neon"
};

/* Detection results, written once. Every field is a pure function of the
 * hardware and the environment, so repeating detection writes identical
 * values; that is what makes the unsynchronized lazy initialization below
 * safe in practice. Callers that want the write to happen at a known point
 * anyway can call cpu_features_detect() during startup. */
static int detected;
static int isa_supported[CPU_ISA_COUNT];
static cpu_isa_t max_isa = CPU_ISA_COUNT; /* sentinel: no cap requested yet */
static char summary[160];
static cpu_features_config_status_t config_status = CPU_FEATURES_CONFIG_OK;
static char invalid_environment_value[64];

#ifdef DRANZER_ARCH_X86_64
/* XCR0, the OS's declaration of which extended register state it preserves.
 * Only valid to execute when CPUID reported OSXSAVE. */
static unsigned long long read_xcr0(void) {
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
}

static void detect_x86(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, NULL) < 1) return;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return;

    const int has_osxsave = (ecx & (1u << 27)) != 0;
    const int has_avx     = (ecx & (1u << 28)) != 0;
    const int has_fma     = (ecx & (1u << 12)) != 0;
    if (!has_osxsave) return; /* no OS-enabled extended state at all */

    const unsigned long long xcr0 = read_xcr0();
    /* bit 1 XMM, bit 2 YMM */
    const int ymm_saved = (xcr0 & 0x6ULL) == 0x6ULL;
    /* additionally bit 5 opmask, bit 6 ZMM_Hi256, bit 7 Hi16_ZMM */
    const int zmm_saved = (xcr0 & 0xE6ULL) == 0xE6ULL;

    if (__get_cpuid_max(0, NULL) < 7) return;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return;

    const int has_avx2     = (ebx & (1u << 5)) != 0;
    const int has_avx512f  = (ebx & (1u << 16)) != 0;
    const int has_avx512vl = (ebx & (1u << 31)) != 0;

    isa_supported[CPU_ISA_AVX2] = has_avx && has_avx2 && has_fma && ymm_saved;
    isa_supported[CPU_ISA_AVX512] =
        isa_supported[CPU_ISA_AVX2] && has_avx512f && has_avx512vl && zmm_saved;
}
#endif

/* Read the DRANZER_CPU_ISA cap. Unset or unparseable leaves the cap alone -
 * a typo must not silently disable SIMD, so it is reported instead. */
static void apply_environment_cap(const char **note) {
    const char *requested = getenv("DRANZER_CPU_ISA");
    if (!requested || requested[0] == '\0') return;

    cpu_isa_t parsed;
    if (cpu_isa_from_name(requested, &parsed) != 0) {
        config_status = CPU_FEATURES_CONFIG_INVALID_ENV;
        snprintf(invalid_environment_value, sizeof(invalid_environment_value),
                 "%s", requested);
        return;
    }
    max_isa = parsed;
    *note = "capped by DRANZER_CPU_ISA";
}

static void build_summary(const char *note) {
    char available[96];
    size_t used = 0;
    available[0] = '\0';
    for (int i = CPU_ISA_BASELINE + 1; i < CPU_ISA_COUNT; i++) {
        if (!isa_supported[i]) continue;
        int written = snprintf(available + used, sizeof(available) - used,
                               "%s%s", used ? ", " : "", isa_names[i]);
        if (written <= 0 || (size_t)written >= sizeof(available) - used) break;
        used += (size_t)written;
    }
    if (used == 0) snprintf(available, sizeof(available), "none");

    if (note) {
        snprintf(summary, sizeof(summary), "%s (detected: %s; %s)",
                 cpu_isa_name(cpu_isa_best()), available, note);
    } else {
        snprintf(summary, sizeof(summary), "%s (detected: %s)",
                 cpu_isa_name(cpu_isa_best()), available);
    }
}

static void detect_once(void) {
    if (detected) return;

    memset(isa_supported, 0, sizeof(isa_supported));
    isa_supported[CPU_ISA_BASELINE] = 1;

#if defined(DRANZER_ARCH_X86_64)
    detect_x86();
#elif defined(DRANZER_ARCH_AARCH64)
    /* Advanced SIMD is mandatory in ARMv8-A, so there is nothing to probe:
     * if this translation unit compiled for aarch64, NEON is present. 32-bit
     * ARM, where NEON is optional and needs getauxval(AT_HWCAP), is not
     * covered - it falls through to baseline. */
    isa_supported[CPU_ISA_NEON] = 1;
#endif

    const char *note = NULL;
    if (max_isa == CPU_ISA_COUNT) {
        max_isa = CPU_ISA_COUNT - 1; /* no cap */
        apply_environment_cap(&note);
    } else {
        note = "capped by cpu_features_set_max_isa";
    }

    detected = 1; /* set before build_summary: it queries through the API */
    build_summary(note);
}

/* An ISA is usable when the hardware supports it and the cap allows it.
 *
 * The cap is an ordinal comparison against the enum order, which is a
 * within-architecture ordering: on x86-64 it reads as baseline < avx2 <
 * avx512, which is what a cap is for. Naming another architecture's ISA is
 * harmless rather than meaningful - only one architecture's ISAs are ever
 * supported in a given binary, so an x86 build capped to "neon" is simply
 * uncapped. cpu_features_clear_max_isa() is the portable way to say that. */
int cpu_isa_available(cpu_isa_t isa) {
    if ((int)isa < 0 || (int)isa >= CPU_ISA_COUNT) return 0;
    detect_once();
    if (isa == CPU_ISA_BASELINE) return 1;
    if ((int)isa > (int)max_isa) return 0;
    return isa_supported[isa];
}

cpu_isa_t cpu_isa_best(void) {
    detect_once();
    for (int i = CPU_ISA_COUNT - 1; i > CPU_ISA_BASELINE; i--) {
        if (cpu_isa_available((cpu_isa_t)i)) return (cpu_isa_t)i;
    }
    return CPU_ISA_BASELINE;
}

const char *cpu_isa_name(cpu_isa_t isa) {
    if ((int)isa < 0 || (int)isa >= CPU_ISA_COUNT) return "unknown";
    return isa_names[isa];
}

int cpu_isa_from_name(const char *name, cpu_isa_t *isa_out) {
    if (!name || !isa_out) return -1;
    for (int i = 0; i < CPU_ISA_COUNT; i++) {
        if (strcmp(name, isa_names[i]) == 0) {
            *isa_out = (cpu_isa_t)i;
            return 0;
        }
    }
    return -1;
}

const char *cpu_features_summary(void) {
    detect_once();
    return summary;
}

cpu_features_config_status_t cpu_features_config_status(void) {
    detect_once();
    return config_status;
}

const char *cpu_features_invalid_environment_value(void) {
    detect_once();
    return config_status == CPU_FEATURES_CONFIG_INVALID_ENV
               ? invalid_environment_value : NULL;
}

void cpu_features_detect(void) {
    detect_once();
}

int cpu_features_set_max_isa(cpu_isa_t isa) {
    if ((int)isa < 0 || (int)isa >= CPU_ISA_COUNT) return -1;
    detect_once();
    max_isa = isa;
    build_summary("capped by cpu_features_set_max_isa");
    return 0;
}

cpu_isa_t cpu_features_max_isa(void) {
    detect_once();
    return max_isa;
}

void cpu_features_clear_max_isa(void) {
    detect_once();
    max_isa = CPU_ISA_COUNT - 1;
    build_summary(NULL);
}
