/*
 * The "threads at all?" decision. See core/parallel.h for the contract and
 * docs/threading.md for the measurements the threshold was chosen from.
 */

#include "core/parallel.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* Work below which a parallel region costs more than it saves.
 *
 * Bracketed by measurement rather than derived, from two directions:
 *
 *   - `bench_parallel.out` times the model's own matmul shapes with forking
 *     forced on and forced off. The largest that loses does 4160 multiply-adds
 *     (1x16x260, 0.21x to 0.25x over four runs - forking it costs four to five
 *     times what computing it does); the smallest that wins does 64000
 *     (1x64x1000, 1.16x to 1.37x).
 *   - Whole-model runs bracket it tighter from above. Serializing the decode
 *     head loop at the small tier, estimated at 13824, cost 1.20x, 1.19x and
 *     1.45x on sliding decode across three sessions, so the threshold has to
 *     admit that loop.
 *
 * That leaves (4160, 13824]. 2^13 is near its geometric middle and classifies
 * every measured configuration correctly. The bracket is narrow enough to be
 * worth re-deriving on very different hardware, and the setter below changes
 * it without a rebuild.
 *
 * The two mistakes are not symmetric, which is why the bracket's lower end is
 * respected strictly: too high forgoes the 1.2x-1.6x forking a large shape is
 * worth, while too low pays a fixed ~0.5 us on a shape that takes 0.3 us to
 * compute. Erring toward not forking costs a fraction; erring toward forking
 * costs a multiple. */
#ifndef DRANZER_PARALLEL_MIN_WORK
#define DRANZER_PARALLEL_MIN_WORK ((size_t)1 << 13)
#endif

/* Process-wide, written once at startup or by the benchmark between
 * measurements, and only read - see the threading note in core/parallel.h. */
static size_t min_work = DRANZER_PARALLEL_MIN_WORK;

int parallel_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int parallel_should_fork(size_t chunks, size_t work) {
#ifdef _OPENMP
    /* Order matters for cost, not correctness: the two integer comparisons
     * come before the OpenMP call so the common small-shape case answers
     * without touching the runtime at all. */
    if (chunks < 2) return 0;
    if (work < min_work) return 0;
    return omp_get_max_threads() > 1;
#else
    (void)chunks;
    (void)work;
    return 0;
#endif
}

size_t parallel_min_work(void) {
    return min_work;
}

int parallel_set_min_work(size_t work) {
    if (work == 0) return -1;
    min_work = work;
    return 0;
}
