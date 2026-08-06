#ifndef PARALLEL_H
#define PARALLEL_H

#include <stddef.h>

/* When a parallel region is worth entering, and the one loop form that asks.
 *
 * Every OpenMP loop in this project sits inside a hot function that is called
 * thousands of times per token. Entering a region is not free: on the CPU this
 * was measured on it costs about 0.6 us with two threads and about 0.4 us with
 * one, which is more than an entire small-model matmul takes to compute. A
 * region whose loop cannot be spread across the team - and with the measured
 * 256-element tile, most decode and prefill shapes produce one or two blocks -
 * pays that in full and gets nothing back.
 *
 * So the policy here is not "how many threads" but "threads at all". See
 * docs/threading.md for the measurements, including why a persistent worker
 * pool was measured and rejected in favour of this.
 *
 * Nothing in this header is synchronized. Read it before compute starts, not
 * from inside a parallel region, exactly like the setters in core/matmul.h. */

/* Whether a parallel loop of `chunks` independent iterations performing `work`
 * float multiply-adds in total should run inside an OpenMP region.
 *
 * Returns 0 in a build without OpenMP, when the team would be one thread, when
 * there are fewer than two chunks to hand out, or when `work` is below
 * parallel_min_work(). Costs a few nanoseconds - it reads OpenMP's thread
 * count (measured at 2 ns) and compares two integers.
 *
 * `work` is a cost estimate, not a promise: callers pass m*k*n for a matmul or
 * the equivalent product for an attention head loop. It only has to be
 * accurate enough to sit on the right side of the threshold (docs/threading.md).
 *
 * The unit is one multiply-add as the model actually issues them. A vectorized
 * matmul near the threshold and attention's scalar strided score loop measured
 * within 10% of each other per multiply-add on the reference machine - 0.145 ns
 * against 0.156 ns - because a shape that small is waiting on memory rather
 * than arithmetic either way. So the two kinds of loop share one unit, and one
 * threshold, without a conversion factor. */
int parallel_should_fork(size_t chunks, size_t work);

/* What one softmax element costs in those same multiply-add units.
 *
 * Attention's head loops are not all multiply-adds: each row runs a softmax,
 * and an exponential is not an FMA. One softmax element measured 3.4 ns
 * against 0.155 ns for a multiply-add in the loop beside it, so it is worth
 * about 22 of them. Left out, the estimate for a decode head loop undercounts
 * by 40% - enough to put the small tier on the wrong side of the threshold,
 * which is how this constant came to be measured at all.
 *
 * softmax_backward() is a dot product and a subtraction with no exponential,
 * so the backward head loop does not use this. */
#define DRANZER_PARALLEL_SOFTMAX_WORK 22

/* The OpenMP team size a region would get, 1 in a build without OpenMP. */
int parallel_max_threads(void);

/* The work threshold parallel_should_fork() compares against, and a setter so
 * the benchmark can sweep it without a rebuild. Returns 0, or -1 for a zero
 * threshold, leaving the previous setting untouched. Setting it to SIZE_MAX
 * disables forking entirely, which is how the benchmark measures the serial
 * side of the comparison in the same process. */
size_t parallel_min_work(void);
int parallel_set_min_work(size_t work);

/* A parallel loop over [0, COUNT) that enters a region only when the above
 * says it pays. INDEX is declared by the macro; BODY is everything after it
 * and may contain commas.
 *
 * This is a macro, and it expands the loop body twice, for one reason: OpenMP's
 * own `if` clause does not avoid the cost. `#pragma omp parallel for if(0)`
 * still measured 0.38 us against 0.52 us for `if(1)` and 0.004 us for no pragma
 * at all - libgomp builds a team of one and calls the outlined function anyway,
 * so the clause skips the threads but not the region. An explicit branch is the
 * only construct that actually skips it, and a macro is what keeps the two
 * copies of the loop from drifting apart.
 *
 * Bodies long enough to make the duplication awkward should be a static
 * function that this calls - which is how core/transformer.c uses it. */
#ifdef _OPENMP
#define DRANZER_PARALLEL_FOR(COUNT, WORK, INDEX, ...)                        \
    do {                                                                     \
        const size_t dranzer_pf_count = (COUNT);                             \
        if (parallel_should_fork(dranzer_pf_count, (WORK))) {                \
            _Pragma("omp parallel for schedule(static)")                     \
            for (size_t INDEX = 0; INDEX < dranzer_pf_count; INDEX++) {      \
                __VA_ARGS__                                                  \
            }                                                                \
        } else {                                                             \
            for (size_t INDEX = 0; INDEX < dranzer_pf_count; INDEX++) {      \
                __VA_ARGS__                                                  \
            }                                                                \
        }                                                                    \
    } while (0)
#else
#define DRANZER_PARALLEL_FOR(COUNT, WORK, INDEX, ...)                        \
    do {                                                                     \
        const size_t dranzer_pf_count = (COUNT);                             \
        (void)(WORK);                                                        \
        for (size_t INDEX = 0; INDEX < dranzer_pf_count; INDEX++) {          \
            __VA_ARGS__                                                      \
        }                                                                    \
    } while (0)
#endif

#endif // PARALLEL_H
