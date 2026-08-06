# CPU threading

[← Back to README](../README.md)

DRANZER's CPU parallelism is optional OpenMP, enabled with `make OMP=1`. Every parallel loop splits
work that is already disjoint — blocks of an output matrix, rows of a gradient, attention heads — so
threads never reduce into each other's memory and an `OMP=1` build agrees bit-for-bit with a serial
one.

This document is about the part that is not free: **entering a parallel region costs about half a
microsecond, and this project issues a lot of very small matmuls.** It describes the cutoff that
decides whether to enter one at all, the measurements the cutoff's threshold comes from, and the
persistent worker pool that was prototyped, measured, and rejected in its favour.

The policy lives in `src/core/parallel.c` behind `src/include/core/parallel.h`. The measurement tool
is `src/tools/bench_parallel.c`. Every OpenMP loop in `src/core/matmul.c`, `src/core/matmul_x86.c`,
`src/core/matmul_arm.c`, and `src/core/transformer.c` goes through it.

## The problem

A transformer forward pass is not one big matrix multiply. It is dozens of small ones per layer per
token, and at this project's model sizes many of them are genuinely tiny — a single-token decode
projection at the tiny tier does 256 multiply-adds and takes about 60 nanoseconds.

Measured on the reference machine (GCC 11.4, libgomp, two threads):

| Dispatch | Cost |
|---|---:|
| No pragma at all | 0.0013–0.0025 µs |
| `#pragma omp parallel for` | 0.475–1.29 µs |
| `#pragma omp parallel for if(0)` | 0.367–0.492 µs |
| Persistent spinning worker pool | 0.073–0.337 µs |

Against a 60 ns kernel call, a 0.5 µs region entry is not overhead — it is the whole cost, eight
times over. Worse, the shipped 256-element tile means most decode and prefill shapes decompose into
one or two blocks, so the region is entered, the team is woken, and one thread does everything
anyway.

## The cutoff

`parallel_should_fork(chunks, work)` answers one question: enter a region, or not? It says no when

- there are fewer than two chunks to hand out — nothing to distribute at any size,
- the loop performs less than `parallel_min_work()` multiply-adds, or
- the team would be a single thread, which includes every build without OpenMP.

Call sites use it through `DRANZER_PARALLEL_FOR(count, work, index, body)`, which is a macro for a
reason given below.

`work` is a cost estimate in multiply-adds: `m*k*n` for a matmul, and for the attention head loops
the equivalent product plus a softmax term. It is deliberately crude. It only has to land on the
right side of one threshold.

### Why the unit is the same for matmul and attention

A multiply-add inside a vectorized AVX-512 matmul and one inside attention's scalar strided score
loop ought to cost very different amounts. Near the threshold they do not: 0.145 ns against
0.156 ns on the reference machine. A matmul that small is waiting on memory rather than on
arithmetic, so vector width buys it nothing, and the two kinds of loop can share one unit and one
threshold with no conversion factor. That was measured, not assumed — the conversion factor was
written first and then deleted when it measured as 1.

### Why softmax is counted separately

Attention's head loops are not all multiply-adds. Each row runs a softmax, and an exponential is not
an FMA: one softmax element measured 3.4 ns against 0.155 ns for a multiply-add beside it, so
`DRANZER_PARALLEL_SOFTMAX_WORK` prices it at 22.

This is not a detail. Left out, the estimate for a single-token decode head loop undercounts by 40%,
which put the small tier on the wrong side of the threshold and made sliding decode there 1.19x–1.45x
*slower* across three sessions. The constant exists because that regression showed up in whole-model
measurement after the isolated matmul measurements had already looked clean.

The backward head loop runs `softmax_backward()`, which is a dot product and a subtraction with no
exponential, and so carries no softmax term.

## Why an explicit branch and not OpenMP's `if` clause

OpenMP has a clause for exactly this: `#pragma omp parallel for if(condition)`. It does not work,
for values of "work" that matter here.

`if(0)` measured 0.367–0.492 µs against 0.475–1.29 µs for `if(1)` and 0.0013 µs for no pragma —
roughly 38–77% of a full region entry, and hundreds of times a plain loop. libgomp still builds a
team of one and calls the outlined function; the clause skips the *threads*, not the *region*. An
explicit `if`/`else` around the pragma is the only construct that actually skips it.

That is why `DRANZER_PARALLEL_FOR` is a macro. The branch needs the loop written twice, once inside
a region and once outside, and a macro is what stops the two copies from drifting apart. Loop bodies
long enough to make that awkward are lifted into a static function that the macro then calls — which
is what `core/transformer.c` does with its three attention head loops.

## Why not a persistent worker pool

The obvious alternative is to stop entering regions repeatedly: keep a pool of threads alive across
calls and dispatch work to them. `tools/bench_parallel.c` contains a prototype of the cheapest
version of that idea — workers that spin on an atomic generation counter and never sleep, so a
dispatch is one release store and one acquire poll, with no runtime, no scheduling policy, and no
dynamic work distribution. It is a lower bound on what any persistent worker strategy could cost.

It measured **0.073–0.337 µs per dispatch against 0.475–1.29 µs for a region**: real, and a factor
of 2 to 15 depending on how contended the machine was.

It was still rejected, for three reasons that the measurement makes concrete:

1. **It solves the smaller half of the problem.** A pool makes dispatch cheaper. The cutoff makes it
   free. At the shapes where dispatch cost actually dominates — a 60 ns matmul — even 0.073 µs is
   more than the entire computation, so the right answer there is not a cheaper thread but no
   thread.
2. **Above the cutoff, there is nothing left to win.** At every shape this project issues where
   forking pays at all, region entry is under 2% of the call. A pool would be optimizing a rounding
   error.
3. **Spinning workers are not free to have around.** This was measured accidentally and is the most
   useful thing the prototype produced: the first version of the tool left the pool alive during the
   crossover sweep that follows it, and its spinning workers held one of two cores for the entire
   run. Fork/serial ratios for the *same* shape swung between 0.86x and 2.62x across sessions.
   Stopping the pool first made them repeatable to within a few percent. A pool that spins competes
   with the compute it exists to accelerate, and a pool that sleeps has given back the latency it
   was built for.

So the answer to "replace repeated OpenMP entry with a persistent worker strategy" is: the repeated
entry was worth removing, and not entering is strictly better than entering more cheaply.

## Choosing the threshold

`DRANZER_PARALLEL_MIN_WORK` is 2^13 = 8192 multiply-adds, bracketed from two directions.

**From below, by isolated shapes.** `bench_parallel.out` times each of the model's own matmul shapes
twice on the same buffers — once with forking forced on, once forced off — so kernel, tile, data and
cache state are all held constant and the ratio is the value of forking and nothing else. Of the
shapes that can fork at all:

| Shape | Work | Chunks | Forked vs serial | Verdict |
|---|---:|---:|---:|---|
| 1×16×260 (tiny output head) | 4160 | 2 | 0.21x–0.25x | forking costs 4–5x |
| 1×64×1000 (small output head) | 64000 | 4 | 1.16x–1.37x | smallest measured win |
| 1×256×1024 (medium decode FFN) | 262144 | 4 | 1.38x–1.56x | win |
| 1×256×4000 (medium output head) | 1024000 | 16 | 1.58x–1.61x | win |
| 128×256×1024 (medium prefill FFN) | 33554432 | 4 | 1.31x–2.23x | win |

**From above, by whole-model runs.** Serializing the small tier's decode head loop, estimated at
13824, cost 1.19x–1.45x on sliding decode. The threshold has to admit it.

That leaves the interval (4160, 13824]. 2^13 sits near its geometric middle and classifies every
measured configuration correctly.

The two mistakes are not symmetric, which is why the lower end is respected strictly. Setting the
threshold too high forgoes the 1.2x–1.6x that forking a large shape is worth. Setting it too low
pays a fixed ~0.5 µs on a shape that takes 0.3 µs to compute. Erring toward not forking costs a
fraction; erring toward forking costs a multiple.

`test_parallel.c` pins these verdicts, so moving the constant fails a test rather than drifting
silently.

## Results do not depend on any of this

The cutoff decides whether threads are used. It cannot decide what the answer is.

Every parallel loop distributes disjoint work: blocks own disjoint regions of `C`, backward rows own
disjoint rows of the gradient, heads own disjoint slices of `dQ`/`dK`/`dV`. No thread reduces into
another's output, so the forked and serial paths sum identical products in identical order.
`test_parallel.c` asserts that as **bit-identity**, not a tolerance, over every available kernel at
three tile sizes across eight shapes, and separately for both backward functions run twice into a
non-zero destination so an accumulation race would show. It also reports how many shape/tile pairs
actually produced a region, so the check cannot pass vacuously on a machine where nothing forked.

The flattening this required is worth noting: the blocked kernels' `collapse(2)` nest over `(ii, jj)`
became a single loop over a flat block index. A static schedule distributes exactly the same
iteration space in exactly the same order, so no thread's share changed; one loop is simply what
lets the same body serve both paths.

The default build is serial, so it is the one that must not regress. Measured before and after the
change, ABBA-interleaved, clang, small and tiny tiers: 0.92x–1.05x on every metric, which is this
machine's noise.

## What the measurements changed

```text
build:    gcc 11.4.0 -I include/ -I ../libs/include/ -O3 -ffast-math -fopenmp -lm
os:       Linux 5.15.167.4-microsoft-standard-WSL2 x86_64
cpu:      11th Gen Intel(R) Core(TM) i5-11320H @ 3.20GHz, 2 online cores (1 physical, SMT)
threads:  2
method:   bench.out --cpu-only --quick, two binaries differing only in the cutoff,
          ABBA-interleaved, best of N rounds each
```

Ratios below are old/new — above 1.00x means the cutoff made it faster.

| Metric | tiny | small | medium |
|---|---:|---:|---:|
| Full-context inference | 1.22x | 1.02x | 1.03x |
| Prompt prefill | **3.42x** | **1.25x–1.45x** | 1.04x |
| Growing-cache decode | **2.29x** | **1.27x** | 0.97x |
| Full-ring sliding decode | **2.29x** | **1.11x–1.22x** | 1.03x |
| Training step | 1.15x | 0.96x–1.01x | 1.00x |

The shape of that table is the result. The win is concentrated in exactly the places the analysis
predicted — small models, and the decode and prefill paths where per-call work is smallest — and the
medium tier is flat because every shape there is far above the threshold and nothing changed for it.
The medium column is the control, and it reading 0.97x–1.04x is what makes the tiny and small
columns believable.

## Reproducing the measurements

```bash
cd src
make bench-parallel CC=gcc OMP=1
OMP_NUM_THREADS=2 ./bench_parallel.out --rounds 7
```

Clang needs `libomp-dev` for `OMP=1`; without it, build with GCC. A build without OpenMP produces a
binary that says there is nothing to measure rather than one that fails.

The tool prints the dispatch table, then the per-shape crossover sweep, and appends every row to
`parallel_results_v1.csv` with the full build/OS/CPU/SIMD/thread provenance every results file in
this project carries.

### Reading the results

**Watch the `cannot fork` rows.** Shapes whose blocked decomposition yields fewer than two blocks
run identical code on both sides of the comparison, so whatever ratio they report is the machine's
noise and nothing else. The tool collects their worst excursion and prints it as an explicit noise
floor. On the reference machine — two hyperthreads on one physical core, under WSL2 — that band ran
0.81x–1.14x. A forkable row inside the band has not measured anything.

This is not a formality. Before the noise floor was reported, three sessions of the same sweep
disagreed on whether `1×256×4000` should fork, giving 2.41x, 1.18x and 0.85x. Two causes turned out
to be present at once: the prototype pool spinning through the sweep, and a genuinely contended
machine. The first was a bug and was fixed; the second is why the band is printed.

Whole-model numbers come from the ordinary benchmark with two binaries that differ only in the
change under test, interleaved ABBA within one session, best of each side. See
[CPU matmul kernels](matmul.md#reading-the-results) for the rest of the measurement discipline this
project uses — it applies here unchanged.

## Limitations

- **Measured on two hyperthreads of one physical core.** That is the worst realistic case for
  parallel speedup and the best case for the cutoff, since a second thread on the same core adds no
  execution resources. The cutoff's *threshold* should not move much — it is a ratio of two
  quantities that scale together, so a faster CPU shortens the compute and the region entry alike —
  but the whole-model speedups above are specific to this machine and a machine with more real cores
  should re-run the sweep.
- **GCC and libgomp only.** Clang's `libomp` was not installed on the reference machine, so every
  number here is libgomp's. The `if(0)` finding in particular is a statement about one OpenMP
  runtime; another may implement the clause as a real branch, which would make the macro's
  duplication unnecessary there but not wrong.
- **The work estimates are per-site formulas, not instrumentation.** They are algebraic guesses at
  what a loop costs, calibrated once. A future kernel with a very different cost profile —
  quantized, packed, or with a transcendental in its inner loop — would need its estimate revisited
  rather than inheriting `m*k*n`.
- **Nested parallelism is not considered.** Every region here is a leaf, and the head loops are not
  entered from inside another region. Introducing an outer region would make these cutoffs interact
  in ways nothing in this document measures.
- **The threshold is one number for all loops.** Two call sites measured close enough to share it,
  which is the finding above, but that is an empirical fact about this project's shapes on this
  machine and not a law.
