# CPU matrix multiplication

[← Back to README](../README.md)

Matrix multiplication is where DRANZER spends most of its time, and it is the only primitive in the
project with more than one implementation. This document describes the kernels, how the default is
chosen, what the choice guarantees about reproducibility, and how to repeat the measurements the
choice was made from.

The policy and the portable kernels live in `src/core/matmul.c` behind `src/include/core/matmul.h`.
The SIMD kernels are in `src/core/matmul_x86.c` and `src/core/matmul_arm.c` behind
`src/include/core/matmul_simd.h`, and the runtime detection that gates them is
`src/core/cpu_features.c`. The measurement tool is `src/tools/bench_matmul.c`, driven by
`src/tools/matmul_sweep.sh`.

## Kernels

All seven kernels compute the same thing — `C (m x n) = A (m x k) @ B (k x n)`, row-major, fully
defining `C` — and differ only in the order they traverse it and the instructions they traverse it
with. `A`, `B`, and `C` must not overlap; the parameters are `restrict`-qualified, which is
load-bearing rather than decorative (see
[What the measurements changed](#what-the-measurements-changed)).

| Kernel | Traversal | Requires | Intended for |
|---|---|---|---|
| `scalar` | Unblocked `i/j/l`, one dot product at a time | — | The reference. Never selected automatically |
| `rowwise` | Unblocked `i/l/j`, contiguous stores | — | Cache-resident shapes: single-token decode, small models |
| `tiled` | Cache-blocked `i/l/j` over `tile x tile` blocks | — | Larger shapes where a block of `B` is reused |
| `tiled_mr4` | Blocked, four rows of `A` per pass | — | The portable default, and the fallback for all of the below |
| `avx2_mr4` | `tiled_mr4` with 8-wide vector FMAs | AVX2 + FMA | x86-64 |
| `avx512_mr4` | `tiled_mr4` with 16-wide vector FMAs and masked tails | AVX-512F + VL | x86-64 |
| `neon_mr4` | `tiled_mr4` with 4-wide vector FMAs | NEON (baseline on AArch64) | AArch64 |

`tiled_mr4` keeps four accumulator rows in flight so each loaded `B` element feeds four independent
chains: the same number of loads does four times the arithmetic, and the four chains hide FMA
latency. Row counts that are not a multiple of four finish through the one-row body, so every `C`
element accumulates `l` in the same increasing order `tiled` uses.

The three SIMD kernels are that same algorithm with the innermost loop over `j` issued as vector
FMAs. **Vectorizing along `j` rather than `k` is what keeps them comparable to the reference**: each
`C` element still accumulates its products over `l` in increasing order, so the only numerical
difference from the portable kernel is FMA contraction, never a reassociated sum. A kernel that
vectorized along `k` would need a horizontal reduction and a different error story.

The scalar kernel exists to be correct, not fast. It is what `model->use_scalar_matmul` and
`bench.out --scalar` select, what every candidate is checked against in
`test_matmul_kernels.c`, and what `test_scalar_reference.c` compares whole-model and cached-decode
logits against.

## How the default is chosen

`matmul_select()` resolves `MATMUL_KERNEL_AUTO` to the widest vector kernel this CPU can execute,
and to `tiled_mr4` when there is none:

```text
avx512_mr4  →  avx2_mr4  →  neon_mr4  →  tiled_mr4
```

It does **not** split by shape. Every shape gets the same kernel, which is a measured outcome rather
than an unimplemented feature — see [What the measurements changed](#what-the-measurements-changed)
for the two shape splits that were tried and rejected, and
[Runtime dispatch](#runtime-dispatch-avx2-avx-512-neon) for why the SIMD substitution did not need
one either.

Overrides are available for measurement and debugging:

```c
matmul_set_kernel(MATMUL_KERNEL_TILED);      /* pin one kernel process-wide */
matmul_set_tile_size(128);                   /* change the blocked-kernel tile */
cpu_features_set_max_isa(CPU_ISA_BASELINE);  /* pretend the SIMD is not there */
```

and from the benchmark: `./bench.out --kernel tiled --tile 128`. `DRANZER_CPU_ISA=baseline|avx2|…`
applies the same instruction-set cap from the environment, which is how one binary can be measured
along each dispatch path without recompiling. The compile-time `-DDRANZER_MATMUL_BLOCK_SIZE=N` still
sets the starting tile. None of these setters is synchronized: set them before compute starts, not
from inside a parallel region.

## Runtime dispatch (AVX2, AVX-512, NEON)

The build deliberately does not pass `-march=native` (see `src/Makefile`): a binary tuned for the
build machine raises SIGILL on any CPU that lacks an instruction it used. The SIMD kernels get their
instructions a different way — each vector function carries a `__attribute__((target(...)))` — so
the wide instructions exist in the binary but only inside functions nothing branches to unless
`cpu_isa_available()` says so. The same executable therefore uses AVX-512 on a Tiger Lake laptop,
AVX2 on an older x86-64, and portable C on a machine with neither.

Two properties of the detection are worth stating because getting either wrong produces a crash
rather than a slowdown:

- **CPUID's feature bit is not permission to execute.** AVX and AVX-512 add register state the OS
  must agree to preserve across context switches. `cpu_features.c` checks `XCR0` via `XGETBV` as
  well as the feature bit, because an OS that has not enabled that state leaves the CPUID bit set
  while the first instruction faults.
- **A pinned kernel that cannot run is a fallback, not an error.** `--kernel avx512_mr4`,
  a config file, and `matmul_set_kernel()` all travel to machines the person who wrote them never
  saw. `matmul_run()` substitutes `tiled_mr4` for any unavailable kernel, and every kernel *name*
  parses on every architecture for the same reason.

The OpenMP loop sits in a plain untargeted function that calls the targeted block through a pointer.
Keeping the parallel region out of a target-attributed function sidesteps the outlining-versus-
target-attribute interaction between compilers entirely, and the indirect call is amortized over a
whole `tile x tile` output block.

## Reproducibility

Within a process the policy is a pure function of `(m, k, n)`, which is what keeps training runs
reproducible: the same call shape always takes the same kernel, so a run repeats its own
accumulation order exactly. The instruction set is resolved once from the hardware, not per call, so
runtime dispatch does not weaken this. `test_matmul_kernels.c` asserts it directly.

The exactness contract is what it always was — bit-for-bit on the same executable and execution
backend — but **runtime dispatch makes the CPU's instruction set part of "execution backend."**
Before, one executable implied one kernel; now the same executable selects `avx512_mr4` on a machine
that has it and `tiled_mr4` on one that does not, and those agree to within FMA contraction rather
than bit-for-bit. Concretely:

- Exact resume, checkpoint round-trips, and `--resume` remain bit-identical on one machine, which is
  what that contract has always covered.
- Comparing artifacts *across* machines now needs the ISA pinned on both:
  `DRANZER_CPU_ISA=baseline` is the portable common denominator, and benchmark rows record the
  detected ISA in a `simd` provenance column so a result file stays interpretable either way.

Different kernels sum the same products in different orders, so a build with a different tile size,
a different compiler, or a different kernel override remains equivalent within float tolerance, not
bit-identical. Nothing in the selection policy varies on thread count or wall-clock timing.

OpenMP parallelism does not change results either: blocks own disjoint regions of `C`, so there is
no cross-thread reduction and an `OMP=1` build agrees bit-for-bit with a serial one.

## Reproducing the measurements

```bash
cd src
tools/matmul_sweep.sh                        # gcc + clang, all tiers
tools/matmul_sweep.sh --tiers medium --repeats 5
tools/matmul_sweep.sh --compilers clang --omp
```

The script cleanly rebuilds `bench.out` per compiler, sweeps every kernel and tile candidate over
the shapes each model tier actually issues, and prints the fastest candidate per shape. Every row it
summarises is also appended to `matmul_results_v3.csv` with the exact build command, compiler,
OS, CPU model, core count, OpenMP version, and thread count.

To measure one configuration without the script:

```bash
./bench.out --matmul-only --sweep --tier small --repeats 5
./bench.out --matmul-only --tier small          # just scalar vs the shipped default
./bench.out --tier small --kernel tiled --tile 128   # whole-model effect of a kernel
```

The shapes are taken from live call sites in `src/core/transformer.c`: `decode_*` from
`model_forward_token()`, `prefill_*` from `model_forward()`, and `training_ffn_down` from the FFN
down-projection that dominates the backward pass.

### Reading the results

Candidates are **ranked by their fastest round, not the median.** Contention and frequency scaling
can only ever make a round slower, so the minimum is the closest available estimate of what a kernel
costs when it has the machine. The median is reported beside it as a noise indicator: when the two
are far apart, the session was not quiet enough to retune from. The sweep summary prints this spread
as a `noise` column — treat anything above roughly 10% as advisory only.

Two further guards are built into the measurement, both from mistakes made while doing this work:

- **The first round of each candidate is discarded.** A cold first measurement on this project's
  own hardware overstated a kernel's cost by more than 10x — enough to invert a comparison.
- **Rounds are interleaved across candidates** rather than run candidate by candidate, so a slow
  window lands on every candidate instead of penalising whichever one was being measured when
  another process woke up.

Results are per-machine. `matmul_results_v3.csv` is gitignored for that reason; the provenance
columns exist so a local history stays interpretable, not so numbers can be compared across
machines.

## What the measurements changed

The selection sweep behind the current defaults:

```text
build:    {gcc 11.4.0, clang 14.0.0} -I include/ -I ../libs/include/ -O3 -ffast-math -lm
os:       Linux 5.15.167.4-microsoft-standard-WSL2 x86_64
cpu:      11th Gen Intel(R) Core(TM) i5-11320H @ 3.20GHz, 2 online cores
threads:  1 (no OpenMP)
sweep:    tools/matmul_sweep.sh --repeats 5   (all tiers, both compilers)
```

Four candidate kernels at five tile sizes, over six shapes per tier and three tiers, under two
compilers: 36 (compiler, tier, shape) combinations. Three things came out of it.

**`restrict` was worth 3-4x on its own.** The multi-row kernel was originally written without it.
Clang then had to assume each accumulator store could alias the next `B` load and gave up on
vectorizing: on the medium prefill and training shapes `tiled_mr4` measured *slower* than the plain
tiled kernel by up to 4x. GCC vectorized it either way. Adding `restrict` to the kernel parameters
made the two compilers agree.

**The default kernel and tile changed, and are now measured rather than assumed.** The project
previously shipped one blocked kernel at a 64 tile. `tiled_mr4` at tile 256 is the best single
choice across both compilers: it is faster than the scalar reference on all 36 combinations —
1.46x at worst, 4.47x geometric mean, 9.20x at best — and is never more than 1.40x off the fastest
candidate for any individual shape. Against the previously shipped `tiled`/64 it is 1.13x faster
in geometric mean across the sweep.

The one shape where it is meaningfully behind is GCC on `prefill_ffn_up` at 128x256x1024, where the
plain tiled kernel is 1.40x faster. That is what `--kernel tiled` and `matmul_set_kernel()` are for.

**Two shape splits were rejected.** Both were worth measuring and neither survived:

| Candidate rule | Result |
|---|---|
| single-row (`m < 4`) decode uses `rowwise` | Up to 1.26x faster under Clang, up to 1.5x *slower* under GCC. Geometric mean across both: worse (1.14 vs 1.09 relative to per-shape best) |
| wide outputs (`n >= 1024`) use plain `tiled` | ~1% better on average — smaller than the run-to-run spread of the session that produced it |

So `matmul_select()` returns one kernel for every shape. That is a measured outcome, not an
unimplemented feature, and `test_matmul_kernels.c` pins it so a future split has to update this
document deliberately.

### Whole-model effect

Isolated shapes are not the deliverable, so the chosen kernel was also checked end to end with
`bench.out --tier small --kernel ...`, alternating configurations and taking the fastest run of
each. Under GCC every metric improved:

| Metric (GCC, small tier) | tiled/64 → mr4/256 |
|---|---|
| Full-context inference | 2.882 → 2.388 ms/token (1.21x) |
| Prompt prefill | 0.0894 → 0.0683 ms/token (1.31x) |
| Growing-cache decode | 0.1256 → 0.0964 ms/token (1.30x) |
| Full-ring sliding decode | 0.1265 → 0.1034 ms/token (1.22x) |
| Training step | 22.22 → 20.87 ms/step (1.06x) |

**The equivalent Clang runs did not reproduce, and that is reported rather than resolved.** Three
batches on the same machine disagreed on the same comparison — 0.87x, 1.22x, and 0.89x on
inference. The cause is visible in the raw runs: within a single batch, successive runs of the
*same* configuration drifted from 5.36 to 9.74 ms/token, so whichever configuration ran first
looked best. Re-running in drift-cancelling ABBA order narrowed the spread but left the result
inconclusive.

Two honest consequences:

- The Clang half of the kernel choice rests on the isolated sweep, which is built to survive a
  contended machine — interleaved rounds, discarded warm-up, ranking by fastest round — and which
  put `mr4`/256 ahead of `tiled`/64 on every Clang small-tier shape by 1.03x to 1.48x.
- The GCC whole-model table above ran old-configuration-first throughout, which the drift analysis
  says favours the *old* kernel. Those numbers are therefore conservative, not flattering.

If you are evaluating this on your own hardware, re-run both: `tools/matmul_sweep.sh` for the
kernels and `bench.out --tier small --kernel ...` for the model. A two-core laptop under WSL2 with
other work on it — which is what produced everything above — is the environment these guards exist
for. One cold measurement taken during this work overstated a kernel's cost by more than 10x.

## Limitations

- **The NEON kernel is written but unmeasured.** No AArch64 machine was available. It is checked for
  correctness by the same equivalence test wherever it is built, and the preference order places it
  ahead of the portable kernel on the same reasoning as the x86 kernels, but no number in this
  document was produced on ARM. Re-run `tools/matmul_sweep.sh` before trusting it there.
- **The "runs without the instruction set" gate is tested by simulation, not by hardware.** No
  pre-AVX2 machine was available either. `cpu_features_set_max_isa(CPU_ISA_BASELINE)` makes
  detection report exactly what such a CPU would report, so every dispatch decision below it is the
  real code path; what that cannot prove is the absence of a wide instruction on a reachable path,
  which is a property of the `target` attributes rather than of dispatch.
- The kernels are register-blocked but not packed. `A` and `B` are read from their original
  row-major layout, so the innermost loop's `B` access is contiguous but its `A` access is strided.
  A packing pass is the usual next step and is not attempted here.
- The backward-pass kernels (`matmul_backward_input`, `matmul_backward_weight`) are not part of the
  candidate set. They have one implementation each, still portable C, and are unchanged by this
  work — which means training benefits from SIMD only through its forward pass.
- Tiles are square. A rectangular blocking scheme (different extents for `m`, `k`, and `n`) is not
  explored.
