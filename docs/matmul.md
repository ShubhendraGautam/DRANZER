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

`matmul_select()` resolves `MATMUL_KERNEL_AUTO` to the widest vector kernel the CPU can execute:

```text
avx512_mr4  →  avx2_mr4  →  tiled_mr4
```

**That ordering was measured, rejected, and then measured again**, and the history is the more
useful part. While the SIMD kernels held their accumulators in memory, `avx2_mr4` was worth 1.47x
under GCC and 0.92x under Clang, so the policy declined to select it — correctly, on the evidence
then available. Rewriting the kernels to keep accumulators in registers (see
[Accumulators in registers](#accumulators-in-registers)) removed the cause, and both compilers now
agree the width ordering holds.

`neon_mr4` is still not selected, for one reason only: no AArch64 hardware was available to measure
it on. The earlier second reason — that a hand-written kernel cannot beat what a compiler emits for
a baseline instruction set — is precisely the claim the rewrite falsified, so it no longer counts.

It does **not** split by shape. Every shape gets the same kernel — see
[What the measurements changed](#what-the-measurements-changed) for the two shape splits that were
tried and rejected.

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
no cross-thread reduction and an `OMP=1` build agrees bit-for-bit with a serial one. Whether a
kernel enters a parallel region at all is a separate, purely-performance decision described in
[CPU threading](threading.md); it cannot change a result, and `test_parallel.c` asserts that as
bit-identity rather than as a tolerance.

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

## What runtime dispatch changed

The SIMD sweep used the same workflow, the same shapes, and the same machine as the one above:

```text
build:    {gcc 11.4.0, clang 14.0.0} -I include/ -I ../libs/include/ -O3 -ffast-math -lm
os:       Linux 5.15.167.4-microsoft-standard-WSL2 x86_64
cpu:      11th Gen Intel(R) Core(TM) i5-11320H @ 3.20GHz (Tiger Lake), 2 online cores
simd:     avx512 (detected: avx2, avx512)
threads:  OMP_NUM_THREADS=2
sweep:    ./bench.out --matmul-only --sweep --repeats 5     (all tiers, per compiler)
```

Three candidate kernels were added and swept at five tile sizes over the same six shapes per tier
and three tiers, under both compilers — 36 (compiler, tier, shape) combinations, as before.

**AVX-512 wins everywhere, and the compilers agree.**

| `avx512_mr4` vs | worst | geometric mean | best | loses on |
|---|---|---|---|---|
| `tiled_mr4` (both compilers) | 1.07x | **1.83x** | 3.02x | 0 of 36 |
| `tiled_mr4` (Clang only) | 1.10x | 1.83x | 3.02x | 0 of 18 |
| `tiled_mr4` (GCC only) | 1.07x | 1.83x | 3.00x | 0 of 18 |
| `scalar` reference | 1.48x | 8.78x | 30.89x | 0 of 36 |

Two compilers landing on 1.83x independently is the strongest signal in this table: it means the
result is a property of the instruction set, not of one optimizer's mood. Tile 256 stays the right
single choice — never more than 1.11x off the best per-shape tile.

**AVX2 splits the compilers, so it is not the default.**

| `avx2_mr4` vs `tiled_mr4` | worst | geometric mean | best | loses on |
|---|---|---|---|---|
| Clang | 0.72x | **0.90x** | 1.12x | 14 of 18 |
| GCC | 1.07x | **1.35x** | 2.12x | 0 of 18 |

Same source, same shapes, opposite verdicts, and consistent within each compiler across all three
tiers. This is the mirror image of the `rowwise` shape split T10 rejected for helping Clang and
hurting GCC, and it gets the same answer for the same reason: **the project's default compiler is
Clang, and a default must not regress the default build.**

What it is *not* is a failure to vectorize. Disassembling both builds shows packed FMAs in both —
GCC's `block_avx2` and Clang's each emit `vfmadd*ps` over `%ymm`. The gap is in scheduling around
those FMAs, and it was not isolated further; chasing it would mean tuning against one compiler's
scheduler rather than against the hardware. Recorded as an open question rather than resolved.

**The useful generalisation.** Dispatch pays where it unlocks instructions the baseline target
cannot emit at all, and not otherwise:

| Kernel | Outside the baseline target? | Result |
|---|---|---|
| `avx512_mr4` | Yes — the compiler can never emit AVX-512 without `-march` | 1.83x, both compilers |
| `avx2_mr4` | Yes, but the width gain is small enough for scheduling to erase it | Compiler-dependent |
| `neon_mr4` | **No** — NEON is baseline on AArch64, so `tiled_mr4` already compiles to it | No new instruction to offer |

That is why `neon_mr4` ships unselected rather than assumed-good. Selecting an unmeasured kernel
over a measured one, on the theory that hand-written beats the compiler, is exactly the assumption
`avx2_mr4` falsified.

### Whole-model effect of dispatch

Isolated shapes are not the deliverable, so the same binary was run end to end with the instruction
set capped and uncapped — `DRANZER_CPU_ISA=baseline` against the detected `avx512` — six runs each,
interleaved ABBA so run-order drift cancels instead of favouring whichever went first. Clang, small
tier, `--cpu-only`, best of six with the observed spread beside it:

| Metric (Clang, small tier) | `baseline` | `avx512` | Speedup |
|---|---|---|---|
| Full-context inference | 2.642 ms/token (1.11x spread) | 2.054 ms/token (1.07x) | **1.29x** |
| Prompt prefill | 0.0515 ms/token (1.77x) | 0.0410 ms/token (1.36x) | **1.26x** |
| Growing-cache decode | 0.0600 ms/token (1.81x) | 0.0553 ms/token (1.31x) | 1.08x |
| Full-ring sliding decode | 0.0624 ms/token (2.51x) | 0.0566 ms/token (1.29x) | 1.10x |
| Training step | 18.59 ms/step (1.13x) | 18.44 ms/step (1.08x) | *no measurable change* |

Inference is the trustworthy row: the two sets of six runs do not overlap at all (baseline
2.64–2.93, `avx512` 2.05–2.20), so the separation is larger than the session's noise. Prefill agrees
in direction with a wider spread.

**Training shows no measurable change, and that is the design working as documented rather than a
disappointing result.** The backward-pass kernels are not dispatched — `matmul_backward_input` and
`matmul_backward_weight` remain portable C — and the backward pass dominates a training step. The
ranges overlap (baseline 18.59–21.03, `avx512` 18.44–19.93), so the honest reading is "unchanged",
not "1.01x". Making training benefit from SIMD means dispatching the backward kernels, which is
listed below as a limitation and not attempted here.

The gap between 1.83x on isolated matmul shapes and 1.29x on whole-model inference is the ordinary
Amdahl result: attention, softmax, layer norm, and the tokenizer are untouched by any of this.

## The backward kernels

`matmul_backward_input` and `matmul_backward_weight` are not selectable kernels — one
implementation each, no dispatch, no tile — but one of them was carrying a large and entirely
avoidable cost.

```text
backward_input   dA (m x k) += dC (m x n) @ B_transposed
backward_weight  dB (k x n) += A_transposed @ dC (m x n)
```

`backward_input` was always fine: its innermost loop runs over `j`, walking a row of `dC` and a row
of `B` contiguously, which is a dot product a compiler vectorizes on its own.

`backward_weight` was not. Written the obvious way — accumulate a dot product into a register and
store each output element once — it puts `i` innermost, where it strides `A` by `k` and `dC` by `n`
*simultaneously*. Every iteration of the hot loop touched two fresh cache lines to use one float
from each. At 128x1024x256 it cost about 40 ms against roughly 6 ms for a forward matmul of
identical FLOP count.

Putting `j` innermost instead walks `dB` and `dC` contiguously and reduces `A` to one scalar load
per `(l, i)`. Four rows of `dB` are kept in flight for the same reason `tiled_mr4` keeps four rows
of `C`. Measured with `./gpu_latency.out`, which times the CPU backward path beside the GPU one:

| Shape (m×k×n) | before | after | speedup |
|---|---:|---:|---:|
| 1×64×1000 | 229.0 µs | 10.5 µs | **21.8x** |
| 1×256×4000 | 2432.2 | 161.1 | **15.1x** |
| 64×64×64 | 167.0 | 29.6 | 5.6x |
| 64×256×64 | 770.8 | 126.9 | 6.1x |
| 128×256×256 | 9761.8 | 1038.8 | 9.4x |
| 128×256×1024 | 39928.9 | 6499.0 | 6.1x |
| 128×1024×256 | 39697.6 | 4662.3 | 8.5x |

Whole-model, medium tier, `--cpu-only`, old and new binaries interleaved ABBA in one session, best
of six runs each: **1125.3 → 408.6 ms/step, 2.75x.** The two ranges do not overlap (1125–1689
against 409–547), and inference — unchanged code, measured as a control in the same runs — read
77.9 ms/token on both sides, ratio 1.00x.

Two consequences worth stating plainly:

- **This is the largest single speedup in the project's history so far**, and it came from a loop
  order rather than from SIMD or a GPU. It is also nearly twice what dispatching the backward pass
  to the GPU was worth (1.54x), and unlike that, it helps every user.
- **It reassociates the sum.** The old order added a fully-accumulated dot product to `dB` once;
  the new one adds each `i`'s contribution in turn. Results differ in the last bits from builds
  before this change — the same class of difference as switching matmul kernels, covered by the
  gradient checks' tolerances rather than by bit-identity. Within one build it stays deterministic,
  which is what exact resume actually requires.

### Accumulators in registers

The SIMD kernels were first written with `j` innermost, reading `C`, adding, and writing `C` back on
every step of `k`. That produced the one result in this project that a compiler disagreed about:
`avx2_mr4` measured **1.47x** under GCC and **0.92x** under Clang, from identical source. It shipped
unselected for that reason, with the cause recorded as unexplained.

Disassembling both builds explained it. In `fmadd(a, b, load(c))` the loaded value sits in the
addend position, which x86 can encode as an FMA memory operand:

| hot loop | lines | `vfmadd_ps` | with memory operand | `vmovups` |
|---|---:|---:|---:|---:|
| GCC | 18 | 4 | **4 of 4** | 5 |
| Clang | 22 | 4 | **1 of 4** | 8 |

Clang folded one of four and issued three extra loads per iteration. Not scheduling, as had been
assumed — an encoding choice, and the timing gap tracked the instruction count.

The fix was to stop needing the fold. For each vector-wide strip of columns, `C` is now read once
into accumulator registers, the whole `k` range accumulates into them, and they are written back
once; the inner loop carries no `C` traffic at all. Nine of sixteen `ymm` registers are live, so it
does not spill. `B` is walked with stride `n` instead of contiguously, which the existing blocking
already keeps resident.

| geometric mean vs `tiled_mr4` | before | after |
|---|---:|---:|
| `avx2_mr4`, GCC | 1.47x | **1.95x** |
| `avx2_mr4`, Clang | **0.92x** | **2.02x** |
| `avx512_mr4`, GCC | 1.83x | **~3.1x** |
| `avx512_mr4`, Clang | 1.83x | **~3.2x** |

Two things worth taking from this beyond the numbers.

**The same defect was costing the shipped default kernel a further 1.7x.** `avx512_mr4` had the
identical structure and had been selected for months on the strength of beating the portable kernel
1.83x. It was never compared against a *better version of itself*. Between the two rewrites, the
restructured 8-wide kernel beat the un-restructured 16-wide one on five of six shapes — width does
not help a loop whose limit is memory traffic.

**"Measured slower" is a fact about an implementation, not an instruction set.** The policy was
right to decline `avx2_mr4` on the evidence available, and equally right to select it once the cause
was found and removed. The general rule this project keeps relearning: when a result is surprising,
the baseline is the cheap thing to suspect, and a kernel that loses deserves a disassembly before it
deserves retirement.

`neon_mr4` received the same restructure on the structural argument alone, since no AArch64 hardware
was available to measure it. That exposed a separate gap: `core/matmul_arm.c` sits behind
`#ifdef DRANZER_HAVE_NEON`, so on x86 — every developer machine and every CI runner here — its body
had never been compiled by anything. `make arm-check` now type-checks it by cross-compiling with
clang, which needs no AArch64 machine and confirms the kernel emits `fmla`.

### Vector backward kernels

Both backward functions then gained an AVX-512 path, in `core/matmul_x86.c` beside the forward
kernels and gated the same way through `cpu_isa_available()`. The weight kernel is an axpy — the
same four-accumulator shape as `block_avx512` — while the input kernel is a reduction, taking four
rows of `B` at once so one load of `dC` feeds four independent chains that are reduced across lanes
at the end.

They were **AVX-512 only** at first, and that asymmetry with the forward path was justified as the
T11 result applied rather than ignored: `avx2_mr4` measured as a regression under Clang and
`neon_mr4` was never measured; both still ship forward because `--kernel` can select them
explicitly, while the backward functions have no selection mechanism, so an AVX2 version "would be
code nothing could reach."

**Both halves of that argument turned out to be wrong, and CI found it.** An AVX2-only hosted
runner failed the backward-versus-forward cost invariant in `tests/perf/test_perf_invariants.c` at
4.09x against its 3.00x limit. The invariant's own failure message blames cache traversal, which
was the wrong diagnosis: `matmul_select()` ships `avx2_mr4` for the *forward* path, so on a CPU
with AVX2 and no AVX-512 a vectorized forward was being compared against a backward that fell all
the way back to portable C. The ratio was measuring a missing kernel, not a traversal.

The first half of the argument had already been falsified and nobody noticed: the register-resident
rewrite took `avx2_mr4` to 1.95x/2.02x and promoted it in `matmul_select()` wherever AVX-512 is
absent, which is exactly the hardware in question. The second half confused *unselectable by flag*
with *unreachable* — hardware selects these, and an AVX2-only CPU reaches an AVX2 version on every
single call.

Both backward functions now have an AVX2 rung, and dispatch takes the widest the CPU offers. The
kernels are the AVX-512 ones at eight lanes instead of sixteen, with a scalar tail (AVX2 has no
mask-predicated load/store) and a hand-written horizontal fold (no `_mm512_reduce_add_ps`).

Portable against AVX2, ABBA-interleaved, best of seven rounds, `DRANZER_CPU_ISA` capped either
side:

| Shape (m×k×n) | `backward_input` | `backward_weight` |
|---|---:|---:|
| 1×64×1000 (head, single position) | 0.028 → 0.012 ms (2.37x) | 0.021 → 0.018 ms (1.15x) |
| 128×256×4000 (head, all positions) | 59.15 → 36.21 ms (1.63x) | 38.84 → 32.82 ms (1.18x) |
| 32×256×64 (prefill FFN down) | 0.171 → 0.088 ms (1.96x) | 0.122 → 0.070 ms (1.75x) |
| 64×64×256 (prefill FFN up) | 0.335 → 0.164 ms (2.04x) | 0.277 → 0.149 ms (1.87x) |
| 128×1024×256 (medium FFN) | 10.91 → 5.51 ms (1.98x) | 9.55 → 5.07 ms (1.89x) |

The two output-head rows are the weakest at 1.15–1.18x on the weight kernel, and that is the
expected shape: `k*n` is 64000 and 1024000 there, so the axpy is walking a destination far larger
than cache and is bandwidth-bound rather than issue-bound. The FFN shapes, where the same kernel is
reused across rows, land at the ~1.9x the eight-wide FMA predicts.

The invariant that caught this now reads 1.34x at AVX2, against 2.78x before the kernels existed
and 2.27x at AVX-512 on the same machine.

NEON still has no backward kernel. That one remains genuinely unmeasured for want of AArch64
hardware, which is a different situation from this one.

The lesson worth keeping is not that AVX2 was left out — it is that **a justification can be
falsified by later work and stay in the codebase as a comment.** The AVX2 regression finding was
overturned in the very rewrite that promoted `avx2_mr4` forward, and the sentence explaining why
the backward path did not need it survived unchanged for months, load-bearing and wrong.

Measured with `./gpu_latency.out` in one session, `DRANZER_CPU_ISA=baseline` against uncapped:

| Shape (m×k×n) | `backward_input` | `backward_weight` |
|---|---:|---:|
| 1×64×1000 | 7.2 → 2.5 µs (2.88x) | 7.3 → 3.2 µs (2.28x) |
| 64×256×64 | 105.3 → 36.6 (2.88x) | 79.0 → 24.9 (3.17x) |
| 128×256×256 | 883.8 → 393.9 (2.24x) | 706.7 → 219.0 (3.23x) |
| 128×256×1024 | 3949.2 → 1815.3 (2.18x) | 3822.2 → 1871.2 (2.04x) |
| 128×1024×256 | 3597.5 → 1826.7 (1.97x) | 2785.0 → 1381.8 (2.02x) |

Whole-model, medium tier, `--cpu-only`, two binaries differing *only* in the backward dispatch,
interleaved ABBA, best of six each: **333.5 → 256.2 ms/step, 1.30x**, ranges non-overlapping
(334–396 against 256–300), with inference as an unchanged control at 61.9 against 61.8 ms/token —
1.00x.

Together with the loop-order fix, both measured the same controlled way, medium-tier CPU training
improved **2.75x × 1.30x = 3.58x** in one sitting, entirely on the CPU.

`test_matmul_backward.c` pins this. It reaches the portable path by capping detection to baseline,
which is how a CPU without any vector rung would behave, and then does something the obvious
version of this test misses: it calls each rung's entry point *directly* and requires the
dispatched result to match it bit for bit. Comparing capped against uncapped alone would pass just
as happily if the uncapped run had quietly used the portable code too. The test data is
deliberately scaled by 1/7 rather than a power of two for the same reason — an earlier version used
1/8, making every product and partial sum exactly representable, so both paths agreed to the bit
and the comparison could not have detected a dispatch that never fired.

It checks **every rung the CPU offers, not just the widest**, and that gap is how the AVX2 kernels
could have shipped unverified: the file knew only about AVX-512, so on an AVX2-only machine it
printed `PASSED (portable only)` and proved nothing about the code actually running there. Each
rung is now capped *at* its own level so dispatch must land exactly on it, and the shape sweep runs
once per available rung. Availability is sampled once before anything touches the cap, because
`cpu_features_clear_max_isa()` removes the `DRANZER_CPU_ISA` cap too — re-reading it afterwards
reports what the silicon has rather than what the run is meant to simulate, which silently turned
an intended AVX2 run back into an AVX-512 one while it was being written.

## bf16 weight storage

`core/bf16.c` adds `matmul_bf16_weight()`: the same matmul with **B stored as bfloat16** and
widened to binary32 as it is loaded. Accumulation stays in binary32 — every product and partial
sum is a full float — so the only difference from `matrix_multiply()` on the same data is that B's
values carry 8 mantissa bits instead of 24. A narrower accumulator is a separate and far more
dangerous change and is not made here.

This exists to answer the checklist's weight-only-quantization part 3: is keeping weights narrow
*in memory* and widening them per tile actually faster, or does the unpacking cost eat the
bandwidth saving? bf16 is the sharpest form of that question. Widening is
`_mm512_slli_epi32(_mm512_cvtepu16_epi32(load), 16)` — three instructions per vector, no multiply,
no per-tile scale, no zero-point — so every other narrow format costs strictly more. A loss would
have settled part 3 in the negative for the whole family; a win only sets the ceiling.

Portable, AVX2, and AVX-512 rungs, dispatched by `cpu_isa_available()` widest-first, each mirroring
`run_blocked()` and its block functions instruction for instruction.

Against `matrix_multiply()` at the same tile, ABBA-interleaved, best of seven rounds, three
sessions:

| Shape (m×k×n) | B fp32 | speedup (3 sessions) |
|---|---:|---:|
| 1×64×1000 decode head | 250 KB | 1.41x, 1.42x, 1.46x |
| 32×256×64 prefill FFN down | 64 KB | 1.13x, 1.13x, 1.13x |
| 64×64×256 prefill FFN up | 64 KB | 1.18x, 1.18x, 1.18x |
| 128×256×1024 medium FFN up | 1024 KB | 1.29x, 1.28x, 1.24x |
| 128×1024×256 medium FFN down | 1024 KB | 1.20x, 1.23x, 1.20x |
| 128×256×4000 all-position head | 4000 KB | 1.16x, 1.06x, 1.13x |
| 128×1024×1024 large square | 4096 KB | 1.40x, 1.28x, 1.45x |
| 128×2048×2048 past L3 | 16384 KB | 1.14x, 1.11x, 1.20x |

Geometric mean 1.21–1.24x per session. Every shape wins in every session; worst single reading
1.06x.

**The mechanism is not the one the hypothesis named.** The prediction was cache capacity: halve the
weights, they fit, the win appears as B outgrows L2 (1.3 MiB) and L3 (8 MiB). The table does not
show that. The steadiest wins are the *smallest* matrices — 64 KB of weights, trivially L2-resident,
reproducing 1.13x and 1.18x to three digits three times running — while the 16 MB shape, far past
L3, is among the weakest at 1.11–1.20x. There is no monotonic trend against B's size. What is
actually being saved is load bandwidth at every level of the hierarchy: half the bytes moved from
L1 into a register. That predicts the win transfers to small models, not only to large ones, which
is the opposite of what the cache story implies.

Accuracy is measured separately, because `core/quantize.h` is right that a speedup and an accuracy
loss arriving together cannot be attributed to each other. Over 60 seeds, projections and
embeddings rounded (98.9% of values), biases and norms left alone: weight-space relative RMS
**0.001645**, logit-space relative RMS movement **0.003903 ± 0.000660**, top-1 flipped on **0.208%**
of windows. Against the INT4 per-column figures in `docs/quantization.md` — 0.0907, 0.14–0.21, ~1%
— bf16 is roughly 55x more accurate in weight space and 40x in logit space, for 2x compression
rather than 8x.

### The measurement that nearly went the other way

The first version of the kernel tiled `(i, j)` and ran the full `k` inside each block, where
`run_blocked()` tiles `(i, j, l)`. Measured against it, bf16 read 0.83x at 128×1024×1024 and
**0.34x** at 128×2048×2048.

That result was seductive precisely because it confirmed the stated hypothesis: it looked exactly
like "the widening cost dominates once B leaves cache," which is what the checklist predicted would
happen. It was "a two-level kernel loses to a three-level one." The same version also fell to
scalar code for `m < 4`, reading 0.13x on a decode head against a vectorized fp32 kernel.

Rewritten to mirror `run_blocked` exactly, those three shapes read 1.36x, 1.23x, and 1.40x. **A
benchmark must vary the one thing under test and nothing else, or it measures the author** — and a
wrong number that agrees with the prediction is the one least likely to be re-examined. This is the
same failure mode recorded for `avx512_mr4`, which shipped for months on 1.83x having never been
compared against a better version of itself.

## Limitations

- **bf16 is kernel-only so far.** `matmul_bf16_weight()` is not wired into `core/transformer.c`,
  there is no bf16 path in the bundle format, and the backward pass is untouched — the numbers
  above are forward-only, with B converted once by the caller.
- **The NEON kernel is written but unmeasured.** No AArch64 machine was available. It is checked for
  correctness by the same equivalence test wherever it is built and cross-compiled by
  `make arm-check`, but `matmul_select()` does not return it — an AArch64 machine falls back to
  `tiled_mr4` — and no number in this document was produced on ARM. Run `tools/matmul_sweep.sh`
  there before promoting it.
- **The "runs without the instruction set" gate is tested by simulation, not by hardware.** No
  pre-AVX2 machine was available either. `cpu_features_set_max_isa(CPU_ISA_BASELINE)` makes
  detection report exactly what such a CPU would report, so every dispatch decision below it is the
  real code path; what that cannot prove is the absence of a wide instruction on a reachable path,
  which is a property of the `target` attributes rather than of dispatch.
- The kernels are register-blocked but not packed. `A` and `B` are read from their original
  row-major layout, so the innermost loop's `B` access is contiguous but its `A` access is strided.
  A packing pass is the usual next step and is not attempted here.
- **The backward-pass kernels are not part of the candidate set.** `matmul_backward_input` and
  `matmul_backward_weight` have one implementation each with no selectable alternative, so nothing
  sweeps them the way `matmul_select()`'s candidates are swept. They are no longer plain C — see
  [The backward kernels](#the-backward-kernels) for the loop-order fix and the AVX-512 paths — but
  their dispatch is a single `cpu_isa_available()` branch rather than a measured policy.
- Tiles are square. A rectangular blocking scheme (different extents for `m`, `k`, and `n`) is not
  explored.
