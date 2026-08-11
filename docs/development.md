# Development

[← Back to README](../README.md)

DRANZER treats numerical correctness as a permanent feature. Compiler coverage, gradient checks,
sanitizers, CPU/GPU comparisons, and scheduled builds are all part of the normal workflow.

## Standard checks

Run from the repository root:

```bash
make -C src clean all test CC=clang
make -C src clean all test CC=gcc
```

`make test` discovers every `src/tests/core/test_*.c`, `src/tests/cli/test_*.c`,
`src/tests/gpu/test_*.c`, `src/tests/perf/test_*.c`, and future `src/tests/tpu/test_*.c` file. Each source becomes an
independent executable. The target runs all tests and returns a nonzero status if any build or test
fails.

## Test suite

| Test | Protection provided |
|---|---|
| `test_gradient_check.c` | Analytical gradients versus finite differences across layers |
| `test_dropout_gradient_check.c` | Backpropagation with a deterministic dropout mask |
| `test_dropout_statistics.c` | Drop rate, mean preservation, and inference behavior |
| `test_adam_convergence.c` | AdamW convergence and lazy moment allocation |
| `test_lr_schedule.c` | Warmup, cosine decay, and minimum learning rate |
| `test_minibatch_accumulation.c` | Averaged sample gradients and one true minibatch optimizer update |
| `test_serialization_roundtrip.c` | Identical logits before and after save/load |
| `test_model_bundle.c` | Canonical round-trip, corruption sweep, bounds, and legacy fixture |
| `test_attention_mask.c` | Padding/general-mask inference parity, empty rows, causality, and masked backward |
| `test_public_api.c` | Opaque handle loading, binary-safe buffers, cache parity, retained ownership, and greedy generation |
| `test_library_diagnostics.c` | Invalid CPU/GPU environment settings are structured and produce no terminal output |
| `test_scalar_reference.c` | Tiled/full-model/cached-decode agreement with portable scalar matmul |
| `test_matmul_backward.c` | Vector backward kernels against the portable reference, accumulation into a non-zero destination, and proof that dispatch reaches the vector path |
| `test_matmul_kernels.c` | Every available kernel (portable and dispatched SIMD) at every tile size versus the scalar reference, plus selection determinism, configuration validation, and correct fallback when the instruction set is unavailable |
| `test_batch_behavior.c` | Bounded deterministic shuffle and batch capacity checks |
| `test_checkpoint_resume.c` | Complete-state round-trip, continued trajectory, latest selection, and retention |
| `test_cli_strict.c` | Unknown/missing/malformed/overflowing CLI input rejection |
| `test_kv_cache.c` | Full-prefix equivalence, reset, absolute positions, and multi-wrap ring layout |
| `test_evaluation_state.c` | Evaluation leaves parameters, optimizer, scheduler, and metrics unchanged |
| `test_config_provenance.c` | Frozen-tokenizer and corpus metadata configuration round-trip |
| `test_evaluation_overfit.c` | Tiny-corpus overfit improvement on a separate held-out file |
| `test_frozen_tokenizer_training.c` | Stable token IDs before and after model optimization |
| `test_generation_controls.c` | Stream callbacks, stop withholding, minimum length, penalties, and vocabulary masking |
| `test_sampling.c` | Greedy/top-k behavior and correct top-p candidate selection |
| `test_special_tokens.c` | Stable IDs, persistence, legacy fixture, control masking, and EOS stop |
| `test_tokenizer_serialization.c` | Learned vocabulary, encoding, and decoding round-trip |
| `test_gpu_matmul.c` | GPU matmul versus CPU reference |
| `test_gpu_model_forward.c` | Full GPU-dispatched forward pass versus CPU |
| `test_gpu_weight_cache.c` | Correct cache invalidation after weight changes |
| `test_gpu_training_step.c` | CPU/GPU agreement across repeated training updates |
| `test_gpu_matmul_backward.c` | GPU backward kernels versus the CPU reference, including accumulation into a non-zero destination |
| `test_perf_invariants.c` | Ratio-only CPU timing gates: backward matmuls within reach of a forward matmul of equal FLOP count, tuned kernel beats the scalar reference, and each AVX-512 path beats its portable counterpart |
| `test_gpu_latency_invariants.c` | Ratio-only GPU timing gates: fixed per-call cost is amortized by larger work, the weight cache removes a per-call upload, and the tiled kernel is not a regression on the naive one |
| `test_gpu_training_backward.c` | CPU/GPU agreement on a model large enough that the dispatched backward kernels are actually used |

GPU tests return success with a clear `SKIP` message when CUDA hardware is unavailable.

`make library-silence-check` mechanically rejects `printf`/`fprintf`/`puts`/`putchar`/`perror`
calls in embeddable core, API, tokenizer, CUDA, and GPU-matmul modules. File-format writers may use
`snprintf` and stream writes; executable CLI/probe/report modules remain free to print.

The suite also runs two shell-level integration gates:

- `test_eval_cli.sh` exercises train-time validation and standalone evaluation, deletes the
  tokenizer sidecar to prove the bundle is self-contained, rejects tokenizer overrides, and
  verifies the model artifact is unchanged.
- `test_resume_cli.sh` interrupts a shuffled, minibatched, gradient-accumulated,
  dropout-and-scheduled-AdamW run at a real periodic checkpoint, removes the model and tokenizer
  sidecar, resumes, and requires byte-identical final model and checkpoint files. It also covers
  terminal `--resume latest` and rejects trajectory-changing overrides.

## Build variants

```bash
# Debug logging and symbols
make -C src clean all test CC=clang DEBUG=1

# AddressSanitizer
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  make -C src clean all test CC=clang ASAN=1

# Parallel CPU paths
make -C src clean all test CC=gcc OMP=1

# Size-optimized build and tools
make -C src clean all test bench gpu-probe CC=clang SIZE=1
```

Always clean when switching compiler or instrumentation flags because object filenames are shared
between build variants.

**Leak detection is disabled for `tests/gpu/` only.** The `make test` recipe runs those binaries
with `ASAN_OPTIONS=detect_leaks=0`, which overrides the setting above for them alone; every other
test keeps full leak detection, and ASAN's buffer-overflow and use-after-free checks stay on
everywhere including the GPU tests. The reason is the NVIDIA driver: it keeps process-global
allocations that outlive `cuCtxDestroy`, so on any machine with a usable GPU each `tests/gpu/`
binary reported about 49 KB of driver state with no project frames in the stack. Suppressing by
library name caught only half of it — LeakSanitizer could not attribute the rest to any module.
A leak in the project's own `gpu_cuda.c` or `gpu_matmul.c` is therefore not caught by this suite
and has to be reasoned about from the shutdown paths, which `gpu_matmul_shutdown()` covers by
destroying the context that owns every device allocation.

## Performance regression tests

`src/tests/perf/` and `test_gpu_latency_invariants.c` run inside the ordinary suite and fail the
build on performance regressions, rather than leaving them to be noticed in a benchmark table.

**Every assertion in them is a ratio between two measurements taken back to back in the same
process. None asserts an absolute time.** A slow, busy, or thermally throttled machine scales both
sides of a ratio equally, which is what makes timing assertions safe to run alongside correctness
tests. It is the same discipline `tools/perf_check.py` applies to CI benchmark rows.

Thresholds are deliberately loose, because the defects worth catching are order-of-magnitude:

| Invariant | Limit | What it catches |
|---|---|---|
| backward-input vs forward of equal FLOP count | ≤ 3.0x | A traversal fighting the cache |
| backward-weight vs forward of equal FLOP count | ≤ 3.5x | The same defect, with room for the 3.06x [2.93x, 3.10x] measured by hosted GCC. The pre-fix kernel reports **24.7x** here. |
| tuned forward kernel vs scalar reference | ≥ 1.5x | Kernel selection silently stopping |
| AVX-512 forward/input paths vs portable | ≥ 1.2x | A dispatch that stopped firing — invisible to correctness tests, which still pass |
| AVX-512 backward-weight path vs portable | ≥ 1.1x | The same collapse, with room for the 1.18x [1.17x, 1.19x] measured on EPYC 9V74 |
| large GPU call vs 1-element call, in GFLOP/s | ≥ 1000x | Per-call overhead exploding |
| GPU weight cache vs invalidating every call | ≥ 1.05x | The cache not being consulted |
| tiled GPU kernel vs the naive baseline | ≥ 1.0x | The shared-memory path regressing |

The backward-vs-forward gate was verified to have teeth by rebuilding the suite against the
pre-fix kernel: it fails at 24.7x against a 3.5x limit, and also catches the AVX-512 dispatch ratio
collapsing to 0.95x. A performance test that has never been seen to fail is not evidence of
anything.

**These tests skip themselves under a sanitizer.** Instrumentation is not applied evenly - portable
scalar loops carry far more of it per useful operation than vector intrinsics do - so under
AddressSanitizer the ratios inverted and inflated: the backward matmuls measured 0.69x a forward
matmul of equal FLOP count, and the AVX-512 speedups read 8-16x instead of about 2-3x. They passed
with enormous margin, which is worse than failing, because it looks like coverage while measuring
the sanitizer. The correctness suite still runs under ASAN, which is where it belongs. The GPU
tests skip there too, for an unrelated reason: the CUDA driver does not initialize under the
sanitizer, so all of `tests/gpu/` self-skips.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request. Its matrix builds the CLI and runs
the complete suite with GCC and Clang on Ubuntu 24.04.

`.github/workflows/nightly.yml` runs daily at 02:17 UTC and is also manually dispatchable. It adds:

- Clang AddressSanitizer with leak detection (except `tests/gpu/`, see above);
- GCC with OpenMP enabled;
- the size-optimized configuration;
- compilation of the benchmark and GPU probe;
- CLI and hardware-probe smoke tests.

`.github/workflows/performance.yml` runs nightly at 03:41 UTC and is manually dispatchable with
custom tiers and repeat counts. It exists because a developer machine is a poor measurement
instrument - on the project's own 2-core WSL2 workstation, repeating one unchanged benchmark six
times spanned a factor of 1.45. A hosted runner is dedicated for the length of the job and has more
cores, so it can answer questions a contended laptop cannot. Three jobs run:

| Job | What it measures |
|---|---|
| Matmul kernel sweep | Every kernel and tile candidate on every model shape, per compiler |
| Whole-model throughput | Inference, prefill, decode, training, and memory per tier, per compiler |
| Thread scaling | One run per thread count, up to the runner's core count |

Each job writes a markdown report to the GitHub step summary and uploads its raw CSV as a 90-day
artifact, so a trend can be followed by downloading successive runs.

**No absolute timing is asserted on.** Hosted runners vary by CPU model and neighbour load, so a
committed "this must finish in X ms" threshold fails for reasons unrelated to the code. Instead
`src/tools/perf_check.py` checks ratios between measurements taken in the same process on the same
machine, which hold regardless of how fast that machine is:

- the shipped matmul kernel must beat the portable scalar reference on every shape;
- it must stay within 2.5× of the fastest candidate measured beside it (1.5× warns), which catches a
  bad default without pinning a speed;
- KV-cached decode must beat recomputing the full prefix by at least 3× (observed 10–44×);
- every kernel must stay within numerical tolerance of the reference.

Run the same checks locally against any CSV the tools produce:

```bash
cd src
python3 tools/perf_check.py --bench bench_results_v3.csv --matmul matmul_results_v3.csv

### Reading the results

Every job in `.github/workflows/performance.yml` writes its full report — machine details, the
measurement tables, and each gate's pass/fail — to its **GitHub job summary**. GitHub stacks those
on the workflow run page, so opening the run shows every result with nothing to download.

The CSV artifacts are for charting a trend across runs by hand. They are not where you look for the
outcome of a single run, and reaching for them usually means the report is missing, which is a bug
rather than the intended workflow.

The report is written with `if: always()`, so it appears even when the measurement step failed — in
that case it says so explicitly and names the CSV it expected, rather than leaving the run page
blank. A blank summary alongside an uploaded artifact was the previous behaviour and is what this
guards against: the artifact upload already ran unconditionally while the report did not, so a
failed sweep produced downloadable numbers and no readable result.
```

`python3` is needed only for this report, not to build, test, or run anything.

GPU offload is not measured in CI because hosted runners have no GPU; use
`make gpu-latency && ./gpu_latency.out` on a machine that has one.

Jobs use read-only repository permissions, pinned action revisions, concurrency controls, and
timeouts. `.github/dependabot.yml` checks workflow actions weekly.

## Benchmarking

```bash
cd src
make bench
./bench.out
```

The benchmark runs representative tiny, small, and medium model configurations. It reports:

- parameter memory;
- peak resident memory;
- full-context inference latency and throughput;
- training-step latency and throughput;
- prompt-prefill, growing KV decode, and full-ring sliding decode latency as separate metrics;
- CPU and, when available, GPU execution.

Results append to `bench_results_v3.csv`, which is intentionally ignored by Git because measurements
from different machines are not directly comparable. Use a stable machine, compiler, thread count,
and power state for before/after performance work. Each row records the exact build command,
compiler, OS/kernel/architecture, CPU model, detected SIMD instruction set, online CPU count,
OpenMP version, and maximum thread
count alongside its measurements.

Bundle startup and resident memory have a focused two-process comparison. It accepts version-1
float bundles because those can be mapped without representation conversion:

```bash
make bench-bundle-load
./bench_bundle_load.out path/to/model.bin --mode copy --repeats 7
./bench_bundle_load.out path/to/model.bin --mode mmap --repeats 7
```

Both rows land in `bundle_load_results_v1.csv` with the same build and host provenance as the other
benchmarks. Keep the modes in separate invocations: peak RSS is process-wide and cannot be reset
after the copy loader has established a higher watermark.

To isolate the matrix-multiplication kernels from the rest of the model, compare them on the decode,
prefill, and training shapes the model actually issues:

```bash
./bench.out --matmul-only --tier small           # scalar vs the shipped default
./bench.out --matmul-only --sweep --repeats 5    # every kernel and tile candidate
tools/matmul_sweep.sh                            # the same sweep across gcc and clang
```

Every candidate is checked against the scalar reference before it is timed, and each row records
latency (fastest and median round), GFLOP/s, speedup, error, iteration counts, and the same
machine/build metadata in `matmul_results_v3.csv`. `--kernel` and `--tile` pin a kernel for a whole
run, including full-model runs, so a kernel choice can be judged end to end and not only in
isolation. Use `--quick` only to validate the workflow; omit `--tier` to cover every model tier.

[CPU matmul kernels](matmul.md) documents the kernel set, the selection policy, the reproducibility
contract, and how to read a sweep — including why candidates are ranked by their fastest round.

The initial single-threaded GCC measurement on an Intel i5-11320H showed the following steady-state
decode results (last eight context positions, prompt prefill excluded):

| Model tier | Full-prefix | KV cache | Speedup |
|---|---:|---:|---:|
| tiny/default | 0.171 ms/token | 0.011 ms/token | 15.80× |
| small | 4.493 ms/token | 0.111 ms/token | 40.42× |
| medium | 419.839 ms/token | 6.207 ms/token | 67.64× |

Treat these as a reproducible historical example, not a cross-machine promise. `bench.out` reports
the same pre-eviction comparison plus independent prompt-prefill and steady-state ring metrics;
all three are stored separately in the version-2 CSV.

To regenerate one consolidated current-machine report across compilers, OpenMP modes, CPU/GPU, and
all model tiers:

```bash
scripts/repro/benchmark_all.sh docs/generated/benchmark-all.md
scripts/repro/benchmark_all.sh /tmp/benchmark-smoke.md --quick --repeats 1
```

Each configuration is a clean rebuild. CPU and GPU sections are emitted as separate rows, every
cell is a median over repeated runs, the training spread stays beside it, and the raw reports are
retained outside the checkout so a failed parser cannot destroy the underlying measurement.

For an OpenMP comparison:

```bash
make clean
make bench OMP=1 CC=gcc
OMP_NUM_THREADS=4 ./bench.out
```

Threading has its own measurement tool, because whether a parallel region is worth entering cannot
be seen from a whole-model number:

```bash
make bench-parallel OMP=1 CC=gcc
OMP_NUM_THREADS=4 ./bench_parallel.out --rounds 7
```

It reports what a region entry costs, how a persistent worker pool compares, and where forking
starts paying for itself, appending every row to `parallel_results_v1.csv`. Read its `cannot fork`
rows first: those run identical code on both sides, so their spread is the machine's noise floor and
anything inside it is not a measurement. [CPU threading](threading.md) documents the cutoff, the
threshold's derivation, and why the worker pool was rejected.

Quantization accuracy has its own tool, for the same reason - a single held-out number cannot say
where the error entered or whether it is larger than seed noise:

```bash
make bench-quant CC=gcc
./bench_quant.out --seeds 60
./bench_quant.out --seeds 12 --per-tensor
```

It trains several models on a corpus with a closed-form entropy floor, quantizes each one, and
reports the cost in weight space, logit space, and held-out cross-entropy, appending rows to
`quant_results_v1.csv`. It exits non-zero if the trained model failed to beat a uniform predictor,
so a broken training configuration fails loudly instead of producing a plausible table. Read the
`seeds req` column before believing any "resolvable" verdict - a 12-seed run in this project
produced a conclusion the 60-seed rerun withdrew. [Weight quantization](quantization.md) documents
the schemes, the three levels, and that reversal.

### Profiling workflow

Build a frame-pointer-enabled benchmark and select one bounded tier before changing a kernel:

```bash
make -C src profile CC=gcc
perf stat --repeat 3 ./src/bench_profile.out --tier small
perf record -g -o perf.data -- ./src/bench_profile.out --tier small
perf report -i perf.data
```

Use `--quick` for a smoke run, not a performance claim. Use `--scalar` to force the portable
unblocked C matmul through the full model; omitting it selects the normal CPU dispatch path, and
`--kernel`/`--tile` pin a specific kernel instead. The scalar path and dispatch path are compared
directly by `test_scalar_reference.c`. `perf` is a Linux developer dependency, not a build or
runtime dependency; `make profile` succeeds without it.

## Hardware probing

```bash
cd src
make gpu-probe
./gpu_probe.out
```

The probe is a diagnostic tool, not a prerequisite. It is expected to run successfully and report
unavailable backends on CPU-only systems. See [GPU backend](gpu.md) for its design.

For the question "is GPU offload worth it here", `make gpu-latency && ./gpu_latency.out` measures
what a single `gpu_matmul()` call costs once device buffers are resident: transfer latency at
several sizes, an empty kernel launch, and whole calls at the shapes the model issues. Small
models lose on the GPU because of that fixed per-call cost, not because of allocation - see
[GPU backend](gpu.md). It exits cleanly with an explanation when no CUDA device is present.

## Adding a test

1. Put backend-neutral tests in `src/tests/core/test_<name>.c`.
2. Put CUDA-specific equivalence tests in `src/tests/gpu/test_<name>.c`.
3. Give the file its own `main()` and return nonzero on failure.
4. Make unavailable optional hardware a successful, explicit skip.
5. Run both GCC and Clang suites before submitting the change.

No Makefile edit is required for a normally named test because the test target discovers files by
glob.

## Build system

`src/Makefile` builds the application, model tests, benchmark, and capability probe. It delegates
the tokenizer/hashmap archive to `libs/Makefile`.

```text
C sources → object files → libs/libattention.a + model objects → executable
```

The application and GPU-facing binaries link `libdl` because runtime GPU APIs are loaded with
`dlopen()`. The normal CPU build does not require vendor headers or SDK libraries.

## Review checklist

Feature work is championed in the ordered [design and maturity checklist](design-checklist.md).
Only one roadmap goal should be active at a time, and its acceptance gate must be satisfied before
advancing to the next goal.

- Does a behavioral change include a focused regression test?
- Do both GCC and Clang builds pass?
- Do numerical tolerances reflect floating-point error rather than hide a mismatch?
- Are allocations paired on every error path?
- Does GPU code retain a correct CPU fallback?
- Are performance claims backed by `bench.out` on identified hardware?
- Are new headers placed under the matching `src/include/` hierarchy?
