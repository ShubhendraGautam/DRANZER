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
`src/tests/gpu/test_*.c`, and future `src/tests/tpu/test_*.c` file. Each source becomes an
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
| `test_scalar_reference.c` | Tiled/full-model/cached-decode agreement with portable scalar matmul |
| `test_matmul_kernels.c` | Every kernel and tile size versus the scalar reference, plus selection determinism and configuration validation |
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

GPU tests return success with a clear `SKIP` message when CUDA hardware is unavailable.

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

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request. Its matrix builds the CLI and runs
the complete suite with GCC and Clang on Ubuntu 24.04.

`.github/workflows/nightly.yml` runs daily at 02:17 UTC and is also manually dispatchable. It adds:

- Clang AddressSanitizer with leak detection;
- GCC with OpenMP enabled;
- the size-optimized configuration;
- compilation of the benchmark and GPU probe;
- CLI and hardware-probe smoke tests.

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

Results append to `bench_results_v2.csv`, which is intentionally ignored by Git because measurements
from different machines are not directly comparable. Use a stable machine, compiler, thread count,
and power state for before/after performance work. Each row records the exact build command,
compiler, OS/kernel/architecture, CPU model, online CPU count, OpenMP version, and maximum thread
count alongside its measurements.

To isolate the matrix-multiplication kernels from the rest of the model, compare them on the decode,
prefill, and training shapes the model actually issues:

```bash
./bench.out --matmul-only --tier small           # scalar vs the shipped default
./bench.out --matmul-only --sweep --repeats 5    # every kernel and tile candidate
tools/matmul_sweep.sh                            # the same sweep across gcc and clang
```

Every candidate is checked against the scalar reference before it is timed, and each row records
latency (fastest and median round), GFLOP/s, speedup, error, iteration counts, and the same
machine/build metadata in `matmul_results_v2.csv`. `--kernel` and `--tile` pin a kernel for a whole
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

For an OpenMP comparison:

```bash
make clean
make bench OMP=1 CC=gcc
OMP_NUM_THREADS=4 ./bench.out
```

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
