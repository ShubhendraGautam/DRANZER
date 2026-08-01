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

`make test` discovers every `src/tests/core/test_*.c`, `src/tests/gpu/test_*.c`, and future
`src/tests/tpu/test_*.c` file. Each source becomes an independent executable. The target runs all
tests and returns a nonzero status if any build or test fails.

## Test suite

| Test | Protection provided |
|---|---|
| `test_gradient_check.c` | Analytical gradients versus finite differences across layers |
| `test_dropout_gradient_check.c` | Backpropagation with a deterministic dropout mask |
| `test_dropout_statistics.c` | Drop rate, mean preservation, and inference behavior |
| `test_adam_convergence.c` | AdamW convergence and lazy moment allocation |
| `test_lr_schedule.c` | Warmup, cosine decay, and minimum learning rate |
| `test_serialization_roundtrip.c` | Identical logits before and after save/load |
| `test_kv_cache.c` | Incremental logits versus full-prefix logits at every position |
| `test_sampling.c` | Greedy/top-k behavior and correct top-p candidate selection |
| `test_tokenizer_serialization.c` | Learned vocabulary, encoding, and decoding round-trip |
| `test_gpu_matmul.c` | GPU matmul versus CPU reference |
| `test_gpu_model_forward.c` | Full GPU-dispatched forward pass versus CPU |
| `test_gpu_weight_cache.c` | Correct cache invalidation after weight changes |
| `test_gpu_training_step.c` | CPU/GPU agreement across repeated training updates |

GPU tests return success with a clear `SKIP` message when CUDA hardware is unavailable.

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
- full-prefix versus KV-cached autoregressive decode latency;
- CPU and, when available, GPU execution.

Results append to `bench_results.csv`, which is intentionally ignored by Git because measurements
from different machines are not directly comparable. Use a stable machine, compiler, thread count,
and power state for before/after performance work.

The initial single-threaded GCC measurement on an Intel i5-11320H showed the following steady-state
decode results (last eight context positions, prompt prefill excluded):

| Model tier | Full-prefix | KV cache | Speedup |
|---|---:|---:|---:|
| tiny/default | 0.171 ms/token | 0.011 ms/token | 15.80× |
| small | 4.493 ms/token | 0.111 ms/token | 40.42× |
| medium | 419.839 ms/token | 6.207 ms/token | 67.64× |

Treat these as a reproducible example, not a cross-machine promise; `bench.out` reports the same
comparison for the machine on which it runs.

For an OpenMP comparison:

```bash
make clean
make bench OMP=1 CC=gcc
OMP_NUM_THREADS=4 ./bench.out
```

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

- Does a behavioral change include a focused regression test?
- Do both GCC and Clang builds pass?
- Do numerical tolerances reflect floating-point error rather than hide a mismatch?
- Are allocations paired on every error path?
- Does GPU code retain a correct CPU fallback?
- Are performance claims backed by `bench.out` on identified hardware?
- Are new headers placed under the matching `src/include/` hierarchy?
