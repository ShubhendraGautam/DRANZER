# GPU backend

[← Back to README](../README.md)

DRANZER can offload forward-pass matrix multiplication to an NVIDIA GPU without CUDA headers,
`nvcc`, or NVRTC. It dynamically loads `libcuda.so.1` and asks the installed driver to JIT-compile
an embedded hand-written PTX kernel.

## Scope

GPU acceleration is:

- optional and disabled by default;
- NVIDIA-only;
- Linux-focused;
- limited to matrix multiplications in the forward pass;
- transparent when unavailable—the same call falls back to CPU.

The backward pass and optimizer remain on the CPU. At tiny model sizes, GPU launch and transfer
overhead can exceed the compute time saved.

## Request GPU execution

```bash
cd src
make
./app.out infer --model dranzer.pth --prompt "hello" --gpu
```

`--gpu` is a request, not a requirement. `dispatch_matmul()` checks availability and uses the CPU
path if initialization fails.

## Capability probe

```bash
cd src
make gpu-probe
./gpu_probe.out
```

The probe uses stable, documented runtime APIs and reports each subsystem independently:

| Probe | What it establishes |
|---|---|
| DRM | Whether Linux GPU device nodes are present |
| NVML | NVIDIA identity, memory, driver version, and telemetry |
| CUDA Driver API | Whether a CUDA context and kernels can run |
| OpenCL | Whether the ICD loader and a vendor platform are registered |
| Device attributes | Supported `cuDeviceGetAttribute` values across codes 1–128 |
| Microbenchmarks | Measured memory bandwidth and FP32 FMA throughput |

A driver library being present is not the same as a functional device. The report distinguishes
`READY`, `PARTIAL`, and `UNAVAILABLE` states and gives a CPU fallback recommendation.

## PTX execution path

```text
host matrices
   → persistent CUDA buffers
   → embedded PTX loaded by cuModuleLoadData
   → driver JIT compilation
   → kernel launch
   → result copied back to host
```

`gpu_cuda.c` contains the minimal dynamically loaded Driver API wrapper. `gpu_matmul.c` owns the
kernel, buffer sizing, weight cache, launch parameters, and CPU-compatible API.

## Persistent buffers and weight cache

The first GPU implementation allocated and freed every device buffer on every matmul. That was
correct but dominated runtime for small tensors. The current backend keeps:

- scratch buffers for activations and output, grown only when a larger operation arrives;
- device copies of weight matrices keyed by their host pointer.

Weights remain cached until `gpu_matmul_invalidate_weights()` is called. The optimizer invalidates
them after every update, and model creation/destruction also invalidates the cache so a reused host
address can never expose stale weights from an older model.

Two tests protect this behavior:

- `test_gpu_weight_cache.c` mutates weights and verifies invalidation exposes the new values.
- `test_gpu_training_step.c` compares CPU and GPU training across multiple updates.

Both self-skip when a CUDA device is unavailable.

## Measured crossover

Historical measurements from the project's NVIDIA MX450 test system illustrate why `--gpu` is not
automatically faster:

| Model tier | CPU inference | Cached GPU inference | Result |
|---|---:|---:|---|
| tiny/default | 0.31 ms/token | about 7× slower | launch overhead dominates |
| small | 11.9 ms/token | about 2× faster | GPU begins to win |
| medium | 542.6 ms/token | about 5.1× faster | compute dominates overhead |

These numbers are not portable benchmarks; hardware, driver, thermals, and model dimensions all
matter. Run `bench.out` on the target system before deciding whether to enable GPU execution.

## Capability cache

When NVML can identify the GPU and driver, the probe stores a plain-text result under
`gpu_capability_cache/`:

```bash
./gpu_probe.out
./gpu_probe.out --load gpu_capability_cache/<gpu>_<driver>.cache
```

The cache includes identity, attributes, telemetry-derived facts, and benchmark results. Its key
contains both the model and driver version so incompatible observations are not silently reused.

## Attributes, telemetry, and theoretical performance

The attribute scanner queries numeric CUDA device attribute codes 1 through 128. Only names that
have been independently verified are labeled; other successful codes are retained as unlabeled
values rather than guessed.

NVML telemetry can include temperature, power, clocks, ECC state, performance state, persistence
mode, utilization, and memory. Each value is optional because support varies by device.

The theoretical FP32 estimate combines:

```text
SM count × FP32 cores per SM × graphics clock × 2 operations per FMA
```

The microbenchmark complements that ceiling with achieved copy bandwidth and FP32 FMA throughput.
Runtime inputs are passed into the PTX FMA loop so the driver's JIT cannot replace the workload
with a constant.

## Sanitizer note

Some NVIDIA driver libraries retain small process-lifetime allocations that LeakSanitizer reports
inside `libcuda` or `libnvidia-ml`. The project shuts down NVML and closes libraries on its own
paths; vendor-owned reports should be distinguished from allocations whose stack traces point into
this repository.

## Limitations and next steps

- No AMD, Intel, or Apple GPU execution backend
- No GPU kernels for backward, softmax, layer norm, dropout, or optimizer updates
- Host/device transfers remain in the forward path
- No multi-GPU execution

The natural next performance step is keeping more of the forward graph resident on the device;
the natural training step is implementing and validating backward kernels against the CPU
reference.
