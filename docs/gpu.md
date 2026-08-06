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
- limited to matrix multiplications — all of them in the forward pass, and the two backward
  matmuls above a measured shape threshold;
- transparent when unavailable—the same call falls back to CPU.

The optimizer step, attention scores, softmax, and layer normalization remain on the CPU, and
activations round-trip to host memory between operations. At tiny model sizes, GPU launch and
transfer overhead can exceed the compute time saved — for the forward pass this is now true well
beyond tiny, see [Measured crossover](#measured-crossover).

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

## What a call still costs

Because buffers are already resident, what remains per `gpu_matmul()` call is one activation
upload, the launch, and one result download. Reproduce with:

```bash
cd src
make gpu-latency
./gpu_latency.out
```

Measured on the WSL2 test system below (fastest of five rounds of 200 calls), each of those is a
driver round-trip with a latency floor that has nothing to do with how much data moves:

| Operation | Cost | Effective rate |
|---|---:|---|
| Upload 1 KB | 53 µs | 0.02 GB/s |
| Upload 16 KB | 38 µs | 0.43 GB/s |
| Upload 1 MB | 258 µs | 4.06 GB/s |
| Download 1 MB | 308 µs | 3.41 GB/s |
| Empty kernel launch + synchronize | 51 µs | — |

Small transfers are pure latency: 1 KB costs more than 16 KB. That fixed cost — roughly 100–140 µs
per matmul — is why small models lose on the GPU no matter how good the kernel is. A medium-tier
forward pass issues 37 matmuls, so it pays that toll 37 times.

The kernel itself is not the problem. The same measurement timing whole calls:

| Shape | Whole call | Achieved |
|---|---:|---|
| 1×256×256 (decode projection) | 98 µs | 1.3 GFLOP/s |
| 1×256×4000 (decode output head) | 212 µs | 9.7 GFLOP/s |
| 128×256×256 (prefill projection) | 275 µs | 61 GFLOP/s |
| 128×256×1024 (prefill FFN up) | 652 µs | 103 GFLOP/s |

At prefill shapes the GPU reaches 4–14× the CPU kernel's throughput. At single-token decode shapes
the fixed cost swamps roughly two microseconds of actual arithmetic.

One redundant round-trip was removed as a result of this measurement: the launch used to block on
`cuCtxSynchronize()` and was then immediately followed by a blocking device-to-host copy, which
already orders itself after the kernel on the default stream. `gpu_matmul()` now uses
`gpu_cuda_launch_2d_async()` and lets the download do the waiting, which measured 4–18% off each
call. Whole-model runs on this contended machine could not resolve a difference that size, so no
model-level claim is made for it. `gpu_cuda_launch_2d()` still blocks and is what `gpu_microbench.c`
uses, since timing a kernel requires waiting for it.

The remaining overhead is structural rather than fixable in the backend: only matmul runs on the
GPU, so activations return to host memory between every operation. Keeping them device-resident
would require layer normalization, softmax, and attention on the device too.

## Backward pass

The two backward matmuls are dispatched as well, but on a threshold rather than unconditionally.

```text
backward_input   dA (m x k) += dC (m x n) @ B_transposed
backward_weight  dB (k x n) += A_transposed @ dC (m x n)
```

Both are hand-written PTX in the same module as the forward kernel, so the driver JITs once and
switching between the three costs nothing at launch. Both vectorize nothing and tile nothing — one
thread per output element, matching the forward kernel's deliberately naive design.

**They accumulate into their destination**, because every backward call site adds into a gradient
buffer that a whole minibatch shares. That is the reason for the threshold: a forward matmul moves
two buffers (activations up, result down) with its weight operand already resident, while a
backward matmul moves four — both inputs up, and the destination both up *and* down. Below a
certain amount of arithmetic those extra transfers cost more than the kernel saves.

Measured with `./gpu_latency.out`, CPU against GPU on the same buffers in the same process:

| Shape (m×k×n) | `backward_input` | `backward_weight` |
|---|---:|---:|
| 1×64×1000 | 0.02x | 0.01x |
| 1×256×4000 | 0.16x | 0.05x |
| 64×64×64 | 0.09x | 0.08x |
| 64×256×64 | 0.23x | 0.16x |
| 128×256×256 | 0.64x | 0.62x |
| 128×256×1024 | 0.95x | **1.62x** |
| 128×1024×256 | 0.93x | **1.32x** |

`core/training.c` dispatches `backward_weight` above 2²⁵ multiply-accumulates and `backward_input`
above 2²⁶. Below either threshold the CPU runs, which is always correct and never worse than having
no GPU.

Only the weight threshold is measured. `backward_weight` wins at the two largest shapes the
benchmark issues (both 2²⁵) and loses at 2²³, so 2²⁵ sits on the far side of an observed crossover.
`backward_input` never won at any measured shape — it reaches 0.95x and is still climbing — so its
threshold extrapolates that trend beyond every shape this project benchmarks. In practice
`backward_input` runs on the CPU for every model here, which is what the measurements support; the
path stays for larger models and faster cards, where it should be re-measured rather than trusted.

**These numbers have been revised downwards twice, and the reason matters more than the numbers.**

| When | `backward_weight` best | Thresholds |
|---|---:|---|
| Backward dispatch first added | up to **30x** | 2²⁰ / 2²³ |
| After the CPU loop-order fix | 3–5x | 2²³ / 2²³ |
| After AVX-512 backward kernels | 1.3–1.6x | 2²⁵ / 2²⁶ |

The GPU never got slower. Each time, the CPU implementation it is measured against got faster — a
loop order worth 5.6–21.8x, then vector kernels worth another 2–3x (see
[CPU matmul kernels](matmul.md#the-backward-kernels)) — and the range of shapes where a device round
trip pays shrank accordingly.

The GPU did not get worse; the competition got better. The lesson for anyone reading a speedup
table in this repository: a large ratio is a claim about *two* implementations, and the cheap one to
suspect first is the baseline. These thresholds are measured against the CPU implementation as it
exists today and must be re-derived if it changes again.

## Measured crossover

`--gpu` is not automatically faster, and since the CPU gained runtime-dispatched AVX-512 kernels
(see [CPU matmul kernels](matmul.md)) it is no longer even reliably faster for inference.

Current measurements, medium tier, best of three runs each on the MX450 test system:

| Workload | CPU | GPU | Result |
|---|---:|---:|---|
| Inference (ms/token) | 65.8 | 89.3 | **CPU wins, 0.74x** |
| Training (ms/step) | 536.4 | 477.4 | **GPU wins, 1.12x** |

The inference ranges do not overlap (CPU 65.8–72.0 against GPU 89.3–96.6). The training margin is
narrow and the ranges do touch (CPU 536–601 against GPU 477–552), so treat 1.12x as "the GPU is
still ahead on training, but not by much" rather than as a precise figure.

**That training row used to read 969.3 against 627.9, a 1.54x GPU win.** Fixing
`matmul_backward_weight()`'s loop order (see [CPU matmul kernels](matmul.md#the-backward-kernels))
made CPU training 2.75x faster on its own and took most of the GPU's advantage with it. Both
numbers above are honest; they were measured five hours apart against different CPU code.

**This inverted an earlier result and the old table is kept below as a record of that.** Before
AVX-512 dispatch, the same machine measured medium-tier inference about 5.1x *faster* on the GPU.
Speeding up the CPU forward path by roughly 1.8x on matmul moved the crossover past the largest
tier this project benchmarks. Training still favours the GPU because the backward pass is where the
CPU is weakest — see the caveat about `matmul_backward_weight` above.

Superseded, retained for context (pre-AVX-512, forward-only GPU dispatch):

| Model tier | CPU inference | Cached GPU inference | Result |
|---|---:|---:|---|
| tiny/default | 0.31 ms/token | about 7× slower | launch overhead dominates |
| small | 11.9 ms/token | about 2× faster | GPU begins to win |
| medium | 542.6 ms/token | about 5.1× faster | compute dominates overhead |

None of these are portable benchmarks; hardware, driver, thermals, compiler, and model dimensions
all matter, and as the inversion above shows, a change on the *CPU* side can flip the answer. Run
`bench.out` on the target system before deciding whether to enable GPU execution.

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
