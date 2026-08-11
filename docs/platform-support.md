# Supported platforms

[← Back to README](../README.md)

This document distinguishes configurations continuously exercised by release gates from paths that
are compile-covered or manually measured. “Expected to work” is not used as a synonym for
“supported.”

## Support levels

| Level | Meaning |
|---|---|
| Release-supported | Covered by blocking CI/release workflows, including GCC/Clang, sanitizers, compatibility, and packaging |
| Feature-supported | Correctness tests exist and the path is maintained, but required hardware is not present on hosted release runners |
| Experimental | Source/compile coverage exists; runtime correctness and performance are not continuously established |
| Unsupported | No compatibility promise or release blocking coverage |

## Operating systems and architectures

| Platform | Level | Exact coverage |
|---|---|---|
| Ubuntu 24.04, x86-64, CPU | Release-supported | GCC and Clang full suites; ASan/LSan/UBSan; serial and OpenMP; public libraries and release packages |
| Other 64-bit Linux distributions, x86-64 | Experimental | The runtime uses POSIX/Linux interfaces and should be source-buildable, but no other distribution/libc is a release gate |
| WSL2, x86-64 | Experimental | CPU and NVIDIA paths have been measured during development; no WSL release job exists |
| 64-bit Linux, AArch64 | Experimental | The real NEON source is cross-compiled by `make arm-check`; no AArch64 full-suite runner or performance result exists |
| 32-bit x86/ARM | Unsupported | Model sizes, file offsets, SIMD dispatch, and ABI layouts are not validated |
| macOS, BSD, Windows | Unsupported | The build and shared-library packaging assume Linux/POSIX, ELF, `libdl`, and GNU-compatible linker behavior |

The public header itself is C99 and has `extern "C"` guards. That makes it consumable by C++ on a
supported binary platform; it does not make the library ABI portable between operating systems or
architectures.

## Build toolchain and system interfaces

Release support means the GCC and Clang versions shipped on the pinned `ubuntu-24.04` GitHub runner,
not an untested historical minimum version. The implementation uses GNU/Clang per-function target
attributes for x86 SIMD, POSIX file APIs, `mmap`, and `dlopen`. A normal CPU build needs GNU Make, a
C compiler, libc, `libm`, and `libdl`. The versioned shared-library target additionally needs an
ELF linker supporting `--version-script` and `-z defs`.

`OMP=1` requires an OpenMP runtime. GCC uses libgomp from the normal toolchain; Clang generally
needs `libomp-dev`. The default build is serial and has no OpenMP runtime dependency.

Release packaging requires Git, GNU tar (including sorted entries, numeric ownership, and PAX
options), gzip, SHA-256 utilities, and ordinary POSIX shell tools. Those tools are packaging
requirements, not runtime dependencies of the library.

## CPU execution

The default build deliberately omits `-march=native`. One binary contains the portable kernels and
the applicable architecture-specific kernels, then selects at runtime:

| Runtime path | Architecture and requirements | Selection status |
|---|---|---|
| `baseline` | Portable C for the build target | Always available and the fallback after any unavailable optimized path |
| `avx2` | x86-64 AVX, AVX2, FMA, plus OS-enabled XMM/YMM state in XCR0 | Automatically selected on measured shapes when all checks pass |
| `avx512` | All AVX2 requirements, AVX-512F, AVX-512VL, plus OS-enabled opmask/ZMM state | Automatically selected on measured shapes when all checks pass |
| `neon` | AArch64 Advanced SIMD, mandatory in ARMv8-A | Compiles and has correctness hooks, but remains excluded from automatic selection until measured on AArch64 hardware |

CPUID feature bits alone are not accepted as permission to execute AVX state; the XCR0 checks avoid
an illegal instruction on an OS that does not save wide registers. `DRANZER_CPU_ISA` accepts
`baseline`, `avx2`, `avx512`, or `neon` as a maximum. It can disable wider paths but cannot enable an
ISA absent from the CPU/OS. Invalid values leave hardware selection active and are returned as a
structured configuration diagnostic for the application to render.

`NATIVE=1` is intentionally outside portable release support: it bakes the build host's ISA into
ordinary compiler output and the resulting executable may fault on an older CPU.

## Memory-mapped bundles

Copy loading supports bundle formats 1 through 3 on a host with IEEE-754 binary32. Direct mmap loading
is Linux/POSIX, inference-only, and limited to canonical lossless version-1/version-3 files. It additionally
requires a little-endian host because parameter views point directly at on-disk bytes. The loader
checks all of these conditions and returns `BUNDLE_UNSUPPORTED`/`DRANZER_UNSUPPORTED` rather than
silently copying or reinterpreting data.

## NVIDIA GPU execution

The CUDA wrapper and kernels are compiled into the CLI and public libraries on Linux, but GPU
execution is opt-in (`--gpu` or the internal model flag) and has no CUDA build-time dependency. On
first use it loads `libcuda.so.1` (falling back to `libcuda.so`), initializes device 0, and asks the
installed driver to JIT embedded PTX.

The current PTX declares ISA version 7.0 and target `sm_75`. Consequently, a usable GPU path requires
an NVIDIA device with compute capability 7.5 or newer and a driver that accepts PTX 7.0. The project
does not infer support from a marketing/driver version number: `gpu_probe.out` and
`gpu_matmul_available()` establish whether context creation, PTX JIT, symbol resolution, allocation,
launch, and transfer actually work. Failure at any stage leaves CPU execution available.

The opaque public API version 1 does not expose a GPU-enable control, so its model handles currently
run on the CPU. The optional GPU request is supported by the CLI and internal model API only.

CUDA is feature-supported, not release-runner-supported: GPU correctness tests cover forward,
backward, cache invalidation, repeated training updates, and CPU agreement when hardware is present,
but self-skip on hosted runners without it. The manually measured development device is an NVIDIA
MX450 under WSL2; those timings are evidence for that configuration, not a minimum-performance
promise.

Only matrix multiplication is offloaded. All forward matmuls are eligible; backward input/weight
matmuls use measured work thresholds. Attention scores, softmax, normalization, dropout, optimizer
updates, and all tokenizer/file work remain on the CPU, with activations crossing the host/device
boundary around each offloaded call. Only device 0 and one process-global CUDA context/cache are
used. Multi-GPU, AMD, Intel, Apple GPU, CUDA toolkit/NVRTC compilation, and GPU-resident end-to-end
execution are unsupported.

`DRANZER_GPU_MATMUL` accepts `tiled` (default) or `naive` (comparison baseline). An invalid value
keeps the tiled default and is exposed as a structured diagnostic. OpenCL, DRM, and NVML support in
`gpu_probe.out` is detection/telemetry only; it is not an OpenCL or non-NVIDIA execution backend.

## What blocks a support-level promotion

A platform moves to release-supported only after a pinned runner executes the complete suite and
release gates there. AArch64 additionally needs real-hardware numerical and performance results
before NEON can become an automatic default. A GPU generation needs the existing GPU tests on that
hardware plus fresh crossover measurements; successful PTX JIT alone is not a performance claim.
