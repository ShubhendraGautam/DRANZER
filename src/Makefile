CC = clang
# -ffast-math is portable (no special CPU features required) and helps on
# weak FPUs. Deliberately NOT using -march=native by default: a binary
# built with it only runs on the exact CPU that compiled it and raises
# SIGILL ("illegal instruction") on any other machine - including most
# low-end/older hardware you'd want to deploy this to.
#
# -I include/ makes every #include use a path relative to that root
# (e.g. "core/model.h", "backends/gpu/gpu_cuda.h") regardless of which
# subdirectory the including .c file itself lives in - see include/'s
# layout, which mirrors src/'s (core/, backends/gpu/, cli/, common/).
CFLAGS = -I include/ -I ../libs/include/ -O3 -ffast-math
LDFLAGS = -lm

# Tune for the exact CPU you're building on: make NATIVE=1
# Fastest option, but the resulting binary is NOT portable to other machines.
ifdef NATIVE
    CFLAGS += -march=native
    $(info Native CPU tuning enabled (NATIVE=1) - binary will only run on this build machine's CPU)
endif

# Optimize for code size instead of speed: make SIZE=1
# Useful on storage- or icache-constrained low-end/embedded devices.
ifdef SIZE
    CFLAGS := $(filter-out -O3,$(CFLAGS)) -Os
    $(info Size-optimized build enabled (SIZE=1))
endif

# Enable debug output with: make DEBUG=1
ifdef DEBUG
    CFLAGS += -g -DDEBUG
    $(info DEBUG mode enabled)
endif

# Enable AddressSanitizer with: make ASAN=1
ifdef ASAN
    CFLAGS += -fsanitize=address -fno-omit-frame-pointer -O1 -g
    $(info AddressSanitizer enabled)
endif

# Enable multi-threaded matmul/attention with: make OMP=1
# Parallelizes across independent output rows/heads, so results are
# bit-for-bit identical to a serial build (no cross-thread reductions).
# clang needs libomp-dev installed for this; if you don't have it, build
# with gcc instead: make OMP=1 CC=gcc
ifdef OMP
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
    $(info OpenMP parallelization enabled (OMP=1))
endif

# Main source - unified neural model implementation
MAIN_SOURCE = cli/main.c

# --- Directory layout ---
# core/           backend-agnostic model logic (math primitives, forward/
#                 backward pass, optimizer, training loop, serialization,
#                 metrics) - the same code runs regardless of which
#                 backend (CPU/GPU/TPU) is doing the matmuls.
# backends/gpu/   CUDA-specific code: the driver API wrapper, the
#                 hand-written PTX matmul kernel, and the whole gpu_probe
#                 capability-detection/benchmarking tool.
# backends/tpu/   reserved for a future TPU backend - not implemented yet.
# cli/            the train/infer/generate command-line application:
#                 argument parsing, config files, sampling, checkpoints,
#                 the streaming tokenizer, and main() itself.
# tools/          standalone developer tools unrelated to the CLI: the
#                 throughput/memory benchmark, the isolated matmul kernel
#                 comparison, and their shared provenance helpers.
# tests/core/     tests/gpu/   permanent correctness test suite, split the
#                 same way as the source it tests.

# Source files for the main app.out build.
# backends/gpu/gpu_cuda.c and gpu_matmul.c are always linked in (not
# behind a build flag): they're pure dlopen() at runtime and compile/link
# on any Linux machine, reporting "unavailable" and falling back to CPU
# with zero fuss if no NVIDIA GPU is present - see
# core/transformer.c's dispatch_matmul(). Needs -ldl.
SRC = $(MAIN_SOURCE) \
      cli/tokenizer.c cli/evaluation.c cli/generation.c cli/attention.c cli/config.c cli/manifest.c cli/sampling.c cli/batch.c cli/checkpoint.c cli/cli.c cli/stream.c \
      core/model.c core/model_params.c core/model_quantize.c core/quantize.c core/parallel.c core/matmul.c core/matmul_x86.c core/matmul_arm.c core/cpu_features.c core/tensor_ops.c core/transformer.c core/optimizer.c core/training.c core/evaluation.c core/bundle.c core/serialization.c core/metrics.c \
      backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
OBJ = $(SRC:.c=.o)

TARGET = app.out

# Standalone memory/throughput benchmark (tools/bench.c) - links only the
# model modules it actually needs, not cli/*.o. A separate binary on
# purpose: it has nothing to do with training-run orchestration.
BENCH_SRC = tools/bench.c tools/bench_matmul.c tools/bench_support.c \
            core/model.c core/model_params.c core/model_quantize.c core/quantize.c core/parallel.c core/matmul.c core/matmul_x86.c core/matmul_arm.c core/cpu_features.c core/tensor_ops.c core/transformer.c core/optimizer.c core/training.c core/evaluation.c core/serialization.c core/metrics.c \
            backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
BENCH_OBJ = $(BENCH_SRC:.c=.o)
BENCH_TARGET = bench.out
PROFILE_TARGET = bench_profile.out

# Preserve the exact compile/link command in benchmark metadata. Single
# quotes are consumed by the shell; the embedded double quotes become part
# of the -D string literal seen by C.
BENCH_METADATA_FLAGS = -DDRANZER_BUILD_COMMAND='"$(CC) $(CFLAGS) $(LDFLAGS)"'

# GPU capability probe (dlopen-based DRM/NVML/CUDA/OpenCL detection) -
# Linux-only, needs libdl but nothing else (no CUDA/OpenCL SDK required to
# build or run it - that's the point). Its own binary: unrelated to the
# model or CLI.
GPU_PROBE_SRC = backends/gpu/gpu_probe.c backends/gpu/gpu_backend_drm.c backends/gpu/gpu_backend_nvml.c \
                backends/gpu/gpu_backend_cuda.c backends/gpu/gpu_backend_opencl.c backends/gpu/gpu_capability_cache.c \
                backends/gpu/gpu_cuda.c backends/gpu/gpu_cuda_attrs.c backends/gpu/gpu_nvml_telemetry.c \
                backends/gpu/gpu_arch_specs.c backends/gpu/gpu_theoretical_perf.c backends/gpu/gpu_microbench.c
GPU_PROBE_OBJ = $(GPU_PROBE_SRC:.c=.o)
GPU_PROBE_TARGET = gpu_probe.out

# Library
LIB_DIR = ../libs
LIB_NAME = attention
LIB = $(LIB_DIR)/lib$(LIB_NAME).a

all: $(LIB) $(TARGET)

# Step 1: Build library first
$(LIB):
	$(MAKE) DEBUG=$(DEBUG) ASAN=$(ASAN) -C $(LIB_DIR)

# Step 2: Link everything
$(TARGET): $(OBJ) $(LIB)
	$(CC) $(OBJ) -L$(LIB_DIR) -l$(LIB_NAME) $(CFLAGS) $(LDFLAGS) -ldl -o $(TARGET)

# Step 3: Compile src files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# bench_support.c is the only file that needs the recorded build command,
# but bench.o is listed too so touching either rebuilds with it.
tools/bench_support.o: tools/bench_support.c
	$(CC) $(CFLAGS) $(BENCH_METADATA_FLAGS) -c $< -o $@

tools/bench.o: tools/bench.c
	$(CC) $(CFLAGS) $(BENCH_METADATA_FLAGS) -c $< -o $@

bench: $(BENCH_TARGET)

$(BENCH_TARGET): $(BENCH_OBJ)
	$(CC) $(BENCH_OBJ) $(CFLAGS) $(LDFLAGS) -ldl -o $(BENCH_TARGET)

# Frame-pointer build for Linux perf. Use bench tier selection to keep a
# profiling session bounded, e.g. `perf record -g -- ./bench_profile.out
# --tier small`. This target compiles directly to avoid reusing objects made
# with incompatible optimization/debug flags.
profile:
	$(CC) $(CFLAGS) -g -fno-omit-frame-pointer \
		-DDRANZER_BUILD_COMMAND='"$(CC) $(CFLAGS) -g -fno-omit-frame-pointer $(LDFLAGS)"' \
		$(BENCH_SRC) $(LDFLAGS) -ldl -o $(PROFILE_TARGET)
	@echo "Profile build ready: perf record -g -- ./$(PROFILE_TARGET) --tier small"

# What weight-only quantization costs in accuracy, measured in weight space,
# logit space, and held-out cross-entropy - the tool docs/quantization.md is
# written from. Links the model modules because levels 2 and 3 need a real
# forward pass, but no CLI and no GPU: the question is about values, not
# orchestration or hardware.
BENCH_QUANT_SRC = tools/bench_quant.c tools/bench_support.c \
                  core/model.c core/model_params.c core/model_quantize.c core/quantize.c \
                  core/parallel.c core/matmul.c core/matmul_x86.c core/matmul_arm.c \
                  core/cpu_features.c core/tensor_ops.c core/transformer.c \
                  core/optimizer.c core/training.c core/evaluation.c core/metrics.c \
                  backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
BENCH_QUANT_OBJ = $(BENCH_QUANT_SRC:.c=.o)
BENCH_QUANT_TARGET = bench_quant.out

bench-quant: $(BENCH_QUANT_TARGET)

$(BENCH_QUANT_TARGET): $(BENCH_QUANT_OBJ)
	$(CC) $(BENCH_QUANT_OBJ) $(CFLAGS) $(LDFLAGS) -ldl -o $(BENCH_QUANT_TARGET)

# What entering an OpenMP parallel region costs, and where it starts paying
# for itself - the tool docs/threading.md is written from, and where
# core/parallel.c's threshold comes from. Needs OMP=1 to measure anything;
# without it the binary builds and says so. Links no model code: like the
# matmul sweep, it exists to measure one primitive in isolation.
BENCH_PARALLEL_SRC = tools/bench_parallel.c tools/bench_support.c \
                     core/cpu_features.c core/parallel.c core/matmul.c \
                     core/matmul_x86.c core/matmul_arm.c
BENCH_PARALLEL_OBJ = $(BENCH_PARALLEL_SRC:.c=.o)
BENCH_PARALLEL_TARGET = bench_parallel.out

bench-parallel: $(BENCH_PARALLEL_TARGET)

$(BENCH_PARALLEL_TARGET): $(BENCH_PARALLEL_OBJ)
	$(CC) $(BENCH_PARALLEL_OBJ) $(CFLAGS) $(LDFLAGS) -pthread -o $(BENCH_PARALLEL_TARGET)

# GPU per-call latency probe: what a gpu_matmul() call costs once device
# buffers are already resident. Links the CUDA wrapper and matmul backend
# plus the shared benchmark provenance/timing helpers - no model code.
GPU_LATENCY_SRC = tools/gpu_latency.c tools/bench_support.c core/cpu_features.c \
                  core/parallel.c core/matmul.c core/matmul_x86.c core/matmul_arm.c \
                  backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
GPU_LATENCY_OBJ = $(GPU_LATENCY_SRC:.c=.o)
GPU_LATENCY_TARGET = gpu_latency.out

gpu-latency: $(GPU_LATENCY_TARGET)

$(GPU_LATENCY_TARGET): $(GPU_LATENCY_OBJ)
	$(CC) $(GPU_LATENCY_OBJ) $(CFLAGS) $(LDFLAGS) -ldl -o $(GPU_LATENCY_TARGET)

# Compile the NEON kernel for AArch64 without needing an AArch64 machine.
#
# core/matmul_arm.c is behind `#ifdef DRANZER_HAVE_NEON`, so on x86 - which is
# every developer machine and every CI runner this project uses - its body is
# never compiled at all. A syntax error or a bad intrinsic in it would ship
# undetected. This target type-checks the real thing.
#
# clang carries backends and an arm_neon.h for every target it supports, so
# only libc headers are missing; -nostdlibinc drops the host's and the stub
# below supplies the one declaration the file needs. Object code is discarded -
# the point is the compile, not the artifact.
#
# Skips cleanly when clang has no aarch64 backend, so `make arm-check` is safe
# to run anywhere.
ARM_CHECK_STUB = .arm-check-stub

arm-check:
	@if ! clang --target=aarch64-linux-gnu -E -x c /dev/null >/dev/null 2>&1; then \
		echo "arm-check: clang has no aarch64 target here - skipping"; \
		exit 0; \
	fi; \
	mkdir -p $(ARM_CHECK_STUB); \
	printf '#ifndef STUB_STRING_H\n#define STUB_STRING_H\n#include <stddef.h>\nvoid *memset(void *s, int c, size_t n);\n#endif\n' > $(ARM_CHECK_STUB)/string.h; \
	clang --target=aarch64-linux-gnu -nostdlibinc -I $(ARM_CHECK_STUB) -I include/ \
		$(filter-out -march=native,$(CFLAGS)) -Wall -Wextra \
		-c core/matmul_arm.c -o /dev/null \
	&& echo "arm-check: core/matmul_arm.c compiles for aarch64"; \
	status=$$?; \
	rm -rf $(ARM_CHECK_STUB); \
	exit $$status

gpu-probe: $(GPU_PROBE_TARGET)

$(GPU_PROBE_TARGET): $(GPU_PROBE_OBJ)
	$(CC) $(GPU_PROBE_OBJ) $(CFLAGS) $(LDFLAGS) -ldl -o $(GPU_PROBE_TARGET)

# Test suite: every tests/{core,cli,gpu}/test_*.c is its own small,
# independent correctness check (gradient checks, optimizer/schedule/
# dropout sanity, serialization roundtrip, GPU-vs-CPU comparisons) - see
# tests/ for what each one verifies and why. Each is built and run in
# turn; a single failing test doesn't stop the rest from running, and the
# target exits non-zero overall if any failed.
# backends/gpu/gpu_cuda.o and gpu_matmul.o are always linked into the test
# binaries (not just under GPU=1): they're pure dlopen() at runtime, so
# they compile and link on any Linux machine and simply report
# "unavailable" if no NVIDIA driver is present - see
# tests/gpu/test_gpu_matmul.c's self-skip. Needs -ldl.
TEST_MODEL_OBJ = core/model.o core/model_params.o core/model_quantize.o core/quantize.o core/parallel.o core/matmul.o core/matmul_x86.o core/matmul_arm.o core/cpu_features.o core/tensor_ops.o core/transformer.o core/optimizer.o core/training.o core/evaluation.o core/bundle.o core/serialization.o core/metrics.o \
                 backends/gpu/gpu_cuda.o backends/gpu/gpu_matmul.o cli/batch.o cli/checkpoint.o cli/cli.o cli/config.o cli/evaluation.o cli/generation.o cli/sampling.o cli/stream.o cli/tokenizer.o
# tests/perf/ holds timing tests. They assert ratios between measurements
# taken in the same process, never absolute times, so a slow or busy machine
# scales both sides and the assertion holds - see the file header for why
# that makes them safe in an ordinary suite.
TEST_SRCS = $(wildcard tests/core/test_*.c) $(wildcard tests/cli/test_*.c) $(wildcard tests/gpu/test_*.c) $(wildcard tests/perf/test_*.c) $(wildcard tests/tpu/test_*.c)
TEST_BINS = $(TEST_SRCS:.c=.out)

# tests/gpu/ binaries run with LeakSanitizer off (ASAN's buffer-overflow and
# use-after-free checks stay on). The NVIDIA driver keeps process-global
# allocations that outlive cuCtxDestroy, and LSan cannot attribute most of
# them to any module, so leak-checking a process that has loaded libcuda
# reports ~49 KB of driver state on every GPU test. Suppressing by library
# name only caught half of it. This is scoped to tests/gpu/ so every other
# test still gets full leak detection.
test: $(LIB) $(TARGET) $(TEST_MODEL_OBJ)
	@failed=0; \
	for src in $(TEST_SRCS); do \
		bin=$${src%.c}.out; \
		echo "=== $$src ==="; \
		$(CC) $(CFLAGS) $$src $(TEST_MODEL_OBJ) -L$(LIB_DIR) -l$(LIB_NAME) $(LDFLAGS) -ldl -o $$bin || { echo "BUILD FAILED: $$src"; failed=1; continue; }; \
		case $$src in \
			tests/gpu/*) ASAN_OPTIONS=detect_leaks=0 ./$$bin || failed=1 ;; \
			*) ./$$bin || failed=1 ;; \
		esac; \
		rm -f $$bin; \
		echo ""; \
	done; \
	if [ $$failed -ne 0 ]; then echo "=== SOME TESTS FAILED ==="; exit 1; else echo "=== ALL TESTS PASSED ==="; fi
	@bash tests/integration/test_eval_cli.sh ./$(TARGET)
	@bash tests/integration/test_resume_cli.sh ./$(TARGET)

clean:
	rm -f $(OBJ) $(BENCH_OBJ) $(BENCH_PARALLEL_OBJ) $(BENCH_QUANT_OBJ) $(GPU_PROBE_OBJ) $(GPU_LATENCY_OBJ) $(TARGET) $(BENCH_TARGET) $(BENCH_PARALLEL_TARGET) $(BENCH_QUANT_TARGET) $(PROFILE_TARGET) $(GPU_PROBE_TARGET) $(GPU_LATENCY_TARGET) $(TEST_BINS)
	$(MAKE) -C $(LIB_DIR) clean

.PHONY: all clean bench bench-parallel bench-quant profile test gpu-probe gpu-latency arm-check
