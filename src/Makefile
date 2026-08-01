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
# tools/          standalone developer tools unrelated to the CLI
#                 (currently just bench.c).
# tests/core/     tests/gpu/   permanent correctness test suite, split the
#                 same way as the source it tests.

# Source files for the main app.out build.
# backends/gpu/gpu_cuda.c and gpu_matmul.c are always linked in (not
# behind a build flag): they're pure dlopen() at runtime and compile/link
# on any Linux machine, reporting "unavailable" and falling back to CPU
# with zero fuss if no NVIDIA GPU is present - see
# core/transformer.c's dispatch_matmul(). Needs -ldl.
SRC = $(MAIN_SOURCE) \
      cli/tokenizer.c cli/attention.c cli/config.c cli/sampling.c cli/batch.c cli/checkpoint.c cli/cli.c cli/stream.c \
      core/model.c core/tensor_ops.c core/transformer.c core/optimizer.c core/training.c core/serialization.c core/metrics.c \
      backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
OBJ = $(SRC:.c=.o)

TARGET = app.out

# Standalone memory/throughput benchmark (tools/bench.c) - links only the
# model modules it actually needs, not cli/*.o. A separate binary on
# purpose: it has nothing to do with training-run orchestration.
BENCH_SRC = tools/bench.c \
            core/model.c core/tensor_ops.c core/transformer.c core/optimizer.c core/training.c core/serialization.c core/metrics.c \
            backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
BENCH_OBJ = $(BENCH_SRC:.c=.o)
BENCH_TARGET = bench.out

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

bench: $(BENCH_TARGET)

$(BENCH_TARGET): $(BENCH_OBJ)
	$(CC) $(BENCH_OBJ) $(CFLAGS) $(LDFLAGS) -ldl -o $(BENCH_TARGET)

gpu-probe: $(GPU_PROBE_TARGET)

$(GPU_PROBE_TARGET): $(GPU_PROBE_OBJ)
	$(CC) $(GPU_PROBE_OBJ) $(CFLAGS) $(LDFLAGS) -ldl -o $(GPU_PROBE_TARGET)

# Test suite: every tests/{core,gpu}/test_*.c is its own small,
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
TEST_MODEL_OBJ = core/model.o core/tensor_ops.o core/transformer.o core/optimizer.o core/training.o core/serialization.o core/metrics.o \
                 backends/gpu/gpu_cuda.o backends/gpu/gpu_matmul.o cli/sampling.o cli/tokenizer.o
TEST_SRCS = $(wildcard tests/core/test_*.c) $(wildcard tests/cli/test_*.c) $(wildcard tests/gpu/test_*.c) $(wildcard tests/tpu/test_*.c)
TEST_BINS = $(TEST_SRCS:.c=.out)

test: $(LIB) $(TEST_MODEL_OBJ)
	@failed=0; \
	for src in $(TEST_SRCS); do \
		bin=$${src%.c}.out; \
		echo "=== $$src ==="; \
		$(CC) $(CFLAGS) $$src $(TEST_MODEL_OBJ) -L$(LIB_DIR) -l$(LIB_NAME) $(LDFLAGS) -ldl -o $$bin || { echo "BUILD FAILED: $$src"; failed=1; continue; }; \
		./$$bin || failed=1; \
		rm -f $$bin; \
		echo ""; \
	done; \
	if [ $$failed -ne 0 ]; then echo "=== SOME TESTS FAILED ==="; exit 1; else echo "=== ALL TESTS PASSED ==="; fi

clean:
	rm -f $(OBJ) $(BENCH_OBJ) $(GPU_PROBE_OBJ) $(TARGET) $(BENCH_TARGET) $(GPU_PROBE_TARGET) $(TEST_BINS)
	$(MAKE) -C $(LIB_DIR) clean

.PHONY: all clean bench test gpu-probe
