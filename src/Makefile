CC = clang
# -ffast-math is portable (no special CPU features required). It was carried
# for years on the claim that it "helps on weak FPUs", with no number beside
# it; the number now exists. Built twice from identical sources with only this
# flag differing, ABBA-interleaved, five paired repeats per tier
# (clang 14, i5-11320H, AVX-512, serial):
#
#   inference ms/token, small tier:  3.02 with, 3.76 without -> 1.25x
#                                    (5/5 paired readings favour the flag;
#                                     the two distributions do not overlap)
#   training  ms/step,  small tier: 15.23 with, 16.40 without -> 1.09x
#                                    (4/5 paired; one reading inverted, so
#                                     this one is not resolved at N=5)
#
# Worth keeping, and worth knowing where it comes from: the win is
# concentrated on inference, which is where core/transformer.c's scalar
# attention loops dominate. The flag is partly compensating for the one hot
# path that has no SIMD kernel. Vectorizing attention should shrink it.
#
# The flag implies -ffinite-math-only, so no code compiled with it may depend
# on infinity or NaN. Two places used to: the causal mask (now a finite 0.0f -
# see attention_head_forward()) and validation code calling isfinite()/isnan()
# (now bit-pattern checks - see include/common/fp_bits.h). Adding a third is a
# correctness bug, not a style question.
#
# Deliberately NOT using -march=native by default: a binary
# built with it only runs on the exact CPU that compiled it and raises
# SIGILL ("illegal instruction") on any other machine - including most
# low-end/older hardware you'd want to deploy this to.
#
# -I include/ makes every #include use a path relative to that root
# (e.g. "core/model.h", "backends/gpu/gpu_cuda.h") regardless of which
# subdirectory the including .c file itself lives in - see include/'s
# layout, which mirrors src/'s (core/, backends/gpu/, cli/, common/).
#
# -MMD -MP emits a .d file beside every .o listing the headers that object
# actually included, and the -include at the bottom of this file feeds them
# back to make. Without it make compares only .c timestamps, so editing a
# header rebuilds nothing: adding a field to neural_model_t (model_types.h)
# left core/model.o with the new struct layout and core/bundle.o and the
# whole of cli/ with the old one, and the resulting binary aborted in
# free() at the end of an eval run. Nothing was wrong with either
# translation unit - they simply disagreed about sizeof(neural_model_t).
# That failure mode costs more to diagnose than the depfiles cost to carry.
CFLAGS = -I include/ -I ../libs/include/ -O3 -ffast-math -MMD -MP
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

# Enable UndefinedBehaviorSanitizer with: make UBSAN=1
#
# Separate from ASAN=1 rather than folded into it. The two catch different
# things - ASan finds memory errors, UBSan finds signed overflow, misaligned
# loads, invalid float-to-int conversions, and shifts past a type's width - and
# UBSan's failure modes are the ones most likely to differ between the compilers
# and instruction sets this project targets, which makes them the ones most
# likely to be mistaken for a numerical result.
#
# -fno-sanitize-recover makes a finding abort instead of printing and continuing,
# so a nightly run cannot be green with diagnostics buried in its log. Float
# division by zero is excluded: -ffast-math already tells the compiler infinities
# do not occur, so a deliberate 1/0 is not what this build is looking for.
ifdef UBSAN
    CFLAGS += -fsanitize=undefined -fno-sanitize=float-divide-by-zero \
              -fno-sanitize-recover=undefined -fno-omit-frame-pointer -g
    LDFLAGS += -fsanitize=undefined
    $(info UndefinedBehaviorSanitizer enabled (UBSAN=1))
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
      core/model.c core/model_params.c core/model_quantize.c core/quantize.c core/quantized_matmul.c core/parallel.c core/matmul.c core/matmul_x86.c core/matmul_arm.c core/cpu_features.c core/tensor_ops.c core/rng.c core/fingerprint.c core/bf16.c core/transformer.c core/matmul_dispatch.c core/lm_head.c core/weight_decay.c core/optimizer.c core/training.c core/evaluation.c core/bundle.c core/serialization.c core/metrics.c \
      backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
OBJ = $(SRC:.c=.o)

TARGET = app.out

# Standalone memory/throughput benchmark (tools/bench.c) - links only the
# model modules it actually needs, not cli/*.o. A separate binary on
# purpose: it has nothing to do with training-run orchestration.
BENCH_SRC = tools/bench.c tools/bench_matmul.c tools/bench_support.c \
            core/model.c core/model_params.c core/model_quantize.c core/quantize.c core/quantized_matmul.c core/parallel.c core/matmul.c core/matmul_x86.c core/matmul_arm.c core/cpu_features.c core/tensor_ops.c core/rng.c core/fingerprint.c core/bf16.c core/transformer.c core/matmul_dispatch.c core/lm_head.c core/weight_decay.c core/optimizer.c core/training.c core/evaluation.c core/serialization.c core/metrics.c \
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

# core/quantize.c measures the error of an operation that overwrites its own
# input, and tests/core/test_quantize.c pins that measurement by requiring the
# in-place and out-of-place forms to agree. Under -ffast-math a compiler may
# vectorize the accumulation loop behind a runtime alias check: the in-place
# call (src == dst) fails that check and runs the scalar fallback, the copied
# call runs the vector path, and the reassociated sum of squares disagrees in
# the last bits. Clang 18 on ubuntu-24.04 does exactly this; clang 14 does not,
# so it appears only in CI. Without reassociation permission an FP reduction
# cannot be vectorized at all, which makes both paths the same arithmetic by
# construction rather than by luck. Nothing here is hot enough to miss the flag.
core/quantize.o: CFLAGS := $(filter-out -ffast-math,$(CFLAGS)) -ffp-contract=off

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
BENCH_QUANT_SRC = tools/bench_quant.c tools/bench_support.c tools/statistics.c \
                  core/model.c core/model_params.c core/model_quantize.c core/quantize.c \
                  core/parallel.c core/matmul.c core/matmul_x86.c core/matmul_arm.c \
                  core/cpu_features.c core/tensor_ops.c core/rng.c core/fingerprint.c core/bf16.c core/transformer.c core/matmul_dispatch.c core/lm_head.c \
                  core/weight_decay.c core/optimizer.c core/training.c core/evaluation.c core/metrics.c \
                  backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
BENCH_QUANT_OBJ = $(BENCH_QUANT_SRC:.c=.o)
BENCH_QUANT_TARGET = bench_quant.out

bench-quant: $(BENCH_QUANT_TARGET)

$(BENCH_QUANT_TARGET): $(BENCH_QUANT_OBJ)
	$(CC) $(BENCH_QUANT_OBJ) $(CFLAGS) $(LDFLAGS) -ldl -o $(BENCH_QUANT_TARGET)

# Weight bandwidth in isolation: identical float32 accumulation with B kept as
# fp32, bf16, INT8, or packed INT4. Every narrow kernel is validated against a
# dequantized reference before timing.
BENCH_QUANT_MATMUL_SRC = tools/bench_quantized_matmul.c tools/bench_support.c \
                         core/quantize.c core/quantized_matmul.c core/bf16.c \
                         core/parallel.c core/matmul.c core/matmul_x86.c \
                         core/matmul_arm.c core/cpu_features.c
BENCH_QUANT_MATMUL_OBJ = $(BENCH_QUANT_MATMUL_SRC:.c=.o)
BENCH_QUANT_MATMUL_TARGET = bench_quantized_matmul.out

bench-quantized-matmul: $(BENCH_QUANT_MATMUL_TARGET)

$(BENCH_QUANT_MATMUL_TARGET): $(BENCH_QUANT_MATMUL_OBJ)
	$(CC) $(BENCH_QUANT_MATMUL_OBJ) $(CFLAGS) $(LDFLAGS) -o $(BENCH_QUANT_MATMUL_TARGET)

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
		$(filter-out -march=native -MMD -MP,$(CFLAGS)) -Wall -Wextra \
		-c core/matmul_arm.c -o /dev/null \
	&& echo "arm-check: core/matmul_arm.c compiles for aarch64"; \
	status=$$?; \
	rm -rf $(ARM_CHECK_STUB); \
	exit $$status

# Where the forward pass actually spends its time, and specifically how much of
# it is the scalar attention core that has no SIMD kernel. Run this before any
# attention kernel work: it measures the ceiling on that work.
BENCH_ATTENTION_SRC = tools/bench_attention.c \
                      core/model.c core/model_params.c core/model_quantize.c core/quantize.c core/quantized_matmul.c \
                      core/parallel.c core/matmul.c core/matmul_x86.c core/matmul_arm.c \
                      core/cpu_features.c core/tensor_ops.c core/rng.c core/fingerprint.c \
                      core/bf16.c core/transformer.c core/matmul_dispatch.c core/lm_head.c \
                      core/weight_decay.c core/optimizer.c core/training.c core/evaluation.c \
                      core/metrics.c backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
BENCH_ATTENTION_OBJ = $(BENCH_ATTENTION_SRC:.c=.o)
BENCH_ATTENTION_TARGET = bench_attention.out

bench-attention: $(BENCH_ATTENTION_TARGET)

$(BENCH_ATTENTION_TARGET): $(BENCH_ATTENTION_OBJ)
	$(CC) $(BENCH_ATTENTION_OBJ) $(CFLAGS) $(LDFLAGS) -ldl -o $(BENCH_ATTENTION_TARGET)

# Assert the invariant -ffast-math imposes: nothing compiled with it may depend
# on infinity or NaN. See tools/check_finite_math.sh for why the compiler's own
# warning cannot be the gate.
finite-math-check:
	@bash tools/check_finite_math.sh

# What a seed and an architecture produce, as one number: the primitive the
# reproducibility contract (docs/reproducibility.md) is checked with. Links the
# model modules and nothing else - no CLI, no GPU, no timing.
FINGERPRINT_SRC = tools/fingerprint_model.c \
                  core/model.c core/model_params.c core/model_quantize.c core/quantize.c core/quantized_matmul.c \
                  core/parallel.c core/matmul.c core/matmul_x86.c core/matmul_arm.c \
                  core/cpu_features.c core/tensor_ops.c core/rng.c core/fingerprint.c \
                  core/bf16.c core/transformer.c core/matmul_dispatch.c core/lm_head.c \
                  core/weight_decay.c core/optimizer.c core/training.c core/evaluation.c core/metrics.c \
                  backends/gpu/gpu_cuda.c backends/gpu/gpu_matmul.c
FINGERPRINT_OBJ = $(FINGERPRINT_SRC:.c=.o)
FINGERPRINT_TARGET = fingerprint_model.out

fingerprint: $(FINGERPRINT_TARGET)

$(FINGERPRINT_TARGET): $(FINGERPRINT_OBJ)
	$(CC) $(FINGERPRINT_OBJ) $(CFLAGS) $(LDFLAGS) -ldl -o $(FINGERPRINT_TARGET)

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
# tools/statistics.o rides along in the test object list because
# tests/core/test_statistics.c needs it and the test target links one fixed set
# of objects into every test binary. It is measurement infrastructure rather than
# model code, hence tools/ - see include/tools/statistics.h.
TEST_MODEL_OBJ = tools/statistics.o core/model.o core/model_params.o core/model_quantize.o core/quantize.o core/quantized_matmul.o core/parallel.o core/matmul.o core/matmul_x86.o core/matmul_arm.o core/cpu_features.o core/tensor_ops.o core/rng.o core/fingerprint.o core/bf16.o core/transformer.o core/matmul_dispatch.o core/lm_head.o core/weight_decay.o core/optimizer.o core/training.o core/evaluation.o core/bundle.o core/serialization.o core/metrics.o \
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
	@bash tests/integration/test_determinism.sh ./$(TARGET)
	@$(MAKE) --no-print-directory $(FINGERPRINT_TARGET)
	@bash tests/integration/test_libc_independence.sh

# Every .d generated by -MMD, matching the .o list it was derived from.
DEPS = $(OBJ:.o=.d) $(BENCH_OBJ:.o=.d) $(BENCH_PARALLEL_OBJ:.o=.d) \
       $(BENCH_QUANT_OBJ:.o=.d) $(BENCH_QUANT_MATMUL_OBJ:.o=.d) \
       $(GPU_PROBE_OBJ:.o=.d) $(GPU_LATENCY_OBJ:.o=.d) \
       $(FINGERPRINT_OBJ:.o=.d) $(BENCH_ATTENTION_OBJ:.o=.d) \
       $(TEST_MODEL_OBJ:.o=.d)

clean:
	rm -f $(OBJ) $(BENCH_OBJ) $(BENCH_PARALLEL_OBJ) $(BENCH_QUANT_OBJ) $(BENCH_QUANT_MATMUL_OBJ) $(GPU_PROBE_OBJ) $(GPU_LATENCY_OBJ) $(TARGET) $(BENCH_TARGET) $(BENCH_PARALLEL_TARGET) $(BENCH_QUANT_TARGET) $(BENCH_QUANT_MATMUL_TARGET) $(PROFILE_TARGET) $(GPU_PROBE_TARGET) $(GPU_LATENCY_TARGET) $(FINGERPRINT_TARGET) $(BENCH_ATTENTION_TARGET) $(TEST_BINS)
	rm -f $(DEPS)
	$(MAKE) -C $(LIB_DIR) clean

# Header dependencies discovered by -MMD. The leading dash keeps a missing
# .d (first build, or straight after clean) from being an error.
-include $(DEPS)

.PHONY: all clean bench bench-parallel bench-quant bench-quantized-matmul profile test gpu-probe gpu-latency arm-check finite-math-check fingerprint bench-attention
