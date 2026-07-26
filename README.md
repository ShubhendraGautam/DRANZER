# DRANZER — A Transformer Built From Scratch in C

DRANZER is a decoder-only transformer language model implemented entirely in C - no ML framework,
no autodiff library, no GPU SDK required to build or run it. Every piece of the standard deep
learning stack is hand-written and independently verified rather than assumed correct: multi-head
causal self-attention, full backpropagation through every layer (checked against numerical
gradients, not just "trains without crashing"), AdamW with gradient clipping and a
warmup-then-cosine learning-rate schedule, dropout, and OpenMP-parallelized matmul/attention with
results proven bit-identical to the serial path. It even runs on a GPU without a CUDA toolkit -
hand-written PTX assembly, loaded and JIT-compiled directly through the CUDA driver.

The codebase is split into focused, single-responsibility modules (see Project Structure) with a
permanent regression test suite (`make test`) and a hardware benchmarking/capability-probing
toolchain (`make bench`, `make gpu-probe`) to characterize exactly what a given machine - from a
Raspberry Pi to a CUDA GPU - can actually run.

## 🎯 Project Phases

### Phase 1: Attention Core
- ✅ Multi-head self-attention mechanism (4 parallel attention heads)
- ✅ Xavier weight initialization
- ✅ Residual connections
- ✅ Next-token prediction and inference

### Phase 2: Stability & Learning
- ✅ Layer normalization (learnable γ and β parameters)
- ✅ Adaptive learning rate scheduling
- ✅ Loss tracking and learning metrics
- ✅ Feedforward networks with ReLU activation

### Phase 3: Production & Scale
- ✅ Configuration file management (JSON-compatible)
- ✅ Training checkpoints (resumable training)
- ✅ 4 sampling strategies: greedy, top-k, top-p, beam search
- ✅ Batch processing infrastructure
- ✅ **Command-line interface for all modes**

### Phase 4: Research-Grade Rigor
- ✅ **Stacked multi-layer transformer** with full hand-rolled backpropagation (verified via numerical gradient checking, not just "trains without crashing")
- ✅ **Causal attention masking** - each position only attends to itself and the past, as in a real decoder-only transformer
- ✅ **AdamW optimizer** (default) with global gradient-norm clipping and decoupled weight decay; plain SGD still available via `--optimizer sgd`
- ✅ **Linear-warmup + cosine-decay LR schedule** (opt-in via `--warmup-steps`/`--total-steps`), falling back to plateau decay when unconfigured
- ✅ **Dropout** on both sub-layer outputs (attention and FFN), with inverted scaling so inference needs no rescaling
- ✅ **OpenMP parallelization** (`make OMP=1 CC=gcc`) of every matmul/attention hot loop, with zero change to results (no cross-thread reductions - verified bit-identical against the serial build)
- ✅ **Permanent test suite** (`make test`): numerical gradient checks, dropout/optimizer/schedule correctness, save-load roundtrip
- ✅ **Standalone benchmark tool** (`make bench`): memory footprint and inference/training throughput across model sizes
- ✅ **GPU capability probe** (`make gpu-probe`): detects what this machine can actually do for GPU compute (DRM/NVML/CUDA/OpenCL) via `dlopen()` against whatever's already installed - no CUDA toolkit or OpenCL SDK required just to ask the question
- ✅ **GPU matmul, zero-SDK-dependency** (`gpu_matmul.c`): a hand-written PTX kernel, loaded and JIT-compiled by the CUDA driver itself via `dlopen("libcuda.so.1")` - no CUDA toolkit, no NVRTC, nothing beyond the driver library any NVIDIA install already has. Validated against `tensor_ops.c`'s CPU `matrix_multiply()` in `tests/test_gpu_matmul.c` (self-skips, not fails, on machines without a CUDA GPU)
- ✅ **Modular architecture** - the model is split across focused single-responsibility files (see Project Structure) instead of one monolithic `model.c`

## 📁 Project Structure

```
attention_in_c/
├── README.md                   # This file
├── Makefile                    # Top-level build coordinator
├── test.txt                    # Sample training data
│
├── libs/                       # Core utilities
│   ├── Makefile
│   ├── include/
│   │   ├── hashmap.h          # Type-aware hash table
│   │   ├── byte_pair_encoding.h
│   │   └── debug.h            # DEBUG_PRINT macro
│   └── src/
│       ├── hashmap.c
│       └── byte_pair_encoding.c
│
├── src/                        # Main application (Phase 1-4)
│   ├── Makefile                # app.out, bench.out, and test targets
│   ├── main.c                  # 3 modes: train, infer, generate
│   ├── model.c/h                # Model lifecycle only: model_new/model_free (flat params/grads buffer)
│   ├── tensor_ops.c/h            # Pure math: matmul, softmax, layer norm, dropout, positional encoding
│   ├── transformer.c/h           # Multi-head attention (causally masked) + the stacked forward pass
│   ├── optimizer.c/h              # SGD, AdamW, gradient clipping, warmup+cosine LR schedule
│   ├── training.c/h                # model_train_step's backward-pass orchestration
│   ├── serialization.c/h           # model_save/model_load, shared write/read state
│   ├── metrics.c/h                  # Loss-history accessors
│   ├── bench.c                       # Standalone memory/throughput benchmark (make bench)
│   ├── gpu_probe.c                    # GPU capability probe entry point (make gpu-probe)
│   ├── gpu_backend_drm.c/h             # /dev/dri device node presence (vendor-agnostic)
│   ├── gpu_backend_nvml.c/h            # NVIDIA driver-level profiling (name/VRAM/compute capability)
│   ├── gpu_backend_cuda.c/h            # CUDA driver API + nvcc/NVRTC compiler availability
│   ├── gpu_backend_opencl.c/h          # OpenCL ICD loader + registered vendor platforms/devices
│   ├── gpu_cuda.c/h                    # CUDA Driver API wrapper (dlopen-based; no CUDA toolkit needed)
│   ├── gpu_matmul.c/h                  # Hand-written PTX matmul kernel, loaded through gpu_cuda.h
│   ├── gpu_capability_cache.c/h         # Saves/loads a probed GPU's capability profile (see below)
│   ├── gpu_cuda_attrs.c/h                # Table-driven cuDeviceGetAttribute sweep (see below)
│   ├── tests/                        # Permanent correctness test suite (make test)
│   │   ├── test_gradient_check.c         # Numerical gradient check, 18 params across both layers
│   │   ├── test_dropout_gradient_check.c # Same, with dropout active (deterministic mask)
│   │   ├── test_dropout_statistics.c     # Drop-rate/mean-preservation/inference-noop checks
│   │   ├── test_adam_convergence.c       # AdamW actually converges + lazy moment-buffer allocation
│   │   ├── test_lr_schedule.c            # Warmup+cosine shape check
│   │   └── test_serialization_roundtrip.c # Save/load produces identical logits
│   ├── tokenizer.c/h           # BPE tokenizer API
│   ├── attention.c/h           # Legacy single-head self-attention demo (unrelated to model.c/transformer.c)
│   ├── cli.c/h                # Command-line argument parsing
│   ├── config.c/h             # Configuration management
│   ├── sampling.c/h           # Sampling strategies
│   ├── batch.c/h              # Batch processing
│   ├── checkpoint.c/h         # Training checkpoints
│   ├── include/debug.h        # Debug utilities
│   └── app.out                # Compiled executable
│
├── config.json                 # Training configuration
├── dranzer.pth                 # Latest model weights
├── checkpoints/                # Training checkpoint history
│   ├── checkpoint_epoch_1_step_100.ckpt
│   └── config.txt
│
└── .agents.md, AGENTS.md, SKILL.md    # AI assistant documentation
```

## 🔧 Dependencies

- **Compiler**: clang (C11 support)
- **Libraries**: libc, libm (standard math)
- **Build**: GNU Make

### Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install clang make build-essential
```

**macOS:**
```bash
brew install clang make
```

## 🏗️ Building

### Quick Build

```bash
cd src
make
```

Produces: `src/app.out` (executable)

### Build Options

**Debug Mode** (enables DEBUG_PRINT statements):
```bash
make DEBUG=1
```

**AddressSanitizer** (memory error detection):
```bash
make ASAN=1
```

**Combined:**
```bash
make DEBUG=1 ASAN=1
```

**Low-end/small-platform builds:**
```bash
make SIZE=1              # -Os instead of -O3: smaller binary, storage/icache-constrained devices
make NATIVE=1            # -march=native: fastest, but the binary only runs on this exact CPU
```
The default build deliberately avoids `-march=native` so the binary stays portable to older/weaker hardware instead of raising `SIGILL` on anything but the machine it was compiled on.

**Multi-threaded build:**
```bash
make OMP=1 CC=gcc         # parallelizes matmul/attention across cores; clang needs libomp-dev, gcc works out of the box
```
Every matmul/attention hot loop is parallelized over independent output regions (no cross-thread reductions), so an OMP=1 build produces bit-identical results to the serial build - only wall-clock time changes.

**Clean Build Artifacts:**
```bash
make clean
```

### GPU Capability Probe

```bash
cd src
make gpu-probe
./gpu_probe.out
```
Reports what this machine can actually do for GPU compute, right now - not just "is a GPU present" but which of DRM/NVML/CUDA/OpenCL are usable and what's missing for the rest, with a concrete recommendation. Every check works via `dlopen()`/`dlsym()` against whatever runtime libraries already exist, declaring the small set of stable-ABI prototypes/structs it needs itself - so this tool needs neither the CUDA toolkit nor OpenCL dev headers to build or run. That distinction matters more than it sounds: a driver library being present (`cuInit()`/`nvmlInit()` succeeding) says nothing about whether you can actually get a kernel compiled and onto the GPU - `gpu_probe.out` checks that separately (nvcc/NVRTC for CUDA, a registered vendor ICD for OpenCL) so it can't report a false "ready".

Example output on a machine with an NVIDIA GPU but no CUDA toolkit or OpenCL vendor ICD installed:
```
--- NVML (driver-level GPU profiling): READY ---
    NVIDIA driver version: 581.95
    GPU count: 1
    [0] NVIDIA GeForce MX450 - 2048 MB total / 1906 MB free VRAM, compute capability 7.5, utilization 0%

--- CUDA: PARTIAL ---
    CUDA driver API: functional (version 13.0, 1 device(s) visible)
    nvcc NOT found in PATH -> cannot compile .cu kernels at build time.
    libnvrtc.so NOT found -> no runtime compilation path either.

--- OpenCL: PARTIAL ---
    OpenCL ICD loader (libOpenCL.so): functional
    Registered vendor ICDs (/etc/OpenCL/vendors/): none
    clGetPlatformIDs() found 0 platforms (rc=-1001) - the loader works, but no vendor
    OpenCL implementation is registered, so there is nothing to run on yet.
```
Note: `nvmlInit()`/`cuInit()` leave small fixed-size allocations inside the vendor driver library's own internals (visible under ASAN as a "leak" attributed to `libnvidia-ml.so`/`libcuda.so`, not to any file in this project) - `gpu_probe.c` and its backends call `dlclose`/`nvmlShutdown` on every path; this is a known characteristic of the closed-source driver, not something fixable from application code.

Linux-only (relies on `/dev`, `/proc`, `/etc` paths and `<dlfcn.h>`) - no Windows/macOS support attempted.

### GPU Matmul (hand-written PTX, zero SDK dependency)

`gpu_matmul()` (`gpu_matmul.c`) runs matrix multiplication on an NVIDIA GPU with **no CUDA toolkit
and no NVRTC installed** - the only runtime dependency is `libcuda.so.1`, which ships with any
NVIDIA driver. This is possible because the CUDA Driver API can load and JIT-compile **PTX**
(NVIDIA's stable, documented GPU assembly language) directly from a string via
`cuModuleLoadData()` - `gpu_matmul.c` embeds a hand-written PTX kernel as a C string literal, the
same way the OpenCL probe treats OpenCL C kernels as strings, just one level lower (real assembly,
not a C-like kernel language).

```c
#include "gpu_matmul.h"

if (gpu_matmul_available()) {
    gpu_matmul(A, B, C, m, k, n);   // same signature as tensor_ops.h's matrix_multiply()
} else {
    matrix_multiply(A, B, C, m, k, n);   // CPU fallback
}
```

`gpu_cuda.c`/`gpu_cuda.h` is the reusable low-level piece underneath it - a minimal CUDA Driver API
wrapper (`dlopen`, context/module/memory/launch) that any future PTX kernel (softmax, layer norm,
attention) would go through, the same way every CPU primitive goes through `tensor_ops.c`.

This backend is **NVIDIA-only** - it deliberately trades cross-vendor portability (which an OpenCL
backend would have) for zero SDK dependency and a stable existence-proof that works today, on this
project's own test hardware, without waiting on a vendor to register an OpenCL ICD. It is not yet
wired into `model_forward`/`model_train_step` - at this project's typical model sizes (see
Benchmarking above), per-call host↔device transfer overhead can easily outweigh a single matmul's
compute cost, so that integration needs its own measurement before it's a clear win, not just a
capability to turn on.

`make test` builds and runs `tests/test_gpu_matmul.c` (which validates `gpu_matmul()` against the
CPU reference) on every machine - `gpu_cuda.c`/`gpu_matmul.c` compile and link everywhere (pure
`dlopen`, no CUDA headers needed), and the test self-skips (reports success, not failure) if no
CUDA GPU is present at runtime.

### GPU Capability Cache (reusable, shareable probe results)

Every field `gpu_probe.out` reports comes from a **publicly documented** API - NVML, the CUDA
Driver API, or OpenCL. This project deliberately does not go looking for undocumented vendor
debug/internal entry points: those come with no ABI stability guarantee and are very likely a
breach of the driver's EULA, neither of which belongs in a project aiming to be reproducible and
redistributable.

What *is* fair game - and genuinely useful - is caching what a probe finds so it doesn't need to
be rediscovered every time. Every run of `gpu_probe.out` saves its findings to
`gpu_capability_cache/<gpu_model>_<driver_version>.cache` (simple `key = value` text, same style as
`config.c`):

```bash
./gpu_probe.out                                                          # probes + saves
./gpu_probe.out --load gpu_capability_cache/NVIDIA_GeForce_MX450_581.95.cache   # reuse without re-probing
```

That file is plain text and portable - hand it to another researcher with the same GPU model and
driver version, and they can skip straight to `gpu_probe.out --load` instead of re-probing their
own machine.

### Automated Device Attribute Sweep

Rather than hand-writing a probe function per GPU property, `gpu_cuda_attrs.c` loops a table of
documented `cuDeviceGetAttribute()` codes through that **single** stable CUDA Driver API call -
adding coverage means adding a row to the table, not a new hand-written probe. `gpu_probe.out` runs
this automatically whenever CUDA is `READY`:

```
--- CUDA device attributes (table-driven cuDeviceGetAttribute sweep) ---
    MAX_THREADS_PER_BLOCK          = 1024
    WARP_SIZE                      = 32
    MULTIPROCESSOR_COUNT           = 14
    CLOCK_RATE_KHZ                 = 1575000
    COMPUTE_CAPABILITY_MAJOR       = 7
    COMPUTE_CAPABILITY_MINOR       = 5
    ...
    (17/17 attributes resolved)
```

The table is deliberately conservative - only attribute codes from the long-stable, widely
verified core of the `CUdevice_attribute` enum. Getting a numeric code wrong wouldn't crash
anything (`cuDeviceGetAttribute` just errors on a code it doesn't recognize), but it could silently
report a real value under the *wrong* label if the wrong code happens to be valid for a different
attribute - so before trusting this table, its output was cross-checked against independently
known values: `WARP_SIZE` and `MAX_THREADS_PER_BLOCK` against universal NVIDIA constants (32 and
1024 on any modern GPU), and `COMPUTE_CAPABILITY_MAJOR`/`MINOR` against the value NVML had
*already* reported independently - they matched exactly (7/5 both ways). Extending this table
further should be done the same way: verify against NVIDIA's published `cuda.h`/Driver API
reference (or cross-check against another source, like NVML), not from memory.

### Testing

```bash
cd src
make test
```
Builds and runs every file in `src/tests/` (each is an independent, focused check - see Project Structure above for what each one verifies) and exits non-zero if any fail. This is the project's correctness backstop: with hand-rolled backprop and no autodiff to lean on, a subtly wrong backward pass would otherwise produce a model that trains without actually learning correctly.

### Benchmarking

```bash
cd src
make bench
./bench.out               # or: make bench OMP=1 CC=gcc && OMP_NUM_THREADS=N ./bench.out
```
Reports measured memory footprint (RSS) and inference/training throughput across three model sizes (tiny/small/medium), for both SGD and AdamW. Useful for judging what a given deployment target (embedded, Raspberry Pi-class, a laptop) can realistically run - see the Model Hyperparameters section below for how those sizes translate to CLI flags.

## 🚀 Running the Program

The program supports 3 modes via **command-line interface**:

### 1. Training Mode (Default)

Train a new model on input data:

```bash
./app.out train --input data.txt --epochs 5
```

**Training Options:**
- `--input FILE` - Input training file (default: test.txt)
- `--epochs N` - Number of epochs (default: 1)
- `--batch-size N` - Batch size (default: 1)
- `--learning-rate LR` - Learning rate (default: 0.001)
- `--model FILE` - Model path to save (default: dranzer.pth)
- `--checkpoint-dir DIR` - Checkpoint directory (default: checkpoints)
- `--checkpoint-interval N` - Save checkpoint every N steps (default: 10)

**Model Architecture Options:**
- `--vocab-size N` - Vocabulary size (default: 257)
- `--embedding-dim N` - Embedding dimension; must divide evenly by `--num-heads` (default: 16)
- `--num-heads N` - Attention heads (default: 2)
- `--num-layers N` - Stacked transformer layers (default: 2)
- `--max-seq-len N` - Max sequence length the model's workspace is sized for (default: 32)
- `--train-window N` - Sliding context window used during training; clamped to `--max-seq-len` (default: 16)

**Optimizer / Regularization Options:**
- `--optimizer NAME` - `adam` (default) or `sgd`
- `--dropout RATE` - Dropout rate 0.0-1.0 on both sub-layer outputs (default: 0.0, disabled)
- `--grad-clip NORM` - Global gradient-norm clip; 0 disables it (default: 1.0)
- `--weight-decay W` - AdamW decoupled weight decay (default: 0.01)
- `--warmup-steps N` / `--total-steps N` - Linear-warmup + cosine-decay LR schedule horizon; leave `--total-steps 0` (default) to fall back to plateau decay instead

**Examples:**

Basic training:
```bash
./app.out train --input corpus.txt
```

Training with hyperparameters:
```bash
./app.out train \
  --input corpus.txt \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 0.0005 \
  --checkpoint-interval 50
```

Training with GPU flag (if CUDA implemented):
```bash
./app.out train --input data.txt --gpu --debug
```

**Output Files Created:**
- `dranzer.pth` - Latest model weights
- `checkpoints/checkpoint_epoch_N_step_M.ckpt` - Intermediate checkpoints
- `checkpoints/config.txt` - Training configuration

### 2. Inference Mode

Run predictions on a prompt using a trained model:

```bash
./app.out infer --prompt "Once upon a time"
```

**Inference Options:**
- `--prompt TEXT` - Input prompt (**required**)
- `--model FILE` - Model to load (default: dranzer.pth)
- `--sampling STRATEGY` - Sampling: greedy, topk, topp (default: greedy)
- `--top-k N` - Top-k value (default: 5)
- `--top-p P` - Top-p nucleus threshold 0.0-1.0 (default: 0.9)

**Examples:**

Simple inference:
```bash
./app.out infer --prompt "hello"
```

With top-k sampling:
```bash
./app.out infer --prompt "The future is" --sampling topk --top-k 10
```

With top-p sampling:
```bash
./app.out infer --prompt "In a land" --sampling topp --top-p 0.95
```

### 3. Generation Mode

Generate text continuations from a seed prompt:

```bash
./app.out generate --prompt "hello" --length 100
```

**Generation Options:**
- `--prompt TEXT` - Seed prompt (**required**)
- `--model FILE` - Model to load (default: dranzer.pth)
- `--length N` - Tokens to generate (default: 50)
- `--sampling STRATEGY` - Sampling: greedy, topk, topp (default: greedy)
- `--top-k N` - Top-k value (default: 5)
- `--top-p P` - Top-p value 0.0-1.0 (default: 0.9)
- `--temperature T` - Temperature 0.0-2.0 (default: 0.8)

**Examples:**

Simple generation:
```bash
./app.out generate --prompt "Once upon a time" --length 100
```

Creative generation with sampling:
```bash
./app.out generate \
  --prompt "In the beginning" \
  --length 150 \
  --sampling topp \
  --top-p 0.9 \
  --temperature 0.85
```

### General Options (All Modes)

- `--gpu` - Enable GPU acceleration if available
- `--debug` - Enable debug output
- `--help` - Show comprehensive help

**Examples:**
```bash
./app.out --help
./app.out train --input data.txt --debug
./app.out generate --prompt "hello" --debug
```

## 📊 Quick Examples

### Complete Workflow: Train → Infer → Generate

```bash
# 1. Build
cd src
make
cd ..

# 2. Train model
./src/app.out train --input test.txt --epochs 5

# 3. Run inference
./src/app.out infer --prompt "test"

# 4. Generate text
./src/app.out generate --prompt "hello" --length 50 --sampling topk --top-k 5

# 5. Train with checkpoints (multiple training runs)
./src/app.out train --input test.txt --epochs 10 --checkpoint-interval 25

# 6. Debug mode (verbose output)
./src/app.out train --input test.txt --debug
```

### Model Files Explained

| File | Purpose | Created By |
|------|---------|-----------|
| `dranzer.pth` | **Latest model weights** (for inference) | Training mode |
| `checkpoints/checkpoint_epoch_N_*` | **Intermediate snapshots** (for resume/comparison) | Training mode |
| `checkpoints/config.txt` | **Training configuration** (hyperparameters) | Training mode |

**Key Difference:**
- **`dranzer.pth`** - Use for inference/generation (latest trained model)
- **`checkpoints/`** - Use for resuming training, comparing model quality, or rollback

## 🏛️ Architecture

### Model Components

```
Input Tokens
    ↓
BPE Tokenization (vocab_size, --vocab-size)
    ↓
Token Embedding + Fixed Sinusoidal Positional Encoding
    ↓
┌─────────────────────────────────────────┐
│  Transformer Block (× --num-layers)      │
│                                           │
│  Causally-Masked Multi-Head Attention    │  <- position i attends only to j <= i
│    ↓                                     │
│  Dropout → Residual Add → Layer Norm     │
│    ↓                                     │
│  Feedforward (ReLU) → Dropout            │
│    ↓                                     │
│  Residual Add → Layer Norm               │
└─────────────────────────────────────────┘
    ↓ (last layer's last sequence position only)
Output Projection
    ↓
Softmax + Next Token Prediction
```

Training runs the same forward pass, then full hand-rolled backpropagation through every layer (attention, FFN, layer norms, dropout) and the token embeddings, then an optimizer step (AdamW by default, with global gradient-norm clipping - see Optimizer/Regularization Options above). There is no KV cache: each generated token reprocesses the entire context from scratch, which is the dominant cost at longer sequence lengths (see Benchmarking).

### Key Features

- **Multi-Head Attention**: causally masked, `--num-heads` parallel heads per layer
- **Stacked Layers**: `--num-layers` transformer blocks, each with its own attention + FFN + two layer norms
- **Layer Normalization**: learnable gamma/beta parameters, backpropagated exactly (not approximated)
- **Xavier Initialization**: proper weight initialization for deep networks
- **Residual Connections**: prevents vanishing gradients across the stack
- **Dropout**: on both sub-layer outputs, with inverted scaling (no inference-time rescaling needed)
- **Full Backpropagation**: every parameter trains, verified via numerical gradient checking (see Testing)
- **AdamW + Gradient Clipping + LR Schedule**: research-grade optimization, not just plain SGD
- **Advanced Sampling**: 4 strategies for diverse generation

## 📈 Model Hyperparameters

Every hyperparameter below is a CLI flag - nothing requires editing C code and recompiling anymore.

| Parameter | Default | Flag |
|-----------|---------|-----------|
| Vocabulary Size | 257 | `--vocab-size` |
| Embedding Dim | 16 | `--embedding-dim` |
| Attention Heads | 2 | `--num-heads` |
| Layers | 2 | `--num-layers` |
| Max Sequence | 32 | `--max-seq-len` |
| Training Window | 16 | `--train-window` |
| Learning Rate | 0.001 | `--learning-rate` |
| Optimizer | adam | `--optimizer` |
| Dropout | 0.0 | `--dropout` |
| Gradient Clip Norm | 1.0 | `--grad-clip` |
| Weight Decay | 0.01 | `--weight-decay` |
| Batch Size | 1 | `--batch-size` |
| Epochs | 1 | `--epochs` |
| Checkpoint Interval | 10 | `--checkpoint-interval` |

**Sizing guidance**: see [Benchmarking](#benchmarking) above - the "tiny" config in `bench.c` matches these defaults, and "small"/"medium" show what heavier settings cost in memory and latency before you commit to them on constrained hardware.

## 🔍 Debugging & Development

### Enable Debug Output

```bash
cd src
make DEBUG=1
cd ..
./src/app.out train --input test.txt --debug
```

Shows:
- BPE tokenization details
- Model initialization info
- Loss values during training
- Detailed step-by-step execution

### Memory Safety Checks

```bash
cd src
make ASAN=1
cd ..
./src/app.out train --input test.txt
```

Detects:
- Memory leaks
- Use-after-free bugs
- Out-of-bounds access
- Uninitialized memory

### Combined Debugging

```bash
cd src
make DEBUG=1 ASAN=1
cd ..
./src/app.out train --input test.txt --debug
```

## 📚 Build System Details

### Multi-Level Makefiles

**`Makefile`** (root): Delegates to subdirectories
**`libs/Makefile`**: Builds `libattention.a` containing:
- `hashmap.c` - Generic hash table with type-aware storage
- `byte_pair_encoding.c` - BPE tokenizer implementation

**`src/Makefile`**: Builds three independent targets:
- `make` / `make all` - `app.out`, the full CLI (train/infer/generate), linking every module against `libattention.a`
- `make bench` - `bench.out`, linking only `bench.c` + the model modules (not the CLI/tokenizer)
- `make test` - builds and runs every `tests/test_*.c` against the model modules, one at a time
- Supports `DEBUG`, `ASAN`, `SIZE`, `NATIVE`, and `OMP` flags (combinable, e.g. `make OMP=1 CC=gcc DEBUG=1`)

### Compilation Pipeline

```
source files (.c)
    ↓
[cc -c] object files (.o)
    ↓
[ar rcs] libattention.a (libs)
    ↓
[cc link] app.out executable
```

## 🎓 Learning Resources

### Understanding the Code

- **[.agents.md](.agents.md)** - 8 AI agent definitions for code navigation
- **[AGENTS.md](AGENTS.md)** - Human-readable agent workflows
- **[SKILL.md](SKILL.md)** - 9 technical skill domains with patterns

### File Dependencies

```
main.c
├── model.h            (facade - re-exports every module below;
│                        main.c/cli.c/checkpoint.c only ever include this one header)
│   ├── model_types.h   (neural_model_t, transformer_layer_t, enums - no logic)
│   ├── tensor_ops.h    (matmul, softmax, layer norm, dropout, positional encoding)
│   ├── transformer.h   (causally-masked multi-head attention + the stacked forward pass)
│   ├── optimizer.h     (SGD, AdamW, gradient clipping, LR schedule)
│   ├── training.h      (model_train_step's backward orchestration)
│   ├── serialization.h (model_save/model_load)
│   └── metrics.h       (loss-history accessors)
├── tokenizer.c/h      (BPE encoding)
├── attention.c/h      (legacy single-head self-attention demo)
├── cli.c/h            (argument parsing)
├── config.c/h         (configuration)
├── sampling.c/h       (sampling strategies)
├── batch.c/h          (batch processing)
└── checkpoint.c/h     (checkpoints)
    └── byte_pair_encoding.c/h (in libs)
```

Internal modules include only the narrow header(s) they actually need (e.g. `training.c` includes `transformer.h`+`tensor_ops.h`+`optimizer.h`, not the `model.h` umbrella) - `model.h` is a convenience for external callers, not how the modules talk to each other.

## 🔬 Output Example

```
>>> Mode: TRAIN

=== Neural Model Training (LARGE FILE OPTIMIZED) ===

[1] Setting up streaming tokenization...
   ✓ Streaming reader created (256KB chunks)
   ✓ Token stream buffer created (batch size: 1000)
[2] Initializing neural model...
   Model initialized:
   - Vocabulary: 257 tokens
   - Embedding dim: 16
   - Attention heads: 2
   - Layers: 2
   - Optimizer: adam (grad-clip=1.00, weight-decay=0.0100, dropout=0.00)
   ✓ Model ready for training
[3] Training model on streaming data...
   Epochs: 3, Batch size: 1, Learning rate: 0.00200000
   Processing epoch 1/3...
   Epoch 1/3 - Avg Loss: 3.397431, Tokens processed: 168
   Processing epoch 2/3...
   Epoch 2/3 - Avg Loss: 2.645559, Tokens processed: 336
   Processing epoch 3/3...
   Epoch 3/3 - Avg Loss: 2.336254, Tokens processed: 504
   Training complete!
   Total steps: 501, Total tokens: 504
   ✓ Training finished

[5] Saving model to dranzer.pth
   ✓ Model saved
[6] Saving configuration...
   ✓ Configuration saved
```

## 🚦 Troubleshooting

**Build fails with `undefined reference`**
- Run `make clean` then `make` from `src/` directory
- Ensure libraries are built first

**Model not found for inference**
- Train first: `./app.out train --input data.txt`
- This creates `dranzer.pth`

**Memory issues during training**
- Reduce batch size: `--batch-size 1`
- Use AddressSanitizer to detect leaks: `make ASAN=1`

**Slow compilation**
- Recompile only changed files: `make` (from `src/`)
- Avoid `make clean` unless necessary

## 📄 License

MIT License - Free for educational and research use.

## 👤 Author

Built as a comprehensive neural network implementation in C, covering:
- Core ML concepts (attention, normalization, sampling)
- Low-level systems programming (memory management, binary formats)
- Software engineering (modular design, testing, documentation)

---

**Last Updated**: July 2026
**Status**: Phase 4 Complete ✅
**Version**: 2.0 (Multi-Layer, Full Backprop, AdamW, Modular)
