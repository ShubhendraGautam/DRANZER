# DRANZER

[![CI](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/ci.yml/badge.svg)](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/ci.yml)
[![Nightly](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/nightly.yml/badge.svg)](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/nightly.yml)
[![Performance](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/performance.yml/badge.svg)](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/performance.yml)

### A decoder-only transformer, built from scratch in C

DRANZER implements the model, training loop, backpropagation, optimizer, tokenizer, and inference
runtime without an ML framework or autodiff library. The result is a compact codebase for learning
how transformers work all the way down to memory layout, numerical gradients, and GPU kernel
dispatch.

The default build runs anywhere with a C compiler, and stays portable while still using the vector
instructions of whatever CPU it lands on: the AVX2, AVX-512, and NEON matmul kernels are selected at
runtime from CPUID rather than baked in with `-march`, so one binary uses the widest it finds and
falls back to portable C on a machine with none. On NVIDIA systems, forward and backward matrix
multiplication can also run through hand-written PTX loaded directly by the CUDA driver—no CUDA
toolkit required.

## What makes it interesting

- Multi-layer, causal, multi-head self-attention with residual connections and layer normalization
- Full hand-written backpropagation, checked against numerical gradients
- AdamW/SGD, true minibatch gradient averaging, accumulation, dropout, clipping, and LR schedules
- Frozen tokenization with BOS/EOS, held-out perplexity, and exact resumable training checkpoints
- Single-file model bundles and ring-KV-cached greedy/top-k/top-p decoding
- Incremental generation callbacks, streamed CLI output, stop sequences, and repetition controls
- Portable CPU execution, runtime-dispatched AVX2/AVX-512/NEON matmul kernels, and optional OpenMP
- Optional NVIDIA GPU offload for forward and backward matmuls, with persistent buffers and a validated weight cache
- Built-in tests, benchmarks, hardware probing, serialization, and GitHub Actions CI

## Quick start

DRANZER is currently Linux-focused. You need Clang or GCC and GNU Make.

```bash
git clone https://github.com/ShubhendraGautam/DRANZER.git
cd DRANZER/src
make

# Train on the included sample
./app.out train --input ../test.txt --epochs 3

# Continue the most recent checkpoint exactly
./app.out train --resume latest --checkpoint-dir checkpoints

# Use the saved model
./app.out eval --model dranzer.pth --input ../test.txt
./app.out infer --prompt "hello"
./app.out generate --prompt "hello" --length 20 --sampling topp --top-p 0.9 --seed 42
```

Run `./app.out --help` for the complete CLI.

## Architecture at a glance

```text
tokens → BPE → token + positional embeddings
       → N × [causal multi-head attention → residual + layer norm
              → feed-forward network → residual + layer norm]
       → output projection → next-token logits
```

Training follows the same path in reverse through every transformer layer and trainable parameter.
Parameters and gradients live in contiguous buffers, which keeps optimization and serialization
simple and makes the low-level data flow easy to inspect.

## Try the developer tools

From the repository root:

```bash
make -C src test                         # correctness suite
make -C src clean all CC=gcc OMP=1      # OpenMP build
make -C src clean test CC=clang ASAN=1  # memory-safety checks
make -C src bench && ./src/bench.out     # model benchmarks
make -C src profile CC=gcc               # frame-pointer build for perf
make -C src gpu-probe && ./src/gpu_probe.out
make -C src gpu-latency && ./src/gpu_latency.out  # GPU per-call cost
```

GPU tests compile on every machine and self-skip when CUDA hardware is unavailable.

## Documentation

| Guide | What it covers |
|---|---|
| [Usage and CLI](docs/usage.md) | Installation, build variants, commands, flags, and troubleshooting |
| [Architecture](docs/architecture.md) | Model flow, memory layout, modules, and repository structure |
| [Model bundle](docs/model-bundle.md) | Portable artifact layout, validation, and legacy compatibility |
| [Special tokens](docs/special-tokens.md) | Stable IDs, sequence boundaries, EOS stopping, and legacy mode |
| [Generation runtime](docs/generation.md) | Streaming callbacks, stop sequences, sampling controls, and result semantics |
| [CPU matmul kernels](docs/matmul.md) | Portable and SIMD kernels, runtime dispatch, reproducibility, and the measurement workflow |
| [Weight quantization](docs/quantization.md) | INT8/INT4 accuracy cost measured in weight, logit, and cross-entropy space, and what the seed count does to it |
| [CPU threading](docs/threading.md) | When an OpenMP region is worth entering, the measured cutoff, and why a persistent worker pool was rejected |
| [GPU backend](docs/gpu.md) | PTX execution, capability probing, caching, limitations, and measurements |
| [Development](docs/development.md) | Tests, CI/nightly jobs, sanitizers, benchmarks, and contribution workflow |
| [Design checklist](docs/design-checklist.md) | Prioritized maturity roadmap and acceptance gates |

## Scope

DRANZER is an educational and systems-research implementation, not a production LLM runtime.
Generation can continue beyond the model context window by evicting the oldest per-layer KV rows;
quality is still limited by that fixed retained context and by positions beyond those seen during
training. GPU offload is NVIDIA-only and covers matrix multiplications: all of them in the forward
pass, and the two backward matmuls above a measured shape threshold. The optimizer step, attention
scores, softmax, and layer normalization stay on the CPU, and activations round-trip to host memory
between operations, so a training step is not GPU-resident.

## License

MIT License. Built for learning, experimentation, and research.
