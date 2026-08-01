# DRANZER

[![CI](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/ci.yml/badge.svg)](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/ci.yml)
[![Nightly](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/nightly.yml/badge.svg)](https://github.com/ShubhendraGautam/DRANZER/actions/workflows/nightly.yml)

### A decoder-only transformer, built from scratch in C

DRANZER implements the model, training loop, backpropagation, optimizer, tokenizer, and inference
runtime without an ML framework or autodiff library. The result is a compact codebase for learning
how transformers work all the way down to memory layout, numerical gradients, and GPU kernel
dispatch.

The default build runs anywhere with a C compiler. On NVIDIA systems, forward-pass matrix
multiplication can also run through hand-written PTX loaded directly by the CUDA driver—no CUDA
toolkit required.

## What makes it interesting

- Multi-layer, causal, multi-head self-attention with residual connections and layer normalization
- Full hand-written backpropagation, checked against numerical gradients
- AdamW, SGD, gradient clipping, dropout, and warmup/cosine learning-rate scheduling
- Portable CPU execution plus optional OpenMP parallelism
- Optional NVIDIA GPU offload with persistent buffers and a validated weight cache
- Built-in tests, benchmarks, hardware probing, serialization, and GitHub Actions CI

## Quick start

DRANZER is currently Linux-focused. You need Clang or GCC and GNU Make.

```bash
git clone https://github.com/ShubhendraGautam/DRANZER.git
cd DRANZER/src
make

# Train on the included sample
./app.out train --input ../test.txt --epochs 3

# Use the saved model
./app.out infer --prompt "hello"
./app.out generate --prompt "hello" --length 20
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
make -C src gpu-probe && ./src/gpu_probe.out
```

GPU tests compile on every machine and self-skip when CUDA hardware is unavailable.

## Documentation

| Guide | What it covers |
|---|---|
| [Usage and CLI](docs/usage.md) | Installation, build variants, commands, flags, and troubleshooting |
| [Architecture](docs/architecture.md) | Model flow, memory layout, modules, and repository structure |
| [GPU backend](docs/gpu.md) | PTX execution, capability probing, caching, limitations, and measurements |
| [Development](docs/development.md) | Tests, CI/nightly jobs, sanitizers, benchmarks, and contribution workflow |

## Scope

DRANZER is an educational and systems-research implementation, not a production LLM runtime.
Generation currently uses greedy decoding and reprocesses the context without a KV cache. GPU
offload is NVIDIA-only and covers forward-pass matrix multiplications; training's backward pass
remains on the CPU. These boundaries are intentional and documented so performance claims stay
honest.

## License

MIT License. Built for learning, experimentation, and research.
