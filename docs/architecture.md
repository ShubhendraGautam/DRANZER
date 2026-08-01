# Architecture

[← Back to README](../README.md)

DRANZER is a small decoder-only transformer whose forward pass, backward pass, optimizer, and
runtime are all implemented in C.

## Model flow

```text
token IDs
   │
   ├─ token embeddings
   └─ fixed sinusoidal position embeddings
   │
   ▼
┌────────────────────────────────────────────┐
│ Transformer block × num_layers             │
│                                            │
│ causal multi-head self-attention           │
│   → dropout → residual add → layer norm    │
│ feed-forward network (4× width, ReLU)      │
│   → dropout → residual add → layer norm    │
└────────────────────────────────────────────┘
   │
   ▼
last-position hidden state
   → output projection + bias
   → next-token logits
```

The causal mask prevents position `i` from attending to any position after `i`. Each attention
head operates on a disjoint slice of the embedding dimension, then the head outputs are
concatenated and projected back to model width.

## Training path

`model_train_step()` performs four phases:

1. Run the normal forward pass while retaining activations needed by backward.
2. Compute softmax cross-entropy for the target token.
3. Backpropagate through the output head, every transformer block, and token embeddings.
4. Clip gradients and apply AdamW or SGD.

The backward pass covers attention projections, causal softmax attention, layer normalization,
feed-forward layers, residual branches, dropout, embeddings, and the output projection. Numerical
gradient tests protect this path from implementations that appear to train but use incorrect
derivatives.

## Parameter and activation memory

All trainable parameters live in one contiguous `params` allocation. Gradients use an identical
contiguous layout in `grads`; Adam moment buffers use that layout as well and are allocated lazily.
Named fields such as `W_q` and `output_projection` are views into those buffers.

This layout provides three useful properties:

- Optimizer updates can iterate over every parameter in one loop.
- Serialization can read and write a single stable parameter block.
- Adding a tensor does not require a separate optimizer or serializer code path.

Forward activations and backward scratch buffers are allocated once by `model_new()` for the
configured maximum sequence length. They are reused on every step, avoiding allocations in the
hot path. A model instance is therefore stateful and not reentrant or thread-safe.

## CPU and GPU boundary

Core model code is backend-neutral except for one deliberate seam: `dispatch_matmul()` in
`core/transformer.c`. It selects the GPU matmul implementation when `model->use_gpu` is enabled and
a CUDA device is usable; otherwise it calls the CPU reference implementation.

Only forward-pass matrix multiplications cross this seam today. The backward pass, nonlinear
operations, layer normalization, softmax, optimizer, and tokenizer remain on the CPU.

## Repository layout

```text
.
├── libs/                    tokenizer and hashmap utility library
├── src/
│   ├── core/                model, tensor ops, transformer, training, optimizer
│   ├── backends/gpu/        CUDA driver wrapper, PTX matmul, hardware probe
│   ├── cli/                 commands, tokenizer adapter, config, sampling
│   ├── include/             public headers mirroring the source layout
│   ├── tests/core/          numerical and behavioral correctness tests
│   ├── tests/gpu/           GPU/CPU equivalence and cache tests
│   └── tools/               standalone benchmark
├── docs/                    focused project documentation
└── .github/                 CI, nightly, and dependency automation
```

## Module map

| Module | Responsibility |
|---|---|
| `core/model.c` | Model allocation, contiguous tensor layout, and lifecycle |
| `core/tensor_ops.c` | Matmul, softmax, layer norm, dropout, and positional encoding |
| `core/transformer.c` | Causal attention, transformer blocks, forward pass, backend dispatch |
| `core/training.c` | Cross-entropy and full backward-pass orchestration |
| `core/optimizer.c` | SGD, AdamW, clipping, and learning-rate schedules |
| `core/serialization.c` | Model save/load format |
| `core/metrics.c` | Loss history and running metrics |
| `cli/main.c` | Train, infer, and generate modes |
| `cli/tokenizer.c` | Adapter around the BPE utility library |
| `cli/sampling.c` | Greedy, top-k, top-p, and beam-search helpers |
| `backends/gpu/gpu_cuda.c` | Minimal dynamically loaded CUDA Driver API wrapper |
| `backends/gpu/gpu_matmul.c` | Hand-written PTX matmul and device-side caches |
| `tools/bench.c` | Model-size and throughput benchmark |

Headers live under `src/include/` and mirror the source hierarchy. External callers can include
`core/model.h` as a facade; internal modules include only the narrow headers they need.

## Current design boundaries

- Fixed sinusoidal positional encodings
- ReLU feed-forward network with width `4 × embedding_dim`
- Full-context generation without a KV cache
- Last-position next-token prediction
- Linux-focused runtime and hardware probing
- Optional NVIDIA-only GPU acceleration for forward matmuls

These constraints keep the code inspectable. They are also the clearest extension points for KV
caching, alternate activations, additional backends, and GPU-resident training.
