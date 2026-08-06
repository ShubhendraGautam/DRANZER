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

The CLI first makes a dedicated corpus pass to train the BPE vocabulary, records a deterministic
corpus fingerprint, and freezes the encoder before model initialization. A frozen encoder rejects
further `bpe_train()` calls, so token IDs and merge order cannot change during optimizer updates.
Passing an existing explicit tokenizer path reuses that frozen vocabulary after a model-vocabulary
compatibility check.

New tokenizers keep byte IDs 0–255, reserve PAD/UNK/BOS/EOS at 256–259, and learn merges from 260.
Training and evaluation both model one file as `BOS, encoded bytes, EOS`; training retains its
sliding context across file chunks. Legacy tokenizers are tagged separately and keep merges at 256
without acquiring new boundary semantics.

`model_train_step()` performs four phases:

1. Run the normal forward pass while retaining activations needed by backward.
2. Compute softmax cross-entropy for the target token.
3. Backpropagate through the output head, every transformer block, and token embeddings.
4. Clip gradients and apply AdamW or SGD.

The CLI's minibatch path separates phases 2–3 from phase 4. It zeroes gradients once, calls
`model_accumulate_gradients()` for each example across the configured minibatches and accumulation
steps, then calls `model_apply_accumulated_gradients()` once. That function divides by the actual
sample count before clipping and optimization. The legacy `model_train_step()` is the same path
with a sample count of one.

Examples remain streamed and only one minibatch of context windows is stored. Optional shuffling is
therefore deliberately minibatch-local. Its stateless seed derives from run seed, epoch, and
minibatch index, keeping data order reproducible without coupling it to model dropout RNG.

The backward pass covers attention projections, causal softmax attention, layer normalization,
feed-forward layers, residual branches, dropout, embeddings, and the output projection. Numerical
gradient tests protect this path from implementations that appear to train but use incorrect
derivatives.

## Evaluation path

`model_evaluate_step()` runs the normal forward path with dropout disabled and computes stable
log-sum-exp cross-entropy. It restores the model's prior training mode and never invokes backward,
the optimizer, the learning-rate scheduler, or training-metric updates.

The CLI corpus evaluator maintains a bounded sliding context across streaming file chunks and
accumulates loss in double precision. Both standalone `eval` and `train --validation` use this same
path, so reported held-out cross-entropy and perplexity have identical semantics.

## Checkpoint and resume path

Training dropout draws from RNG state owned by each model rather than the process-global C RNG.
The checkpoint stream stores that state together with parameters, gradients, optimizer moments,
scheduler counters, loss history, architecture, run configuration, corpus provenance, and the
frozen BPE encoder. A magic value, format version, and terminal footer reject incomplete or
unsupported files.

Each checkpoint cursor is an epoch number plus the count of completed next-token updates within
that epoch. On resume, the CLI streams and tokenizes from the start of the recorded epoch, skips
exactly that many predictions without invoking the model, then continues optimization. This keeps
memory bounded while preserving the model RNG and optimizer trajectory. The input byte count and
FNV-1a fingerprint must match before replay begins.

Writes go to a process-specific temporary file in the destination directory, are flushed and
synced, and become visible through one atomic rename. Afterward, retention removes all but the
configured number of newest checkpoints. Exact bitwise continuation is guaranteed for the same
executable and CPU/GPU execution path.

## Model bundle path

Normal training output uses a separate versioned bundle contract for evaluation and inference. It
stores canonical little-endian binary32 parameters, model dimensions, frozen BPE merge order,
training step/loss metadata, seed, and corpus provenance. Independent weight and tokenizer
checksums, exact-length/footer checks, shape arithmetic, and pre-allocation bounds reject malformed
input. The CLI prefers this embedded tokenizer and falls back read-only to legacy host-native
weights plus sidecars when bundle magic is absent.

Bundles are smaller than checkpoints because they omit optimizer, gradient, scheduler, RNG, and
cursor state. The exact layout and compatibility rules are documented in
[Model bundle format](model-bundle.md).

## Incremental generation

Generation uses `model_forward_token()` instead of running `model_forward()` over the growing
prefix. A `model_kv_cache_t` stores every layer's projected keys and values. For a new position the
runtime computes one query/key/value row, attends over cached rows, and advances one hidden state
through the stack.

Prompt processing fills the cache once. Subsequent decode work grows linearly with the cached
context for attention, while transformer projections and feed-forward layers process only the new
token. `test_kv_cache.c` compares logits from this path against full-prefix logits at every
position before the first eviction, then checks multiple ring wraps against the same logical cache
normalized to a linear layout.

At capacity, the oldest key/value row in every layer is overwritten and the ring start advances.
The cache retains the newest `max_seq_len` contextualized rows; retained higher-layer rows are not
recomputed after older context is evicted. Token positions continue absolutely. Rows inside the
original window use the model's stored sinusoidal table, while later rows evaluate the identical
formula on demand. This makes long decoding explicit and bounded in KV memory, while acknowledging
that positions beyond the training window are extrapolated.

The model-visible prompt begins with BOS in special-token mode. Before each sample the runtime
applies repetition controls and masks PAD, UNK, BOS, and unassigned vocabulary slots. EOS becomes
eligible after the configured minimum length. One shared decode loop handles EOS, caller stop
sequences, and callback cancellation. It emits safe decoded token pieces immediately while holding
only a possible stop-sequence prefix; control IDs and matching stop markers are never emitted. See
[Generation runtime](generation.md) for the interface contract.

## Parameter and activation memory

All trainable parameters live in one contiguous `params` allocation. Gradients use an identical
contiguous layout in `grads`; Adam moment buffers use that layout as well and are allocated lazily.
Named fields such as `W_q` and `output_projection` are views into those buffers.

Forward matmuls normally use the CPU dispatch path (or opt-in GPU path), which picks a kernel from
the shape of the call and the instruction set the running CPU supports - see
[CPU matmul kernels](matmul.md). Setting `model.use_scalar_matmul`
forces the portable unblocked C reference through full-prefix and cached decode instead. That
reference remains independently selectable so every optimized kernel can be checked at the model
level, not only on isolated matrices.

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
│   ├── core/                model, matmul kernels (portable + SIMD), tensor ops,
│   │                        transformer, training, CPU feature detection
│   ├── backends/gpu/        CUDA driver wrapper, PTX matmul, hardware probe
│   ├── cli/                 commands, tokenizer adapter, config, sampling
│   ├── include/             public headers mirroring the source layout
│   ├── tests/core/          numerical and behavioral correctness tests
│   ├── tests/gpu/           GPU/CPU equivalence and cache tests
│   └── tools/               standalone benchmark and matmul kernel sweep
├── docs/                    focused project documentation
└── .github/                 CI, nightly, and dependency automation
```

## Module map

| Module | Responsibility |
|---|---|
| `core/model.c` | Model allocation, contiguous tensor layout, and lifecycle |
| `core/matmul.c` | Portable CPU matmul kernels and the selection policy |
| `core/matmul_x86.c` | AVX2 and AVX-512 kernels, reached only through runtime dispatch |
| `core/matmul_arm.c` | NEON kernel, reached only through runtime dispatch |
| `core/cpu_features.c` | Runtime instruction-set detection behind that dispatch |
| `core/parallel.c` | Whether an OpenMP region is worth entering, and the guarded loop form |
| `core/quantize.c` | Symmetric integer grids and the error they introduce |
| `core/model_params.c` | Inventory of the model's trainable tensors, with shapes and roles |
| `core/model_quantize.c` | Quantization policy over those tensors, and its error report |
| `core/tensor_ops.c` | Softmax, layer norm, dropout, and positional encoding |
| `core/transformer.c` | Causal attention, transformer blocks, forward pass, backend dispatch |
| `core/training.c` | Cross-entropy and full backward-pass orchestration |
| `core/evaluation.c` | Side-effect-free next-token cross-entropy |
| `core/optimizer.c` | SGD, AdamW, clipping, and learning-rate schedules |
| `core/bundle.c` | Canonical model/tokenizer artifact and strict validation |
| `core/serialization.c` | Legacy model I/O and shared checkpoint model state |
| `core/metrics.c` | Loss history and running metrics |
| `cli/main.c` | Train, eval, infer, and generate orchestration |
| `cli/tokenizer.c` | Adapter around the BPE utility library |
| `cli/evaluation.c` | Streaming corpus metrics and perplexity reporting |
| `cli/generation.c` | Prompt policy, generation controls, callbacks, stop matching, and cached decode loop |
| `cli/checkpoint.c` | Atomic complete-state checkpoints, resume loading, and retention |
| `cli/manifest.c` | Exclusive-create, read-only resolved run manifests |
| `cli/sampling.c` | Temperature, greedy, top-k, top-p, and beam-search helpers |
| `backends/gpu/gpu_cuda.c` | Minimal dynamically loaded CUDA Driver API wrapper |
| `backends/gpu/gpu_matmul.c` | Hand-written PTX matmul and device-side caches |
| `tools/bench.c` | Model-size and throughput benchmark |
| `tools/bench_parallel.c` | Parallel-region entry cost and the fork/serial crossover sweep |
| `tools/bench_quant.c` | Quantization cost in weight, logit, and cross-entropy space |

Headers live under `src/include/` and mirror the source hierarchy. External callers can include
`core/model.h` as a facade; internal modules include only the narrow headers they need.

## Current design boundaries

- Fixed sinusoidal positional encodings
- ReLU feed-forward network with width `4 × embedding_dim`
- Ring-KV-cached generation with a fixed retained window and absolute sinusoidal positions
- Last-position next-token prediction
- Linux-focused runtime and hardware probing
- Optional NVIDIA-only GPU acceleration for forward matmuls

These constraints keep the code inspectable. They are also the clearest extension points for
sliding context windows, alternate activations, additional backends, and GPU-resident training.
