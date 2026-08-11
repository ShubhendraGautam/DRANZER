# Usage and CLI

[← Back to README](../README.md)

This guide covers building DRANZER and using its `train`, `eval`, `infer`, and `generate` modes.

## Requirements

- Linux
- Clang or GCC with C11 support
- GNU Make
- Standard C library, `libm`, and `libdl`

The release-tested configuration is Ubuntu 24.04 on x86-64. Other operating systems,
architectures, SIMD paths, OpenMP, mmap, and CUDA have deliberately different support levels; see
[Supported platforms](platform-support.md) before distributing a binary.

Ubuntu/Debian:

```bash
sudo apt-get install clang make build-essential
```

No CUDA or OpenCL SDK is needed for a normal build. Optional GPU support dynamically loads the
runtime libraries already installed by the driver.

## Build

```bash
cd src
make
```

This builds `src/app.out` and the tokenizer utility library in `libs/`.

### Build variants

| Command | Purpose |
|---|---|
| `make DEBUG=1` | Add debug symbols and compile-time debug logging |
| `make ASAN=1` | Enable AddressSanitizer |
| `make SIZE=1` | Optimize for binary size with `-Os` |
| `make NATIVE=1` | Tune for the current CPU with `-march=native` |
| `make OMP=1 CC=gcc` | Parallelize matmul and attention with OpenMP |
| `make clean` | Remove generated objects and binaries |

`NATIVE=1` produces a machine-specific binary. The default build intentionally avoids it so the
binary remains portable across older CPUs.

## Train a model

Run from `src/` so generated paths stay together:

```bash
./app.out train --input ../test.txt --epochs 3
```

A more explicit configuration:

```bash
./app.out train \
  --input ../test.txt \
  --validation ../validation.txt \
  --model dranzer.pth \
  --epochs 10 \
  --learning-rate 0.0005 \
  --optimizer adam \
  --dropout 0.1 \
  --grad-clip 1.0 \
  --warmup-steps 100 \
  --total-steps 5000
```

Before initializing the model, training makes one streaming corpus pass to train and freeze the BPE
vocabulary. Every epoch then uses the same token IDs and merge order. Training atomically saves a
self-contained, versioned model bundle at `--model`. It includes canonical float32 weights,
architecture, the frozen tokenizer, training metadata, and corpus provenance. `config.json` records
the resolved settings and held-out metrics.

Use `--tokenizer FILE` to choose a different sidecar. If that explicit file already exists, training
loads and freezes it instead of learning a new vocabulary, and rejects a `--vocab-size` mismatch. If
it does not exist, training creates it during the tokenizer pass. The sidecar is retained for
tokenizer reuse and older tooling; evaluation and generation of the final bundle do not require it.
See [Model bundle format](model-bundle.md) for the binary contract and compatibility policy.

Fresh tokenizers preserve byte IDs 0–255 and reserve PAD=256, UNK=257, BOS=258, and EOS=259;
learned merges begin at 260. Each input file is trained as `BOS, corpus tokens, EOS`, including
across streaming chunk boundaries. See [Special-token contract](special-tokens.md) for exact train,
evaluation, prompt, decoding, and legacy semantics.

### Checkpoint and resume

Training saves an atomic, self-contained checkpoint every `--checkpoint-interval` optimizer steps
and once at normal completion. Resume a specific file or the newest retained checkpoint:

```bash
./app.out train --resume checkpoints/checkpoint_epoch_0_step_100.ckpt
./app.out train --resume latest --checkpoint-dir checkpoints
```

The checkpoint embeds parameters, gradients, Adam moments, learning-rate scheduler and metric
state, the model-owned dropout RNG, the frozen tokenizer, architecture and training controls, input
provenance, and the exact epoch/update cursor. The original model and tokenizer sidecar are not
needed to resume. `--epochs N` is the total target, so it can extend a run; it is not a number of
additional epochs.

Resume restores the recorded input unless `--input` is supplied. An explicit replacement must have
the same byte count and FNV-1a fingerprint or training stops. `--checkpoint-dir` may redirect new
checkpoints. All other trajectory-affecting controls come from the checkpoint.

Exactness means bit-for-bit final training state when the same executable and execution backend are
used. Resume replays the deterministic frozen-token stream up to its update cursor without running
the model or consuming dropout RNG. Different compiler flags, CPU/GPU paths, libraries, or hardware
may have different floating-point behavior and are outside that guarantee.

Trajectory-changing overrides such as batch size, optimizer, dropout, seed, architecture, or
shuffling are rejected during exact resume. `--epochs` may extend the target, and output model,
tokenizer, checkpoint directory, checkpoint interval, and retention settings may be redirected.

### Minibatches, accumulation, and shuffling

`--batch-size N` now means N next-token examples per minibatch. DRANZER runs their forward/backward
passes serially through its bounded model workspace, sums their gradients, divides by the actual
sample count, and performs one optimizer update. A short final minibatch is averaged over its real
size.

`--gradient-accumulation N` combines N minibatches before that update, so the usual effective batch
size is `batch-size × gradient-accumulation`. Partial final accumulation groups are also applied
using their actual sample count. Learning-rate schedules, checkpoint intervals, `training_steps`,
and loss-history entries count optimizer updates—not individual examples.

`--shuffle` applies a deterministic Fisher–Yates permutation within each bounded minibatch. Its seed
is derived from `--seed`, epoch, and minibatch index without consuming model/dropout RNG. This is an
honest bounded streaming shuffle, not a global whole-corpus permutation.

### Training and model options

| Option | Default | Meaning |
|---|---:|---|
| `--input FILE` | `../test.txt` | Training corpus; passing it explicitly is recommended |
| `--validation FILE` | none | Explicit held-out corpus evaluated after every epoch |
| `--model FILE` | `dranzer.pth` | Model output/input path |
| `--tokenizer FILE` | `<model>.tokenizer` | Frozen BPE path; an existing explicit file is reused |
| `--epochs N` | `1` | Number of passes over the input |
| `--batch-size N` | `1` | Examples averaged in each true minibatch |
| `--gradient-accumulation N` | `1` | Minibatches combined per optimizer update |
| `--shuffle` | off | Deterministically shuffle examples within each minibatch |
| `--learning-rate LR` | `0.001` | Initial learning rate |
| `--checkpoint-dir DIR` | `checkpoints` | Checkpoint output directory |
| `--checkpoint-interval N` | `10` | Save every N optimizer steps; `0` disables checkpoints |
| `--keep-checkpoints N` | `3` | Retain the newest N files; `0` keeps all |
| `--resume FILE` | none | Resume a checkpoint exactly; `latest` selects the newest in the directory |
| `--vocab-size N` | `260` | Total vocabulary including four controls; learned merges start at 260 |
| `--embedding-dim N` | `16` | Hidden width; must divide evenly by the head count |
| `--num-heads N` | `2` | Attention heads per layer |
| `--num-layers N` | `2` | Transformer block count |
| `--max-seq-len N` | `32` | Retained training and ring-KV attention window |
| `--train-window N` | `16` | Training context length, clamped to max sequence length |
| `--train-stride N` | `0` | Tokens the window advances between examples; `0` means a whole window |
| `--tie-embeddings` | off | Share the token embedding table with the output projection |
| `--rope` | off | Rotate per-head attention Q/K pairs by absolute position instead of adding sinusoidal vectors |
| `--rmsnorm` | off | Use RMSNorm scales without LayerNorm mean subtraction or beta tensors |
| `--gelu` | off | Use exact erf-based GELU instead of ReLU in feed-forward layers |
| `--swiglu` | off | Use `SiLU(xW₁+b₁) ⊙ (xW_gate+b_gate)`; mutually exclusive with `--gelu` |
| `--optimizer NAME` | `adam` | `adam` or `sgd` |
| `--dropout RATE` | `0.0` | Sublayer-output dropout rate |
| `--grad-clip NORM` | `1.0` | Global gradient norm limit; `0` disables it |
| `--weight-decay W` | `0.01` | Decoupled AdamW weight decay |
| `--warmup-steps N` | `0` | Linear learning-rate warmup length |
| `--total-steps N` | `0` | Warmup/cosine horizon; `0` uses plateau decay |

### Windows, stride, and what one training example is

Every position in a training window is supervised: a window of `--train-window` tokens carries a
target for each of its positions, so one forward and backward pass produces that many gradient
signals rather than one. The causal mask makes this sound — position `i` cannot have seen token
`i+1` — and `core/lm_head.h` explains the mechanics.

`--train-stride` decides how far the window moves between examples, and the default of a whole
window means non-overlapping: every corpus token is supervised exactly once per epoch. A smaller
stride overlaps windows, so early positions get more preceding context — the first position of a
non-overlapping window has exactly one token of it — at a proportional cost in compute and
optimizer steps.

On a 20 KB corpus at `--train-window 32`, stride 1 and stride 32 cover the same targets in 11044
and 346 passes respectively. Whether the extra context buys any held-out quality is **unmeasured**
at this project's model sizes and should be compared against the seed-variance floor before being
believed; the flag exists so that comparison is possible rather than assumed.

Expect step counts to differ from runs made before this change. The same corpus and the same
`--epochs` now produce roughly `train_window / train_stride` times fewer optimizer steps, because
an example is a window rather than a single target. Learning-rate schedules configured with
`--total-steps` need rescaling to match.

## Evaluation

Evaluate a saved model on an explicit held-out corpus:

```bash
./app.out eval \
  --model dranzer.pth \
  --input ../validation.txt \
  --eval-window 16
```

`eval` reports the corpus byte fingerprint, token and next-token prediction counts, average
cross-entropy, perplexity, elapsed time, and throughput. `--input` is required; DRANZER never hides
an automatic validation split. An evaluation window of `0` (the default) uses the model's maximum
sequence length.

Evaluation disables dropout and does not update parameters, gradients, Adam moments, learning-rate
state, loss history, or training counters. Supplying `--validation FILE` to `train` runs the same
evaluator after every epoch with the training context window. The final validation metrics and
held-out corpus fingerprint are saved in `config.json`.

For current tokenizers the reported token count includes one BOS and one EOS, and the prediction
count includes the corpus's final EOS target. Legacy tokenizers retain their historical counts.

## Inference

Inference loads a trained model and predicts its next token:

```bash
./app.out infer --model dranzer.pth --prompt "Once upon a time"
```

`--prompt` is required. Inference prepends BOS and uses the tokenizer embedded in a current model bundle, then applies
the selected decoding strategy to the next-token logits. Legacy weight files still use their
associated sidecar or byte-vocabulary fallback. Add `--gpu` to request GPU matmul; the runtime falls
back to CPU when a usable CUDA device is not present.

## Generation

```bash
./app.out generate \
  --model dranzer.pth \
  --prompt "In the beginning" \
  --length 20 \
  --sampling topp \
  --top-p 0.9 \
  --temperature 0.8 \
  --repetition-penalty 1.15 \
  --min-length 5 \
  --stop "END" \
  --seed 42
```

Generation stops when it reaches `--length` new tokens, and current models stop earlier when they
sample EOS or a repeated `--stop TEXT` value. `--max-seq-len` is a sliding attention window: once
full, the oldest key/value row is evicted while generation continues with absolute sinusoidal
positions.
Stop markers are withheld. PAD, UNK, BOS, and unassigned vocabulary slots cannot be sampled as
output, and all control tokens are omitted while decoding. Prompt tokens are processed once to
populate a per-layer KV cache; each subsequent step computes only the new query, key, value, and
hidden state. Output is decoded and flushed incrementally. Positions beyond the training window are
an explicit extrapolation, so longer output is supported mechanically but is not a quality promise.

Available decoding strategies:

| Strategy | Command | Behavior |
|---|---|---|
| Greedy | `--sampling greedy` | Always choose the largest logit |
| Top-k | `--sampling topk --top-k 10` | Sample from the `k` highest logits |
| Top-p | `--sampling topp --top-p 0.9` | Sample from the smallest nucleus reaching probability `p` |

`--temperature` scales non-greedy logits before sampling. Lower positive values sharpen the
distribution; higher values flatten it. A zero value falls back to greedy decoding. `--seed`
makes top-k and top-p runs reproducible and defaults to `1`.

| Option | Default | Meaning |
|---|---:|---|
| `--length N` | `50` | Maximum number of new tokens |
| `--temperature T` | `0.8` | Non-greedy logit scaling, from `0.0` to `2.0` |
| `--repetition-penalty P` | `1.0` | Sign-aware penalty for tokens already seen; must be at least `1.0` |
| `--min-length N` | `0` | Content tokens required before EOS or a stop marker may terminate |
| `--stop TEXT` | none | Tokenized stop marker; may be repeated eight times and is not printed |
| `--seed N` | `1` | Reproducible random seed for sampling |

See [Generation runtime](generation.md) for callback semantics, stop-prefix buffering, operation
order, and the token-based matching contract.

## General options

| Option | Meaning |
|---|---|
| `--gpu` | Use NVIDIA GPU matmul when available; otherwise fall back to CPU |
| `--debug` | Enable runtime debug output |
| `--help`, `-h` | Print the complete command reference |

## Generated files

| File | Purpose |
|---|---|
| `dranzer.pth` | Versioned model, frozen tokenizer, architecture, and provenance bundle |
| `dranzer.pth.tokenizer` | Compatibility/reuse sidecar; not required to load a current bundle |
| `config.json` | Saved architecture and training configuration |
| `checkpoints/*.ckpt` | Atomic, complete training snapshots used by `--resume` |
| `checkpoints/run_*.manifest` | Unique read-only resolved run settings and explicit overrides |
| `bench_results.csv` | Machine-local benchmark history, created by `bench.out` |
| `gpu_capability_cache/` | Hardware probe cache, created when a GPU can be identified |

Paths are relative to the directory from which the executable is run.
Every training invocation creates a new manifest with exclusive-create semantics; an existing
manifest is never overwritten. Unknown options, missing values, malformed or overflowing numbers,
invalid enums, and out-of-range values are rejected before a run begins.

## Troubleshooting

### A model cannot be loaded

Train first, or pass the correct path explicitly:

```bash
./app.out infer --model /path/to/model.pth --prompt "hello"
```

### A build has stale objects

```bash
make clean
make
```

This is especially important when changing compiler or sanitizer options.

### Training uses too much memory

Reduce `--embedding-dim`, `--num-layers`, `--max-seq-len`, or `--batch-size`. Activation storage
grows with model width, layer count, and sequence length; attention probability storage grows
quadratically with sequence length.

### `--gpu` still runs on CPU

Build and run the capability probe:

```bash
make gpu-probe
./gpu_probe.out
```

It reports whether the driver, CUDA, NVML, DRM, and OpenCL components are usable and explains the
fallback. See the [GPU backend guide](gpu.md) for details.
