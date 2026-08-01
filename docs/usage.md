# Usage and CLI

[← Back to README](../README.md)

This guide covers building DRANZER and using its `train`, `infer`, and `generate` modes.

## Requirements

- Linux
- Clang or GCC with C11 support
- GNU Make
- Standard C library, `libm`, and `libdl`

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
  --model dranzer.pth \
  --epochs 10 \
  --learning-rate 0.0005 \
  --optimizer adam \
  --dropout 0.1 \
  --grad-clip 1.0 \
  --warmup-steps 100 \
  --total-steps 5000
```

Training saves the final model to the path selected by `--model`, the learned BPE vocabulary to
`<model>.tokenizer`, and `config.json` in the current working directory. Use `--tokenizer FILE` to
choose a different sidecar path. The checkpoint utility module exists, but periodic checkpoint
saving and resume are not currently wired into the training CLI.

### Training and model options

| Option | Default | Meaning |
|---|---:|---|
| `--input FILE` | `../tests/chunk_aa` | Training corpus; passing it explicitly is recommended |
| `--model FILE` | `dranzer.pth` | Model output/input path |
| `--tokenizer FILE` | `<model>.tokenizer` | Learned BPE vocabulary path |
| `--epochs N` | `1` | Number of passes over the input |
| `--batch-size N` | `1` | Token stream batch setting |
| `--learning-rate LR` | `0.001` | Initial learning rate |
| `--checkpoint-dir DIR` | `checkpoints` | Reserved checkpoint directory |
| `--checkpoint-interval N` | `10` | Reserved checkpoint interval |
| `--vocab-size N` | `257` | Token vocabulary size |
| `--embedding-dim N` | `16` | Hidden width; must divide evenly by the head count |
| `--num-heads N` | `2` | Attention heads per layer |
| `--num-layers N` | `2` | Transformer block count |
| `--max-seq-len N` | `32` | Allocated context capacity |
| `--train-window N` | `16` | Sliding training context, clamped to max sequence length |
| `--optimizer NAME` | `adam` | `adam` or `sgd` |
| `--dropout RATE` | `0.0` | Sublayer-output dropout rate |
| `--grad-clip NORM` | `1.0` | Global gradient norm limit; `0` disables it |
| `--weight-decay W` | `0.01` | Decoupled AdamW weight decay |
| `--warmup-steps N` | `0` | Linear learning-rate warmup length |
| `--total-steps N` | `0` | Warmup/cosine horizon; `0` uses plateau decay |

## Inference

Inference loads a trained model and predicts its next token:

```bash
./app.out infer --model dranzer.pth --prompt "Once upon a time"
```

`--prompt` is required. Inference loads the tokenizer sidecar associated with the model and applies
the selected decoding strategy to the next-token logits. Add `--gpu` to request GPU matmul; the
runtime falls back to CPU when a usable CUDA device is not present.

## Generation

```bash
./app.out generate \
  --model dranzer.pth \
  --prompt "In the beginning" \
  --length 20 \
  --sampling topp \
  --top-p 0.9 \
  --temperature 0.8 \
  --seed 42
```

Generation stops when it reaches either `--length` new tokens or the model's `--max-seq-len`
capacity. Prompt tokens are processed once to populate a per-layer KV cache; each subsequent step
computes only the new query, key, value, and hidden state.

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
| `--seed N` | `1` | Reproducible random seed for sampling |

## General options

| Option | Meaning |
|---|---|
| `--gpu` | Use NVIDIA GPU matmul when available; otherwise fall back to CPU |
| `--debug` | Enable runtime debug output |
| `--help`, `-h` | Print the complete command reference |

## Generated files

| File | Purpose |
|---|---|
| `dranzer.pth` | Serialized model weights and training state |
| `dranzer.pth.tokenizer` | Learned BPE tokens and merge order |
| `config.json` | Saved architecture and training configuration |
| `bench_results.csv` | Machine-local benchmark history, created by `bench.out` |
| `gpu_capability_cache/` | Hardware probe cache, created when a GPU can be identified |

Paths are relative to the directory from which the executable is run.

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
