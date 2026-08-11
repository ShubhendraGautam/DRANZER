# Model bundle format

[← Back to README](../README.md)

DRANZER writes one self-contained model bundle at the path selected by `--model`. A bundle carries
the architecture, weights, frozen BPE merge order, and enough provenance to
identify the training run. `eval`, `infer`, and `generate` load the embedded tokenizer; they do not
need a `.tokenizer` sidecar.

Bundles are deployment and evaluation artifacts. They deliberately omit gradients, optimizer
moments, scheduler internals, loss history, RNG state, and the corpus cursor. Use a complete
`.ckpt` file for exact training resume.

## Version 1 contract

Every integer and float bit pattern is stored in little-endian order. Floats are IEEE-754 binary32.
The fixed 152-byte header is followed by the two checked payloads and an eight-byte terminal footer.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | Magic: `DRNZBNDL` |
| 8 | 4 | Format version: `1` |
| 12 | 4 | Endian marker: `0x01020304` |
| 16 | 4 | Numeric type: `1` (binary32) |
| 20 | 4 | Header size: `152` |
| 24 | 48 | Vocabulary, embedding, head, layer, maximum-sequence, and parameter counts as six `u64` values |
| 72 | 16 | Training-step count, header-checksum low half, current-loss bits, header-checksum high half |
| 88 | 32 | Training window, seed, input FNV-1a fingerprint, and input byte count |
| 120 | 32 | Weight/tokenizer byte lengths and their two FNV-1a checksums |
| 152 | variable | Canonical binary32 parameter payload |
| after weights | variable | Portable frozen-tokenizer payload |
| final 8 | 8 | Footer: `DRNZDONE` |

The tokenizer payload begins with `DRNZBPP2`, then stores its maximum and occupied vocabulary
sizes, frozen flag, special-token mode, and every learned token in merge order as length, signed
frequency, and bytes. Token bytes are length-delimited and may contain NUL. The deterministic byte
tokens 0–255 and fixed special IDs are reconstructed rather than duplicated. The loader retains
read support for the older unmarked payload, whose learned tokens were C strings.

The header checksum is calculated with its two split fields zeroed. FNV-1a checksums detect
accidental corruption; they are not cryptographic signatures. Do not use them to establish
artifact authenticity.

## Version 2 quantized weights

`model_bundle_save_quantized()` writes an opt-in version 2 artifact. The existing
`model_bundle_save()` continues to write version 1, so adding storage quantization does not change
an existing caller's bytes or accuracy. The common header fields keep their offsets; version 2
uses numeric type `2`, a 184-byte header, and appends:

| Offset | Size | Field |
|---:|---:|---|
| 152 | 4 | Symmetric-grid bit width (`2..16`) |
| 156 | 4 | Scale granularity (`0` tensor, `1` row, `2` column) |
| 160 | 4 | Policy flags (embeddings; biases and norms) |
| 164 | 4 | Tensor-record count |
| 168 | 8 | Number of values stored quantized |
| 176 | 8 | Number of float32 scales stored |

The weight payload follows the stable `model_param_tensors()` inventory. Every tensor starts with a
32-byte record containing its index, kind, row count, column count, representation, and a reserved
zero field. An excluded tensor is followed by canonical little-endian float32 values. An included
tensor is followed by its little-endian float32 scales and tightly packed codes, with each code
biased by `qmax` and written least-significant bit first. The unused all-ones code and non-zero tail
padding are invalid, making the encoding canonical.

Loading validates the record against the model-created inventory and against the policy in the
header, then dequantizes directly into the ordinary float parameter buffer. Runtime kernels and
memory representation therefore remain unchanged. The save API reports total artifact bytes,
weight-payload bytes, tokenizer bytes, and quantized tensor/value/scale counts so storage can be
compared to the accuracy report produced by `model_quantize_weights()`.

## Version 3 architecture flags

Version 3 carries architecture choices that change the parameter layout. Its 192-byte header keeps
all version-1/version-2 fields at the same offsets and appends:

| Offset | Size | Field |
|---:|---:|---|
| 184 | 4 | Architecture flags (`bit 0`: tied token/input embeddings and output projection; `bit 1`: RoPE) |
| 188 | 4 | Reserved zero |

A version-3 bundle may use numeric type `1` for lossless float32 parameters or type `2` for the
version-2 quantized tensor representation. For lossless files, bytes 152–183 are canonical zeros.
The ordinary writers preserve existing artifacts: an unflagged lossless model is still version 1
and an unflagged quantized model is still version 2. A flagged model uses version 3.

With tied embeddings, the parameter payload contains one `[vocabulary × embedding]` token table
and no independent `[embedding × vocabulary]` output projection. The head reads the table
transposed, and backward accumulates the output-head and input-lookup contributions into that one
gradient slice. Unknown architecture bits, a parameter count computed for the wrong layout, or a
non-zero reserved field are rejected before allocation.

## Loading and safety

The loader rejects unsupported versions, endian/numeric markers, header corruption, shape or
parameter-count inconsistencies, integer overflow, unexpected file sizes, missing footers,
checksum failures, trailing bytes, malformed lengths, embedded NULs in legacy tokenizer payloads,
partial input, tensor-record drift, invalid quantization policy fields, non-finite/negative scales,
unused packed codes, and non-zero packed padding. It validates
the parameter formula and quadratic attention-cache bound before allocating the model. Portable
tokenizer vocabularies are capped at 1,048,576 entries.

Saving writes a process-specific temporary file beside the destination, flushes and syncs it, then
publishes it with one atomic rename. A failed save removes the temporary file and leaves an existing
destination untouched.

## Read-only memory-mapped loading

`model_bundle_load_mmap()` maps a lossless version-1 or version-3 bundle with
`MAP_PRIVATE`/`PROT_READ` and lays every named parameter view directly over the canonical weight
payload. It still validates the header,
exact file size, footer, weight checksum, tokenizer checksum, model shape, and tokenizer vocabulary
before publishing the model. It avoids allocating, randomly initializing, and then overwriting a
second parameter buffer. `model_free()` owns and releases the mapping after a successful load.

A mapped model is explicitly inference-only. Training accumulation, optimizer steps, and direct
weight decay return `MODEL_INVALID_INPUT` before writing. Quantized version-2/version-3 bundles
return `BUNDLE_UNSUPPORTED` from this entry point because their packed codes and scales must be
unpacked;
use `model_bundle_load()` when a writable float model is required. Direct mapping also requires a
little-endian host with IEEE-754 binary32, matching the lossless payload bytes.

Build the focused measurement tool and run each mode in a separate process so peak RSS is not
contaminated by the other allocator strategy:

```bash
make -C src bench-bundle-load
./src/bench_bundle_load.out model.bin --mode copy --repeats 7
./src/bench_bundle_load.out model.bin --mode mmap --repeats 7
```

Each invocation reports median checked-load startup time and process peak RSS, then appends the
artifact size, parameter count, and full compiler/host provenance to
`bundle_load_results_v1.csv`. The implementation and measurement path are present, but their final
numbers are intentionally deferred to the project's bundled validation run.

## Compatibility policy

- Bundle versions 1 through 3 are the current compatibility window and remain readable throughout
  the 1.x release line. Version 1 remains the default lossless writer; version 2 is the default
  opt-in quantized writer; version 3 records parameter-layout architecture flags. Version identifiers
  are never reused. Removing a reader or changing a
  field's meaning requires a project major-version change and migration notes.
- Unknown bundle versions fail explicitly; the loader does not guess. Public callers receive the
  actual loaded version in `dranzer_bundle_info_t.format_version`.
- Files without bundle magic are offered to the read-only legacy host-native weight loader. The CLI
  then loads `<model>.tokenizer`, an explicit `--tokenizer`, or the historical byte-vocabulary
  fallback when no sidecar exists.
- An explicit `--tokenizer` cannot override a bundle's embedded vocabulary. The tokenizer and
  weights are one compatibility unit.
- Portable tokenizer mode `0` is legacy (merges start at 256); mode `1` is `special-v1` (fixed
  PAD/UNK/BOS/EOS IDs 256–259 and merges start at 260).
- Legacy output is no longer written by the CLI, but `model_save()` and `model_load()` remain for
  source compatibility and fixtures.
- Checkpoint compatibility is independent of bundle compatibility. A checkpoint is tied to exact
  resume state; a bundle is the smaller inference/evaluation artifact.

`test_model_bundle.c` pins the magic, numeric type, header size, footer, and reported version for
the original fixed wire formats. `test_tied_embeddings.c` pins version 3 through lossless,
quantized, copied, and mmap round trips. Together they protect exact version-1 round trips, mapped-model
ownership and training rejection, version-2 reconstruction against the
simulated quantizer, artifact-size accounting, both payload checksums, tensor-shape validation,
truncation, unsupported versions, unsafe shapes, malformed tokenizer bounds, a deterministic
mutation sweep, and legacy loading. CLI integration also deletes the sidecar before evaluation to
prove the bundle is self-contained.
