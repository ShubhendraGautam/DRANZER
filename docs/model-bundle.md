# Model bundle format

[← Back to README](../README.md)

DRANZER writes one self-contained model bundle at the path selected by `--model`. A bundle carries
the architecture, canonical float32 weights, frozen BPE merge order, and enough provenance to
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

## Loading and safety

The loader rejects unsupported versions, endian/numeric markers, header corruption, shape or
parameter-count inconsistencies, integer overflow, unexpected file sizes, missing footers,
checksum failures, trailing bytes, malformed tokens, embedded NULs, and partial input. It validates
the parameter formula and quadratic attention-cache bound before allocating the model. Portable
tokenizer vocabularies are capped at 1,048,576 entries.

Saving writes a process-specific temporary file beside the destination, flushes and syncs it, then
publishes it with one atomic rename. A failed save removes the temporary file and leaves an existing
destination untouched.

## Compatibility policy

- Version 1 bundles are read exactly as specified above. Unknown bundle versions fail explicitly;
  the loader does not guess.
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

`test_model_bundle.c` protects exact round trips, both payload checksums, truncation, unsupported
versions, unsafe shapes, malformed tokenizer bounds, a deterministic mutation sweep, and legacy
loading. CLI integration also deletes the sidecar before evaluation to prove the bundle is
self-contained.
