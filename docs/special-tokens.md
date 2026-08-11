# Special-token contract

[← Back to README](../README.md)

Current tokenizers use a `special-v1` ID layout that preserves every historical byte ID:

| Role | ID | Text behavior |
|---|---:|---|
| byte values | 0–255 | Raw byte vocabulary; unchanged from legacy tokenizers |
| PAD | 256 | Padding sentinel; never emitted by text encoding or generation |
| UNK | 257 | Fallback when an internal token lookup cannot be represented; never emitted for ordinary bytes |
| BOS | 258 | Begins each corpus document and model-visible prompt |
| EOS | 259 | Ends each corpus document and stops generation |
| learned BPE merges | 260 onward | Merge order learned from the corpus |

`--vocab-size` includes the four control IDs, so a fresh tokenizer requires at least 260. A size of
260 is byte-level plus controls with no learned merges.

## Sequence semantics

- Training treats one input file as one document and optimizes the next-token sequence
  `BOS, BPE(file bytes), EOS`. The sliding context is retained across streaming chunks, so chunk
  boundaries neither lose nor duplicate predictions.
- Validation and standalone evaluation score the identical sequence. Token/prediction counts
  therefore include BOS and EOS for current bundles; corpus byte counts and fingerprints do not.
- Inference and generation prepend BOS to the encoded prompt. BOS occupies one context slot; when a
  prompt is too long, the newest prompt tokens are retained after BOS.
- PAD, UNK, and BOS logits are masked before sampling. EOS remains eligible after any configured
  minimum length, and the shared decode loop stops immediately after sampling it.
- Decoding omits all four control IDs, so they never appear as literal marker strings in output.

The core mask-aware APIs accept PAD positions through a separate boolean padding mask; the token ID
alone never changes attention semantics. This keeps legacy/plain tokenizers usable and lets callers
pad with any storage value as long as the corresponding mask entry is zero. The current CLI still
uses unpadded variable-length windows; variable-length batch collation is a separate later step.

## Compatibility

Legacy `DRNZBPE1`/`DRNZBPE2` tokenizers remain readable and keep their original ID interpretation:
plain mode uses byte IDs 0–255 and merges from 256; special mode reserves IDs 256–259 and merges
from 260. Binary-safe sidecars are `DRNZBPE3` for plain mode and `DRNZBPE4` for special mode. Bundle
payloads carry their own `DRNZBPP2` version marker and record the mode separately. Loading an old
model and sidecar therefore preserves its old token stream exactly, while new training defaults to
the special-aware, binary-safe format.

Tokenizer mode is part of model compatibility. A legacy tokenizer cannot be substituted for a
special-aware bundle, and an explicit sidecar cannot override a bundle's embedded tokenizer.
