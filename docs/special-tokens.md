# Special-token contract

[← Back to README](../README.md)

Current tokenizers use a `special-v1` ID layout that preserves every historical byte ID:

| Role | ID | Text behavior |
|---|---:|---|
| byte values | 0–255 | Raw byte vocabulary; unchanged from legacy tokenizers |
| PAD | 256 | Reserved for future padded batches; never emitted by text encoding or generation |
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

PAD is reserved but not yet an attention mask. The current CLI uses unpadded variable-length
windows. Padding masks remain an explicit later architecture goal.

## Compatibility

Legacy `DRNZBPE1` tokenizers keep their original interpretation: byte IDs are 0–255 and learned
merges begin at 256. They do not silently gain BOS/EOS behavior. New `DRNZBPE2` sidecars and portable
bundle payloads record `special-v1` mode; checkpoints embed that same flag. Loading an old model and
sidecar therefore preserves its old token stream exactly, while new training defaults to the
special-aware mode.

Tokenizer mode is part of model compatibility. A legacy tokenizer cannot be substituted for a
special-aware bundle, and an explicit sidecar cannot override a bundle's embedded tokenizer.
