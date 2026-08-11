# Changelog

All notable changes are recorded here. The project uses [Semantic Versioning](https://semver.org/);
public C API compatibility is governed separately by `DRANZER_API_VERSION`, and model artifacts by
their bundle-format version.

## [Unreleased]

- Final bundled compiler, sanitizer, compatibility, and benchmark validation for 0.5.0.
- Publication of reference model artifacts remains pending on a verified corpus.

## [0.5.0-dev] - 2026-08-12

### Added

- Opaque API-v1 model, tokenizer, cache, and greedy-generation handles.
- Self-contained static/shared libraries, versioned shared exports, and embedding examples.
- Lossless mmap bundle loading and opt-in version-2 quantized bundle storage.
- Optional tied input/output embeddings with version-3 architecture-aware bundles and checkpoints.
- Optional rotary query/key position embeddings across training and cached decode.
- INT8 and packed-INT4 weight matmul primitives with correctness/performance coverage.
- Padding/general attention masks across inference and training.
- Structured library diagnostics and silent embedding-runtime behavior.
- Release compatibility, fuzz/leak, and reproducible GCC/Clang build gates.
- Explicit release, experimental, CPU ISA, mmap, OpenMP, and CUDA platform-support matrix.

### Changed

- Tokenizer byte sequences and serialization are length-delimited and preserve embedded NUL bytes.
- Model bundle/public API compatibility windows are explicit and mechanically checked.

### Migration

- See [docs/migrations.md](docs/migrations.md) for moving from internal structs, legacy model files,
  terminal diagnostics, and `libattention.a` to the supported 0.5 embedding boundary.
