# Design and maturity checklist

[← Back to README](../README.md)

This is DRANZER's ordered roadmap and the source of truth for maturity work. The project aims to be
a small, inspectable, reproducible decoder-only transformer training and inference runtime written
entirely in C. It does not aim to compete with production-scale LLM frameworks.

## How this checklist is championed

- Work on one unchecked goal at a time, in the order listed, unless a correctness or security issue
  blocks it.
- Mark a goal complete only when its acceptance checks, tests, documentation, and relevant
  benchmarks are complete.
- Keep behavioral changes compatible where practical; document and test intentional format or API
  breaks.
- Prefer measurable workflow integrity over adding isolated model features.
- Record benchmark hardware and compiler settings with every performance claim.

Status labels are `ACTIVE`, `NEXT`, and `LATER`. There must be at most one `ACTIVE` goal.

## Current foundation

- [x] Causal multi-layer transformer with full hand-written backpropagation and numerical gradient
  checks.
- [x] AdamW and SGD, gradient clipping, dropout, and learning-rate scheduling.
- [x] Version-preserving model serialization round-trip tests.
- [x] Persisted BPE vocabulary sidecars with a legacy byte-tokenizer fallback.
- [x] Greedy, top-k, and top-p generation with temperature and reproducible seeds.
- [x] Per-layer KV-cached incremental decoding checked against full-prefix logits.
- [x] GCC, Clang, OpenMP, sanitizer, benchmark, CI, and nightly workflows.

## v0.2 — Trustworthy training

### T1. Frozen tokenizer pipeline — COMPLETE

- [x] Train the tokenizer in a dedicated corpus pass before model optimization begins.
- [x] Freeze token IDs and merge order for every epoch and training sample.
- [x] Support loading an existing tokenizer for training and reject vocabulary mismatches.
- [x] Record tokenizer configuration and a corpus/input fingerprint in the run configuration.
- [x] Add a regression test proving tokenization is identical before and after model training.

Acceptance gate: a training run never mutates its tokenizer after the first model update, and a
repeated run with the same tokenizer and seed produces the same token stream.

Completion evidence: frozen encoders reject further BPE training; focused tests preserve token IDs
through optimizer updates; repeated end-to-end runs produced byte-identical model artifacts.

### T2. Validation and evaluation — COMPLETE

- [x] Add an `eval` command that never updates model or optimizer state.
- [x] Report validation cross-entropy, perplexity, token count, and elapsed time.
- [x] Support explicit training and validation inputs without hidden data splitting.
- [x] Add a tiny-corpus overfit test and a held-out evaluation regression test.

Acceptance gate: every reference training run can report comparable held-out metrics.

Completion evidence: core tests snapshot parameters, gradients, Adam moments, scheduler counters,
and metrics across evaluation; a held-out tiny-corpus regression improves cross-entropy; CLI
integration covers train-time validation, standalone `eval`, corpus fingerprints, and an unchanged
model artifact.

### T3. Exact checkpoint and resume — COMPLETE

- [x] Wire periodic checkpoint saving and `--resume` into the training CLI.
- [x] Persist parameters, optimizer moments, scheduler state, step counters, RNG state, tokenizer,
  and configuration.
- [x] Write checkpoints atomically and retain a configurable number of recent checkpoints.
- [x] Test that interrupted-and-resumed training matches uninterrupted training within a documented
  tolerance.

Acceptance gate: stopping at a test checkpoint and resuming produces equivalent final state and
metrics.

Completion evidence: focused tests round-trip every persistent training buffer and continue both
models along a bit-identical dropout trajectory. CLI integration resumes a scheduled AdamW run from
a periodic checkpoint without model/tokenizer sidecars and requires byte-identical final model and
checkpoint artifacts. The exactness contract is bit-for-bit on the same executable and execution
backend.

### T4. Honest batching and strict CLI behavior — COMPLETE

- [x] Make `--batch-size` perform true minibatching or documented gradient accumulation.
- [x] Add deterministic shuffling and configurable gradient-accumulation steps.
- [x] Reject unknown options, missing values, overflow, and malformed numeric arguments.
- [x] Save an immutable run manifest containing resolved defaults and command-line overrides.

Acceptance gate: every exposed training flag changes documented behavior and has a focused test.

Completion evidence: minibatches average independently computed sample gradients before one
optimizer update; configurable accumulation expands the effective batch; deterministic bounded
shuffling uses a data-only RNG; strict parser tests reject malformed, overflowing, unknown, and
out-of-range input. Every run writes a unique read-only manifest. End-to-end exact resume remains
byte-identical with batching, accumulation, shuffling, dropout, and scheduling enabled.

### T5. Versioned model bundle — COMPLETE

- [x] Define one versioned artifact that associates weights, tokenizer, architecture, and training
  metadata.
- [x] Add endianness, numeric-type, tensor-shape, checksum, and bounds validation.
- [x] Use atomic writes and retain compatibility fixtures from older releases.
- [x] Fuzz corrupted and untrusted tokenizer/model inputs.

Acceptance gate: bundles fail safely when corrupt and older supported fixtures remain loadable.

Completion evidence: version 1 stores canonical little-endian binary32 weights, architecture,
frozen merge order, training metadata, and corpus provenance in one atomic artifact. Header, weight,
and tokenizer checksums plus exact-size/footer, shape-formula, workspace, numeric-type, and tokenizer
bounds reject malformed input before unsafe allocation. A deterministic mutation sweep runs under
AddressSanitizer; a byte-defined pre-bundle fixture and current legacy round trip remain readable.
CLI integration evaluates without a sidecar, rejects tokenizer overrides, and exact resume still
produces byte-identical final bundles. The format and compatibility policy are documented in
`docs/model-bundle.md`.

## v0.3 — Capable generation

### T6. Special-token contract — COMPLETE

- [x] Add explicit BOS, EOS, UNK, and PAD IDs without colliding with byte or learned BPE tokens.
- [x] Define where BOS/EOS enter training, evaluation, prompting, decoding, and bundles.
- [x] Preserve a documented legacy-tokenizer compatibility mode.
- [x] Stop generation on EOS and test that control tokens never leak as text.

Acceptance gate: token roles have stable IDs and identical semantics across train, eval, infer,
generate, checkpoint, and bundle paths; old tokenizers remain readable without reinterpretation.

Completion evidence: byte IDs remain 0–255 and PAD, UNK, BOS, and EOS have stable IDs 256–259;
learned special-mode merges begin at 260. Training and evaluation both model `BOS … EOS`, prompts
prepend BOS, structural controls are masked or omitted from text, and generation stops immediately
on EOS. Sidecars, checkpoints, bundles, configs, and manifests retain tokenizer mode and IDs. Current
portable round trips plus a byte-defined version-1 sidecar fixture prove legacy tokenizers retain
their original merge IDs. Focused forced-EOS and exact-resume tests pass under GCC, Clang, OpenMP,
AddressSanitizer, and size-optimized builds. The contract is documented in
`docs/special-tokens.md`.

### T7. Streaming and caller-controlled stopping — COMPLETE

- [x] Support caller-provided stop sequences.
- [x] Stream decoded tokens through CLI output and a callback-oriented runtime interface.
- [x] Add repetition penalty, minimum length, and sampling edge-case tests.

Acceptance gate: generation can stop incrementally without buffering the full decoded sequence and
library callers receive the same token stream as the CLI.

Completion evidence: CLI and C callers use the same callback decode loop. It withholds matching
tokenized stop markers, buffers only a possible stop prefix, flushes incomplete prefixes at the
length limit, and supports callback cancellation at the accepted prefix. Repeated `--stop`,
`--min-length`, and sign-aware `--repetition-penalty` controls are strictly parsed; PAD, UNK, BOS,
and unassigned vocabulary slots remain unsampleable. Focused forced-logit tests cover stop/minimum
interactions, EOS, callback streams, cancellation, penalties, invalid controls, and sampling edge
cases. GCC, Clang, OpenMP, AddressSanitizer, size builds, CLI smoke generation, and all integration
tests pass. The API and ordering contract are documented in `docs/generation.md`.

### T8. Long-context decode policy — COMPLETE

- [x] Replace the fixed-capacity KV cache with a documented sliding or ring-buffer context policy.
- [x] Benchmark prompt prefill and incremental decode independently.

Acceptance gate: generation can run beyond the original context capacity under an explicit policy,
and prefill/decode performance is reproducible independently.

Completion evidence: each layer stores the newest `max_seq_len` key/value rows in a chronological
ring and evicts one oldest row per token after capacity. Positions remain absolute; the stored
sinusoidal table is used inside the training window and the identical formula is evaluated later,
with the extrapolation limitation documented. Full-prefix logits still match before eviction, and
multi-wrap tests compare the ring against the same logical cache normalized to linear storage.
Callback generation produced 18 sequence tokens with an 8-token retained window in CLI smoke and
focused tests. The benchmark now times and records prompt prefill, growing-cache decode, and
steady-state full-ring decode separately in `bench_results_v2.csv`. GCC, Clang, OpenMP,
AddressSanitizer, size builds, tools, and integration tests pass.

## v0.4 — Efficient runtime

### T9. Reproducible performance baseline — COMPLETE

- [x] Record build, compiler, OS, CPU, and threading metadata with benchmark rows.
- [x] Add a repeatable profiling workflow before changing numeric kernels.
- [x] Preserve and test a portable scalar reference path for future dispatched kernels.

Acceptance gate: a benchmark or profile artifact identifies exactly how and where it was produced,
and optimized paths can always be compared with an independently selectable scalar reference.

Completion evidence: version-2 CSV rows include the benchmark tier, exact build command, compiler,
OS, CPU, online-core count, OpenMP version, and thread count alongside separated inference,
prefill, growing-cache decode, full-ring decode, training, and memory measurements. `make profile`
builds a frame-pointer-enabled binary with documented `perf stat` and `perf record` workflows. A
runtime-selectable portable scalar matmul is exercised both directly and through full and
KV-cached model paths, with worst observed scalar-versus-dispatch logit error below `1e-6`.
GCC, Clang, OpenMP, AddressSanitizer, size-optimized, benchmark, profile, strict-warning, and CLI
integration checks pass; the profiling binary was smoke-tested where `perf` itself was not
installed.

### T10. Profile-guided CPU matrix multiplication — COMPLETE

- [x] Add isolated matrix-multiplication benchmark shapes representative of prompt prefill,
  single-token decode, and training.
- [x] Compare the existing 64×64 tiled kernel with the scalar reference across supported benchmark
  tiers and compilers.
- [x] Tune loop order and tile sizes only where measurements show a repeatable improvement.
- [x] Record numerical error, throughput, and the hardware/build metadata for every selected path.

Acceptance gate: the default portable CPU kernel is selected from reproducible measurements,
outperforms the scalar reference on its target shapes without materially regressing small shapes,
and remains within the scalar-reference tolerance.

Completion evidence: the kernels moved into their own `core/matmul` module holding the scalar
reference, an unblocked row-wise kernel, the previous tiled kernel, and a four-row register-blocked
kernel, all runtime-selectable with a runtime tile size. `bench.out --matmul-only --sweep` measures
every kernel and tile candidate on six shapes taken from live `core/transformer.c` call sites
across all three tiers, and `tools/matmul_sweep.sh` repeats that across GCC and Clang and
summarises the winners. Candidates are checked against the scalar reference before being timed,
ranked by their fastest round with the median reported as a noise indicator, warmed up once, and
interleaved so contention cannot favour one candidate; every row carries build, compiler, OS, CPU,
core-count, and threading provenance.

The measurements selected a four-row register-blocked kernel at a 256 tile, replacing the 64-tile
kernel that had never been measured. It beats the scalar reference on all 36 measured
(compiler, tier, shape) combinations — 1.46× worst, 4.47× geometric mean, 9.20× best — is never
more than 1.40× off the fastest candidate for any shape, and is 1.13× faster than the previous
default in geometric mean. Whole-model GCC runs improve by 1.06–1.31× across inference, prefill,
decode, and training; the matching Clang whole-model runs did not reproduce across three batches on
this contended machine and are documented as inconclusive rather than claimed. Two shape-split
policies were measured and rejected for not holding across compilers or exceeding session noise;
`matmul_select()` therefore returns one kernel for every shape, which `test_matmul_kernels.c` pins
along with kernel/tile equivalence over eleven shapes, six tile sizes, and every kernel. Adding
`restrict` to the kernel signatures was itself worth 3–4× under Clang. GCC, Clang, OpenMP,
AddressSanitizer, size-optimized, benchmark, and integration checks pass, and exact resume remains
byte-identical. The kernels, policy, reproducibility contract, and measurement workflow are
documented in `docs/matmul.md`.

### T11. Runtime-dispatched SIMD kernels — NEXT

- [ ] Add runtime-dispatched AVX2/AVX-512 and ARM NEON kernels where supported.
- [ ] Detect CPU features at runtime so one binary stays portable across machines.
- [ ] Extend the kernel sweep and the equivalence test to cover every dispatched kernel.

Acceptance gate: dispatched kernels are numerically checked against the scalar reference on the
same shapes as the portable kernels, selected by the same reproducible measurement workflow, and a
binary built with them still runs correctly on a machine lacking the instruction set.

T10 left the seam this needs: `matmul_select()` already resolves a shape to a kernel, kernels are
runtime-selectable by name, and `test_matmul_kernels.c` checks every registered kernel against the
reference automatically.

### Later runtime goals

- [ ] Replace repeated OpenMP entry with a measured persistent worker strategy if beneficial.
- [ ] Add INT8 and then INT4 weight-only quantization with accuracy comparisons.
- [ ] Support memory-mapped weights and measure startup time and resident memory.
- [ ] Add nightly performance-regression thresholds on stable benchmark tiers.

Acceptance gate: each optimized path is numerically checked against the reference path and its
speed/memory improvement is reproducible on identified hardware.

## v0.5 — Modern model and stable library

- [ ] Evaluate tied embeddings, RoPE, RMSNorm, and GELU/SwiGLU one change at a time.
- [ ] Add padding and general attention masks before variable-length batching.
- [ ] Consider grouped-query attention only after quality and decode benchmarks exist.
- [ ] Expose opaque model, tokenizer, cache, and generation handles through a stable public C API.
- [ ] Remove terminal output from library code and return structured errors to callers.
- [ ] Produce static/shared library targets and small embedding examples.

Acceptance gate: architectural changes have quality, correctness, speed, and memory comparisons;
the CLI is a client of the same supported API available to external programs.

## v1.0 — Mature reference implementation

- [ ] Publish small reference models, their corpus provenance, and expected evaluation metrics.
- [ ] Guarantee and test the documented model-format and public-API compatibility window.
- [ ] Run fuzzing, full leak detection, and deterministic GCC/Clang build checks in release gates.
- [ ] Maintain semantic versions, a changelog, reproducible release artifacts, and migration notes.
- [ ] Document supported operating systems, CPU features, and optional GPU behavior precisely.

Acceptance gate: a new user can reproduce a reference result, embed the library, upgrade within the
compatibility window, and diagnose failures using only released artifacts and documentation.

## Deliberately deferred

Distributed training, production-scale serving, multiple GPU-vendor backends, and very large model
support remain out of scope until the v0.2 workflow is reproducible and the v0.3 runtime has
measurable quality and performance baselines.
