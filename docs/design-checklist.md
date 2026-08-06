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
steady-state full-ring decode separately in `bench_results_v3.csv`. GCC, Clang, OpenMP,
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
KV-cached model paths, against a `3e-5` tolerance; worst observed scalar-versus-dispatch logit error
was below `1e-6` when dispatch meant the portable kernels, and is `1.43e-6` since T11 made it
AVX-512 (the extra FMA contraction is expected and stays far inside the tolerance).
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

### T11. Runtime-dispatched SIMD kernels — COMPLETE

- [x] Add runtime-dispatched AVX2/AVX-512 and ARM NEON kernels where supported.
- [x] Detect CPU features at runtime so one binary stays portable across machines.
- [x] Extend the kernel sweep and the equivalence test to cover every dispatched kernel.

Acceptance gate: dispatched kernels are numerically checked against the scalar reference on the
same shapes as the portable kernels, selected by the same reproducible measurement workflow, and a
binary built with them still runs correctly on a machine lacking the instruction set.

Completion evidence: `avx2_mr4`, `avx512_mr4`, and `neon_mr4` live in `core/matmul_x86.c` and
`core/matmul_arm.c` behind `core/matmul_simd.h`, which is the single place deciding which
architectures have an implementation. They are `kernel_tiled_mr4` with the innermost loop over `j`
issued as vector FMAs; vectorizing along `j` rather than `k` keeps every `C` element accumulating
`l` in increasing order, so they differ from the reference by FMA contraction rather than by
reassociation. Vector code sits in functions carrying `target` attributes, so the default build
still passes no `-march` and the wide instructions are unreachable unless `cpu_features.c` says the
CPU has them. That detection gates on `XCR0` via `XGETBV` as well as on the CPUID feature bit,
because an OS that has not enabled the extended register state leaves the bit set while the first
instruction faults.

The sweep selected one kernel and rejected two, all three as measured outcomes. `avx512_mr4` beat
`tiled_mr4` on all 36 (compiler, tier, shape) combinations — 1.07x worst, 1.83x geometric mean,
3.02x best, and 8.78x geometric mean over the scalar reference — with GCC and Clang independently
landing on 1.83x. `avx2_mr4` measured 1.35x under GCC and 0.90x under Clang, consistently across all
three tiers; since Clang is this project's default compiler, it ships selectable but never selected,
which is the same call T10 made for the `rowwise` shape split in mirror image. Disassembly confirms
both compilers emit packed FMAs, so the split is a scheduling difference and is recorded as an open
question rather than resolved. `neon_mr4` is unmeasured — no AArch64 hardware was available — and is
also the one kernel offering no instruction the baseline target lacks, since NEON is mandatory on
ARMv8-A and `tiled_mr4` already compiles to it; it ships unselected rather than assumed good.

`test_matmul_kernels.c` checks every *available* kernel at every sweepable tile over 15 shapes that
now include the 8- and 16-float vector boundaries, reports which kernels were skipped rather than
counting them as passing, and pins that the two rejected kernels stay unselected so promoting either
requires re-running the sweep. The unavailable-instruction-set gate is exercised by capping
detection with `cpu_features_set_max_isa(CPU_ISA_BASELINE)`, which is what such a CPU would report:
every kernel including the SIMD ones still computes correctly through the fallback, and selection
returns to the portable kernel. That simulates the dispatch path, not the absence of the
instructions themselves, and `docs/matmul.md` records the distinction. GCC, Clang, OpenMP,
AddressSanitizer, size-optimized, and `-Wpedantic -std=c11` builds all pass.

End to end, the same binary run with the instruction set capped and uncapped (six runs each,
interleaved ABBA, small tier, Clang) improves full-context inference 1.29x and prompt prefill 1.26x,
with the inference distributions not overlapping at all. Training shows no measurable change, which
is the documented consequence of leaving the backward-pass matmuls undispatched rather than a
surprise; it is now the first entry under later runtime goals.

Runtime dispatch narrowed one existing contract, which is documented rather than quietly changed:
bit-identity was always scoped to one executable and one execution backend, and the CPU's
instruction set is now part of that backend. Exact resume on a single machine is unaffected;
comparing artifacts across machines needs `DRANZER_CPU_ISA` pinned on both. Benchmark rows carry the
detected instruction set in a new `simd` provenance column, which moved both results files to
`bench_results_v3.csv` and `matmul_results_v3.csv`. Whole-model scalar-versus-dispatch logit error
moved from below `1e-6` to `1.43e-6` against the test's unchanged `3e-5` tolerance, which is the
expected cost of FMA contraction and is recorded in T9's evidence rather than left stale.

### Later runtime goals

- [x] Dispatch the backward-pass matmuls to the **GPU** (`backends/gpu/gpu_matmul.c`,
  `core/training.c`). Two hand-written PTX kernels join the forward one in a single module, with
  accumulate-into-destination semantics matching the CPU contract. Because a backward call moves
  four buffers where a forward call moves two — both inputs up, plus the accumulating destination
  up *and* down — they dispatch on a measured shape threshold rather than whenever a GPU exists.
  Medium-tier training improved 969.3 → 627.9 ms/step (1.54x, non-overlapping ranges over three
  runs each), the first whole-model training win the GPU backend has produced. Both thresholds are
  now 2²³; the originally-measured 2²⁰ for `backward_weight` was re-derived after the CPU
  loop-order fix below, which also reduced the whole-model figure above to 536.4 → 477.4 ms/step
  (1.12x) by making the CPU side faster. Correctness is covered by
  `test_gpu_matmul_backward.c` (both kernels against the CPU reference, accumulating twice into a
  non-zero destination, on shapes with no extent a multiple of the thread block) and
  `test_gpu_training_backward.c` (CPU/GPU agreement on a model deliberately sized so the thresholds
  are crossed — the pre-existing `test_gpu_training_step.c` model is small enough that every
  backward shape falls below them). Documented in `docs/gpu.md`.
- [x] **Fix `matmul_backward_weight()`'s loop order** (`core/matmul.c`). It ran `i` innermost,
  striding `A` by `k` and `dC` by `n` simultaneously, so every iteration of the hot loop touched two
  fresh cache lines to use one float from each. Moving `j` innermost walks `dB` and `dC`
  contiguously and reduces `A` to one scalar load per `(l, i)`; four rows of `dB` are kept in
  flight for the same reason `tiled_mr4` keeps four rows of `C`. Per-shape improvement is 5.6x to
  21.8x, and whole-model medium-tier CPU training improved **1125.3 -> 408.6 ms/step (2.75x)**,
  measured with old and new binaries interleaved ABBA in one session, best of six each,
  non-overlapping ranges, with inference as an unchanged control reading 77.9 ms/token on both
  sides (ratio 1.00x).

  This is the largest single speedup in the project so far and it came from a loop order, not from
  SIMD or a GPU - nearly twice what dispatching the backward pass to the GPU was worth, and unlike
  that, it helps every user. It also invalidated the GPU thresholds set immediately above: with the
  CPU 5.6-21.8x faster, `backward_weight`'s crossover moved out from 2^20 to 2^23 to meet
  `backward_input`'s, and the GPU's advantage on that function fell from up to 30x to 3-5x. The GPU
  did not get worse; the baseline got better, which is the standing caution for every speedup ratio
  in this repository. The change reassociates the sum, so results differ in the last bits from
  earlier builds - the same class of difference as switching matmul kernels, covered by the
  gradient checks' tolerances rather than by bit-identity, and still deterministic within one build.
  Documented in `docs/matmul.md`.
- [x] Give the backward matmuls **SIMD** kernels (`core/matmul_x86.c`). Both gained an AVX-512 path
  beside the forward kernels, gated the same way through `cpu_isa_available()`: the weight kernel is
  an axpy with the same four-accumulator shape as `block_avx512`, the input kernel a reduction that
  consumes four rows of `B` at once so one load of `dC` feeds four independent chains reduced across
  lanes at the end. AVX-512 only — `avx2_mr4` measured as a regression under Clang and `neon_mr4`
  was never measured, and unlike the forward path there is no `--kernel` here to select them, so an
  AVX2 or NEON version would be unreachable code.

  Per-shape 2.0x to 3.2x, and whole-model medium-tier CPU training **333.5 -> 256.2 ms/step
  (1.30x)**, measured with two binaries differing only in the backward dispatch, interleaved ABBA,
  best of six each, non-overlapping ranges, inference as an unchanged control at 1.00x. Together
  with the loop-order fix above, both measured the same controlled way, medium-tier CPU training
  improved 2.75x x 1.30x = **3.58x**.

  `test_matmul_backward.c` reaches the portable path by capping detection to baseline and then
  requires the dispatched result to match a *direct* AVX-512 call bit for bit — comparing capped
  against uncapped alone would pass equally well if dispatch had silently never fired. Its inputs
  are scaled by 1/7 rather than a power of two for the same reason: an earlier 1/8 made every
  product and partial sum exactly representable, so both paths agreed to the bit and the comparison
  proved nothing. This also moved the GPU thresholds a third time, to 2^25 / 2^26; `backward_input`
  now loses on the GPU at every shape this project benchmarks. Documented in `docs/matmul.md`.
- [x] **Resolve or retire `avx2_mr4`** — resolved. Disassembling both builds located the cause:
  `fmadd(a, b, load(c))` puts the loaded value in the addend position, which x86 can encode as an FMA
  memory operand. GCC folded all four such loads per iteration, Clang folded one of four and issued
  three extra `vmovups`; the timing gap tracked the instruction count. It was an encoding choice, not
  the scheduling difference previously assumed.

  Both x86 kernels were rewritten to hold their accumulators in registers across the whole `k` range
  rather than reading and writing `C` on every step, which removes the load instead of coaxing a
  compiler into folding it. `avx2_mr4` went from 1.47x/0.92x (GCC/Clang) to **1.95x/2.02x**, and the
  compilers now agree. `matmul_select()` therefore selects it where AVX-512 is absent, restoring the
  width ordering the original measurements had correctly forbidden.

  **The same defect was costing `avx512_mr4` a further 1.7x** — it had the identical structure and
  had been the shipped default for months on the strength of beating the portable kernel 1.83x,
  never having been compared against a better version of itself. It now measures ~3.1x (GCC) and
  ~3.2x (Clang). Between the two rewrites the restructured 8-wide kernel beat the un-restructured
  16-wide one on five of six shapes: width does not help a loop whose limit is memory traffic.

  `neon_mr4` received the same restructure on the structural argument, being unmeasurable here. That
  surfaced a gap worth its own note: `core/matmul_arm.c` is behind `#ifdef DRANZER_HAVE_NEON`, so on
  x86 — every machine and CI runner this project uses — its body had never been compiled by
  anything, and a syntax error in it would have shipped undetected. `make arm-check` now
  cross-compiles it with clang, needs no AArch64 machine, and confirms it emits `fmla`.

  Documented in `docs/matmul.md`.
- [ ] Measure `neon_mr4` on real AArch64 hardware and either promote it in `matmul_select()` or
  remove it. It is correctness-checked, now cross-compiled by `make arm-check`, and carries the same
  register-resident structure as the x86 kernels — but it has never been timed. Note that the reason
  previously given for doubting it, that a hand-written kernel cannot beat a compiler targeting a
  baseline instruction set, was falsified by the x86 rewrite above and should not be reused.
- [ ] Replace repeated OpenMP entry with a measured persistent worker strategy if beneficial.
- [ ] Add INT8 and then INT4 weight-only quantization with accuracy comparisons.
- [ ] Support memory-mapped weights and measure startup time and resident memory.
- [x] Run the full benchmark nightly on hosted runners and gate on same-run performance invariants
  (`.github/workflows/performance.yml`, `src/tools/perf_check.py`): kernel versus scalar reference,
  shipped kernel versus the fastest candidate beside it, KV-cache benefit, and numerical tolerance.
  Absolute timings are recorded as artifacts but never asserted on, because hosted-runner speed is
  not reproducible across runs.
- [ ] Add cross-run regression thresholds on stable benchmark tiers, once enough nightly artifacts
  exist to establish what a normal run-to-run spread looks like on that runner class.

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
