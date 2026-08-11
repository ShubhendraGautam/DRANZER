# Design and maturity checklist

[← Back to README](../README.md)

This is DRANZER's ordered roadmap and the source of truth for maturity work. `docs/research-checklist.md`
is the companion along the other axis: not what the software should do next, but what would stop an
outside reader from believing or reproducing a result it produced. Several of its P0 items were
defects in code this file already marked COMPLETE; those are now fixed, with the evidence recorded
against each item there.

Two documents were split out of this file and should be read instead of it for their subjects:
[`docs/results.md`](results.md) holds every measured finding, and
[`docs/reproducibility.md`](reproducibility.md) holds the per-axis reproducibility contract. This
file links to them rather than restating them — the findings below remain as completion evidence for
the goals they closed, but the citable version of each lives in `results.md`. The project aims to be
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
- **A difference is only a result if it exceeds the measured noise of the thing being compared.**
  Establish that noise floor first — repeated runs for timings, repeated seeds for quality — and
  report a comparison against it. "1.4x faster" and "0.3 lower perplexity" mean nothing on their
  own, and this project has already had three sessions of one sweep disagree with each other
  (`docs/threading.md`).
- Record what a measurement *cannot* resolve as carefully as what it can. An honest "this model is
  too small for that delta to be visible" is a finding; a number quoted without its uncertainty is
  not.

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

- [x] **Supervise every sequence position, not just the last** (`core/lm_head.{c,h}`,
  `core/matmul_dispatch.{c,h}`, `core/training.c`, `core/transformer.c`, `cli/batch.c`,
  `cli/main.c`).

  The output head read `cache_hidden[num_layers][seq_len-1]` and the loss had one target, so a
  forward+backward over a length-T window produced exactly **one** gradient signal. Every other
  position's hidden state was computed in full — all of attention, all of the FFN — and discarded.
  The causal mask already guarantees position `i`'s final hidden state depends only on tokens
  `0..i`, which is precisely the condition that makes supervising every position valid, so the same
  pass yields T signals for the cost of turning three m=1 matmuls into m=T.

  **Per supervised token, best of 7 ABBA-interleaved rounds** (clang, -O3 -ffast-math, AVX-512,
  serial): tiny 0.189 → 0.0084 ms (**22.6x**), small 7.023 → 0.116 ms (**60.5x**), medium 224.9 →
  2.008 ms (**112.0x**). The ceiling is T by construction — 32, 64, 128 — so this reaches 71%, 95%,
  and 88% of it. What it costs is 1.41x / 1.06x / 1.14x more time *per call*, which is the whole
  reason the ratio lands near T: the head was always the cheap part of a pass that had already paid
  for T positions of attention and FFN.

  Read those against the rest of this section. Every kernel result above — the loop-order fix, the
  backward AVX-512, the GPU backward dispatch — multiplies out to 3.58x on medium-tier training.
  This is 112x on the same tier, and it came from deleting a `-1` from an index. It is not a faster
  implementation of the same work; it is the removal of work that never needed doing, which is a
  different category and should not be quoted alongside the kernel ratios as though it competed
  with them.

  The tiny tier's rounds spread 0.268–0.450 ms against a 0.189–0.218 ms baseline, so its 22.6x is
  the least trustworthy of the three and is reported as the noisiest rather than the smallest.

  Correctness rests on one claim and `tests/core/test_all_position_loss.c` tests exactly it: one
  all-position pass over a length-T window must equal the **mean of T single-target passes** over
  windows of length 1..T, since each of those can only see tokens up to its own target. It does,
  worst relative 5.4e-5 on entries near 2e-4 (absolute ~1e-8). A leak of future context would
  diverge here and nowhere else in that file, which is why the other three cases are described in
  it as localizing a failure rather than establishing correctness.

  One methodology note, because it invalidated a reading before being caught. The comparison first
  used a bare relative difference with a 1e-3 floor and reported a 4.9e-5 "failure" on an entry
  whose true values were -4.9e-8 and exactly 0.0 — noise divided by nothing, on a gradient that was
  correct. The criterion is now `|a-b| <= ATOL + RTOL*max(|a|,|b|)` with both terms load-bearing
  over different parts of the range, and the test prints both so a real failure says which kind it
  is.

  Separately, `-MMD -MP` header dependency tracking was added to `src/Makefile`. It was missing, so
  adding a field to `neural_model_t` rebuilt `core/model.o` and nothing else: `core/bundle.o` and
  all of `cli/` kept the old struct layout and the binary aborted in `free()` at the end of an eval
  run, with both translation units individually correct and merely disagreeing about
  `sizeof(neural_model_t)`. Any future header change was exposed to the same failure.

  **The CLI loop advances by a whole window now**, behind `--train-stride` (0 = full window). It
  slid by one token per target before, because with only the last position supervised that was the
  only way to reach every target; the same stride under all-position supervision would present
  every target T times per epoch and throw the win away. Stride 0/2/1 at `--train-window 4` gives
  34/68/135 optimizer steps on the same corpus — linear in the stride, with stride 1 recovering the
  old per-target example count of 136 and confirming the two schemes cover the same targets.

  **End to end, one epoch over a 20 KB corpus at `--train-window 32`** (embedding 64, 2 layers,
  ABBA-interleaved, three rounds each): 11044 passes at stride 1 against 346 at stride 32, best of
  42.41 s against 3.64 s. That is 11.7x wall clock against a 31.9x reduction in passes, and the gap
  is not a shortfall in the change — solving the two points for a fixed cost gives 3.63 ms per pass
  and **2.38 s of tokenizer/IO/validation that neither stride can avoid**, which is 65% of the
  3.64 s run. The training portion scales at the full 31.9x and the end-to-end figure approaches it
  as the corpus grows.

  That number is reported with a caveat this project's methodology demands: the stride-1 rounds
  ranged 42.4–88.4 s, a 2x session spread on a contended machine. The direction is far outside that
  noise and the pass-count ratio is exact, but 11.7x should not be quoted to two significant
  figures.

  Two contracts moved and both are enforced rather than documented-and-hoped. `CHECKPOINT_VERSION`
  went to 3, because an example is now a window and the resume cursor counts windows where a
  version-2 cursor counted targets — reinterpreting one in the other's unit would resume at the
  wrong place silently, so version-2 files are rejected outright. `--train-stride` joined the
  exact-resume rejection list beside `--train-window`. The model bundle format was deliberately
  **not** touched: nothing about loading a model depends on the stride it was trained with, and the
  manifest and checkpoint both carry it.

  Users should expect step counts to change for the same corpus and `--epochs` — roughly
  `train_window / train_stride` times fewer — which `docs/usage.md` states, along with the
  consequence for `--total-steps` schedules. The integration test's expected total moved 34 -> 10
  for the same reason and its derivation is written into the assertion rather than left as a bare
  constant.

  Not measured: whether overlapping windows buy any held-out quality. The first position of a
  non-overlapping window has one token of context, and a smaller stride fixes that at proportional
  cost. The flag exists so the comparison is possible; making it is a v0.5 experiment and must be
  run against the seed-variance floor, which at these model sizes may well swamp it.

  GCC, Clang, GCC+OpenMP, ASan, size-optimized, `-Wpedantic -std=c11`, and both CLI integration
  checks pass; exact resume remains byte-identical.
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
- [x] **Give the backward matmuls an AVX2 rung** (`core/matmul_x86.c`, `core/matmul.c`,
  `core/matmul_simd.h`) — found by CI, not by a sweep.

  An AVX2-only hosted runner failed `test_perf_invariants.c`'s backward-versus-forward cost
  invariant at 4.09x against a 3.00x limit. The invariant's message blames cache traversal; the
  real cause was that `matmul_select()` ships `avx2_mr4` for the *forward* path, so a CPU with AVX2
  and no AVX-512 ran a vectorized forward against a portable backward. The ratio was measuring a
  missing kernel. Reproduced exactly on this AVX-512 machine with `DRANZER_CPU_ISA=avx2` (2.78x
  here, 4.09x there — same structure, different microarchitecture), and confirmed identical on
  unmodified HEAD, so it predates the all-position work above and is not a regression from it.

  **The justification for AVX-512-only had been falsified and left in place.** It rested on two
  claims: that `avx2_mr4` measured as a regression under Clang, and that the backward functions
  have no `--kernel` selection so an AVX2 version would be unreachable code. The first was
  overturned by the register-resident rewrite three entries above — the same change that promoted
  `avx2_mr4` in `matmul_select()` wherever AVX-512 is absent. The second confused *unselectable by
  flag* with *unreachable*: hardware selects these, and an AVX2-only CPU reaches an AVX2 version on
  every call. The sentence survived unchanged for months, load-bearing and wrong, in
  `core/matmul_simd.h` and `docs/matmul.md`.

  Portable → AVX2, ABBA-interleaved, best of seven: `backward_input` 1.63–2.37x, `backward_weight`
  1.15–1.89x. The weight kernel's two weakest rows are both output-head shapes where `k*n` is 64000
  and 1024000, so the axpy walks a destination far past cache and is bandwidth-bound; the FFN
  shapes land at the ~1.9x eight-wide FMA predicts. The invariant now reads **1.34x at AVX2**,
  against 2.78x before and 2.27x at AVX-512 on the same machine.

  `test_matmul_backward.c` checked only the widest rung, which is how these kernels could have
  shipped unverified — on an AVX2-only machine it printed `PASSED (portable only)` and proved
  nothing about the code running there. It now sweeps every available rung, capping *at* each level
  so dispatch must land exactly on it. Writing that surfaced a second-order trap worth recording:
  `cpu_features_clear_max_isa()` clears the `DRANZER_CPU_ISA` cap as well, so re-reading
  availability after a clear reports the silicon rather than the simulation — the first version of
  the fix quietly tested AVX-512 while claiming to test baseline. Availability is now sampled once,
  before anything touches the cap.

  Full suite passes at `DRANZER_CPU_ISA` of baseline, avx2, and uncapped, under both GCC and Clang,
  plus both CLI integration checks. Documented in `docs/matmul.md`.
- [ ] Measure `neon_mr4` on real AArch64 hardware and either promote it in `matmul_select()` or
  remove it. It is correctness-checked, now cross-compiled by `make arm-check`, and carries the same
  register-resident structure as the x86 kernels — but it has never been timed. Note that the reason
  previously given for doubting it, that a hand-written kernel cannot beat a compiler targeting a
  baseline instruction set, was falsified by the x86 rewrite above and should not be reused.

  **Blocked on hardware, not on work.** No AArch64 machine is available to this project, and
  emulation cannot answer the question — the decision is a timing comparison, and an emulated
  timing measures the emulator. This is why the threading goal below was taken out of order; it is
  the only item in this section skipped for that reason, and it should be picked up the moment an
  ARM machine is at hand.
- [x] **Replace repeated OpenMP entry with a measured persistent worker strategy if beneficial** —
  measured, and the persistent strategy was rejected in favour of not entering the region at all
  (`core/parallel.{c,h}`, `tools/bench_parallel.c`, `docs/threading.md`).

  Entering a region costs 0.475–1.29 µs here (GCC/libgomp, two threads). A single-token decode
  matmul at the tiny tier takes 60 ns, so the dispatch was eight times the computation, and with the
  measured 256 tile most decode and prefill shapes decompose into one or two blocks — the team was
  woken so one thread could do everything. A spinning persistent pool was prototyped and does
  dispatch for 0.073–0.337 µs, a real 2–15x, but that solves the smaller half: where dispatch
  dominates, even 0.073 µs exceeds the whole computation, and above the crossover region entry is
  already under 2% of a call. `parallel_should_fork(chunks, work)` therefore gates every OpenMP loop
  in `matmul.c`, `matmul_x86.c`, `matmul_arm.c`, and `transformer.c` on having at least two chunks,
  enough work, and more than one thread.

  OpenMP's own `if` clause cannot do this: `if(0)` measured 0.367–0.492 µs against 0.475–1.29 µs for
  `if(1)` and 0.0013 µs for no pragma, because libgomp still builds a team of one and calls the
  outlined function. It skips the threads, not the region. Hence `DRANZER_PARALLEL_FOR`, a macro
  precisely so the explicit branch's two copies of the loop cannot drift; the blocked kernels'
  `collapse(2)` nest was flattened to a single block index to feed it, which a static schedule
  distributes identically.

  The threshold is bracketed rather than derived: the largest shape measured to lose is 4160
  multiply-adds (1×16×260, 0.21–0.25x — forking costs 4–5x) and the smallest to win is 64000
  (1×64×1000, 1.16–1.37x), while whole-model runs showed serializing the small tier's decode head
  loop cost 1.19–1.45x, so it must be admitted at an estimated 13824. 2^13 sits near the geometric
  middle of (4160, 13824]. Getting there required pricing softmax at 22 multiply-adds
  (3.4 ns against 0.155 ns, measured): omitted, the decode head-loop estimate undercounts by 40% and
  the small tier lands on the wrong side — a regression that only whole-model measurement caught,
  after the isolated shapes already looked clean. No scalar-versus-vector conversion factor was
  needed; near the threshold both measured 0.145 vs 0.156 ns per multiply-add, because a shape that
  small waits on memory either way.

  Whole-model, ABBA-interleaved, two binaries differing only in the cutoff (GCC, `OMP=1`, two
  threads): tiny improves prefill **3.42x**, growing-cache and ring decode **2.29x**, inference
  1.22x, training 1.15x; small improves prefill 1.25–1.45x, growing-cache decode 1.27x, ring decode
  1.11–1.22x. Medium is flat at 0.97–1.04x, which is the control — every shape there is far above
  the threshold — and its flatness is what makes the other two columns credible. The default build
  is serial and measured 0.92–1.05x before versus after, i.e. unchanged.

  Two methodology results are recorded because they changed conclusions. The tool's own prototype
  pool was left spinning through the crossover sweep at first and held one of two cores for it,
  swinging the same shape between 0.86x and 2.62x across sessions; and shapes whose decomposition
  yields fewer than two blocks run identical code on both sides, so the tool now prints their spread
  as an explicit noise floor (0.81–1.14x on this machine) and rows inside it are not treated as
  evidence. `test_parallel.c` asserts forked and serial results are **bit-identical** — not within a
  tolerance — for every available kernel over eight shapes and three tiles, for both backward
  functions accumulated twice into a non-zero destination, reports how many shape/tile pairs
  actually forked so it cannot pass vacuously, and pins each recorded verdict against the shipped
  threshold so the constant cannot drift silently. GCC, Clang, OpenMP, ASan, ASan+OpenMP,
  size-optimized, `-Wpedantic -std=c11`, `make arm-check`, and both CLI integration checks pass;
  exact resume remains byte-identical.
- [x] **Weight-only quantization, part 1: what it costs in accuracy — COMPLETE.** Simulated
  (quantize-then-dequantize) INT8 and INT4 at per-tensor, per-row, and per-column scales, under an
  explicit policy over which tensors are touched. No storage change and no kernel change, so the
  accuracy question stays isolated from the engineering. `core/quantize.{c,h}` holds the grid,
  `core/model_params.{c,h}` the tensor inventory, `core/model_quantize.{c,h}` the policy and its
  report, and `tools/bench_quant.c` the measurement. Documented in `docs/quantization.md`.

  **The headline is a withdrawal.** At 12 seeds, per-tensor scaling was resolvably worse than
  per-column at 4 bits: +0.0563 ± 0.0222 in relative logit movement, comfortably past the stated
  threshold. Rerun at 60 seeds the same comparison is +0.0559 ± 0.0308 and does **not** clear it.
  The mean moved under 1%; the spread grew 39%. Five times the samples did not sharpen the estimate,
  it revealed that the 12-seed spread was an underestimate — and nothing else changed between the
  runs. The effect is very likely real, but the 12-seed run was not entitled to say so. Both numbers
  are recorded, because a result that stops at the first seed count where its threshold happens to
  be met is the failure mode this project's methodology exists to prevent.

  What survives is the level that does not depend on seeds: per-tensor scaling costs 0.1308 relative
  RMS in weight space at 4 bits against 0.0907 for per-column, and uses 227 of 255 available levels
  at 8 bits against 255 — one outlier stretching the scale and wasting an eighth of the grid, visible
  as a count rather than inferred. **No ΔCE is resolvable for any scheme at 60 seeds**, while INT4
  simultaneously moves logits 14–21% and flips the top-1 prediction on ~1% of windows. Error that
  moves logits need not move the loss; reporting only one level would have called INT4 free or
  called it expensive, which is why all three are printed.

  The axis question came out against expectation. `quantize.h` argues per-output-channel (this
  layout's *columns*, and what PyTorch's dim-0 convention corresponds to) should win, because every
  product feeding an output element would then share one scale. Paired, row minus column is
  −0.00063 ± 0.00105 at 8 bits and −0.01679 ± 0.02298 at 4 — unresolvable at both widths and
  negative both times. Recorded as unsupported at this scale rather than refuted.

  Two methodology failures are recorded because each invalidated a full run before being caught.
  The error statistics were computed after the in-place store, so they compared the output against
  itself and reported exactly zero error for every tensor while quantizing perfectly correctly — no
  test of the *values* could have caught it, and `test_quantize.c` now requires the in-place and
  out-of-place forms to report identical error. Separately, the first corpus was a function of
  absolute position, so the held-out stretch landed on pattern phases training never covered; the
  model memorized to a training loss of 0.02 and scored *worse than uniform* held out. The corpus is
  now a Markov process with a closed-form entropy floor (0.6508 nats against a uniform 3.4657), and
  the tool exits non-zero rather than interpreting deltas from a model that failed it.

  `test_model_params.c` requires the inventory to tile `params` exactly — every float covered once,
  no gap, no overlap — so a tensor added to `model_new()` and forgotten cannot silently escape
  quantization and look like unusual accuracy. GCC, Clang, OpenMP, ASan, size-optimized, and
  `-Wpedantic -std=c11` builds pass, along with both CLI integration checks.

  Scope deliberately not claimed: nothing here transfers to large models. A two-layer 64-wide model
  on a synthetic corpus has no reason to carry the outlier structure that motivates the real
  activation-outlier literature, and only the weight-space level is even scale-independent in
  principle.
- [~] **Weight-only quantization, part 2: storage.** Implementation is ready for source review;
  the requested end-of-bundle test has deliberately not run yet. Bundle version 2 stores tensors
  selected by the part-1 policy as tightly bit-packed symmetric codes plus float32 scales, while
  excluded tensors remain float32. Each tensor record repeats its stable inventory index, kind,
  and shape; the loader validates those, the policy-derived representation and byte bounds before
  dequantizing into the ordinary float model. Header, weight-payload, tokenizer, exact-size, and
  footer validation all apply to the new version, while version 1 remains readable and is still
  what the existing save API writes. `model_bundle_save_quantized()` is opt-in and reports actual
  artifact/payload bytes beside the same policy's quantized tensor/value/scale counts. The bundle
  fixture covers reconstruction against part 1, size accounting, compression, checksum rejection,
  and a forged-but-checksummed tensor shape; completion awaits review and that single test gate.
- [~] **Weight-only quantization, part 3: whether it is actually faster.** Kernel side answered for
  **bf16** (`core/bf16.{c,h}`, `tests/core/test_bf16.c`). INT8 and packed INT4 portable, AVX2, and
  AVX-512 kernels are now implemented in `core/quantized_matmul.{c,h}`, with every scale granularity
  accepted and float32 products/accumulation retained. `test_quantized_matmul.c` compares each
  available dispatch rung against matmul over the fully dequantized values; the standalone
  `bench_quantized_matmul.out` validates before timing fp32/bf16/INT8/INT4 in ABBA order. Per the
  bundled-test policy, neither has run yet, so this item stays partial until the final correctness
  gate and benchmark supply evidence rather than a predicted speedup.

  bf16 was taken first because it is the sharpest possible test of the question, not because it is
  the most compressive. Its widening is a 16-bit left shift into the high half of a float — three
  instructions per vector, no multiply, no per-tile scale, no zero-point — so every other narrow
  format costs strictly more to unpack. **A loss here would have answered part 3 in the negative for
  the whole weight-only family; a win only establishes the ceiling.**

  **The stated prediction was wrong.** This item predicted the dequantization cost eating the
  bandwidth win as "the plausible result at this project's shapes." It does not: bf16 wins on every
  shape measured, in all three sessions, **geometric mean 1.21–1.24x**, worst single reading 1.06x
  and best 1.46x. Portable, AVX2, and AVX-512 rungs all implemented and checked.

  **But the mechanism is not the predicted one, and that matters more than the ratio.** The
  hypothesis was cache capacity — halve the weights, they fit, the win appears as B outgrows L2/L3.
  The measurements do not show that. The most reproducible wins are on the *smallest* matrices
  (64 KB of weights, trivially L2-resident, reading 1.13x and 1.18x to three digits in three
  consecutive sessions), while the largest (16 MB, far past this machine's 8 MB L3) reads only
  1.11–1.20x. There is no monotonic trend against B's size at all. The win is load bandwidth at
  every level of the hierarchy — half the bytes moved from L1 into a register — not weights newly
  fitting in cache. That predicts it transfers to small models rather than only to large ones,
  which is the opposite of what the cache-capacity story implies.

  Accuracy measured separately, as `core/quantize.h` insists, over 60 seeds on a 3-layer 64-wide
  model with the same policy the INT work used (projections and embeddings rounded, biases and
  norms left alone — 98.9% of values):

  | | bf16 | INT4 per-column (recorded above) |
  |---|---:|---:|
  | weight-space relative RMS | 0.001645 | 0.0907 |
  | logit-space relative RMS movement | 0.003903 ± 0.000660 | 0.14–0.21 |
  | top-1 flipped | 0.208% of windows | ~1% of windows |

  So bf16 is roughly 55x more accurate than INT4 in weight space and 40x in logit space, for 2x
  compression against 8x. Both are defensible; they are not competing for the same slot.

  **One methodology failure, caught only because the numbers were absurd.** The first kernel tiled
  (i, j) but ran the full k inside each block, where `run_blocked()` in `core/matmul_x86.c` tiles
  (i, j, l). Measured against it, bf16 read 0.83x at 128×1024×1024 and **0.34x** at 128×2048×2048,
  and the shape of that result was seductive — it looked exactly like "the widening cost dominates
  once B leaves cache," which is the outcome this item predicted. It was "a two-level kernel loses
  to a three-level one." The same version also dropped m<4 shapes to scalar and read 0.13x on a
  decode head. Rewritten to mirror `run_blocked` instruction for instruction, those three shapes
  read 1.36x, 1.23x, and 1.40x. **A benchmark must vary the format and nothing else, or it measures
  the author** — and a wrong result that confirms the stated hypothesis is the one least likely to
  be questioned.

  A second, smaller one: the conversion test first checked infinity and NaN handling with
  `isnan()` and `!= INFINITY`, and reported two failures against a correct conversion. This project
  builds with `-ffast-math`, which implies `-ffinite-math-only`, so the compiler folds both away.
  Those cases are now checked on bit patterns, which is immune and tests the contract more directly.

  Still open before any of this reaches the model: weights are converted once by the caller and the
  narrow kernels are not wired into `core/transformer.c`; bundle v2 deliberately dequantizes on
  load, and the backward pass is untouched. The existing bf16 bound says INT8/INT4 must beat
  1.21–1.24x while unpacking more expensively; that comparison is now executable but not yet run.
- [~] Support memory-mapped weights and measure startup time and resident memory.
  Version-1 float bundles now have a checked, read-only `model_bundle_load_mmap()` path whose model
  views point directly into the mapped payload; ownership, teardown, inference parity, corruption,
  legacy detection, version-2 rejection, and training rejection are covered in
  `test_model_bundle.c`. `bench_bundle_load.out` compares median checked-load startup and peak RSS
  in separate copy/mmap processes with artifact and host provenance. Per the bundled-validation
  workflow, the test and measurement have been authored but not executed yet; record those final
  numbers before changing this item to `[x]`.
- [x] Run the full benchmark nightly on hosted runners and gate on same-run performance invariants
  (`.github/workflows/performance.yml`, `src/tools/perf_check.py`): kernel versus scalar reference,
  shipped kernel versus the fastest candidate beside it, KV-cache benefit, and numerical tolerance.
  Absolute timings are recorded as artifacts but never asserted on, because hosted-runner speed is
  not reproducible across runs. The gate has now paid for itself: run 31459822100 found the shipped
  SIMD default 5.89x/7.28x behind `rowwise` on the medium single-token FFN expansion under
  GCC/Clang. `matmul_select()` now applies that shape rule from width 64 upward while preserving the
  measured tiny-tier exception; the boundary and ordinary SIMD preference are pinned in
  `test_matmul_kernels.c`.
- [ ] Add cross-run regression thresholds on stable benchmark tiers, once enough nightly artifacts
  exist to establish what a normal run-to-run spread looks like on that runner class.

Acceptance gate: each optimized path is numerically checked against the reference path and its
speed/memory improvement is reproducible on identified hardware.

## v0.5 — Modern model and stable library

- [ ] **Measure the seed-variance floor before evaluating any architecture change.** Train the same
  configuration under several seeds and record the spread in held-out cross-entropy. Nothing in the
  list below can be claimed as an improvement until its delta is compared against that number, and
  at this project's model sizes the floor may well be larger than the effects being chased. Finding
  that out first is cheap; finding it out after publishing a ranking is not.

  **Implementation is ready; measurement is deferred to the bundled run.** The earlier corpus
  blocker is removed by the verified Project Gutenberg ebook 100 manifest. The pre-registered
  `experiments/seed-floor-small/recipe.env` varies only the seed, requires 8–20 samples, and chooses
  N from a deterministic bootstrap precision criterion instead of fixing it after seeing a result.
  `seed_floor.out` emits the machine-readable sample standard deviation used as the comparison floor,
  and `scripts/research/measure_seed_floor.sh` preserves every seed/loss/model hash plus the recipe,
  corpus, compiler, and source identities. The shared comparison API still returns "unresolvable at
  this N" when an interval clears zero but sits inside that floor. No floor number is claimed until
  the reviewed source is built and the sweep runs in the final validation pass.
- [~] Evaluate tied embeddings, RoPE, RMSNorm, and GELU/SwiGLU one change at a time, each against
  the floor above, reporting "no resolvable difference" where that is the honest answer.

  **Tied-embedding, RoPE, RMSNorm, GELU, and SwiGLU implementations are ready; review and measurement are deferred.**
  `--tie-embeddings` removes the duplicate output matrix and uses the token table transposed in
  full-prefix, cached-decode, and all-position training paths. Both gradient contributions land in
  the one shared parameter slice. Architecture flags persist through exact checkpoints, immutable
  manifests/config, and lossless or quantized bundle version 3; unflagged writers retain exact v1/v2
  behavior. Focused tests cover layout reduction, forward/backward math, parameter-inventory tiling,
  checkpoint resume, CLI parsing, and copy/mmap/quantized persistence. No build, test, or ablation has
  run before review. RoPE is independently enabled by `--rope`, removes additive positions, rotates
  Q/K per head in full and cached attention, inverse-rotates gradients, and rejects odd head widths.
  Its focused source test covers rotation/inversion, pair-norm preservation, no-additive-position
  input, cached/full equivalence, a numerical Q-gradient, and pure-RoPE copy/mmap bundle persistence.
  RMSNorm is independently enabled by `--rmsnorm`; it replaces both LayerNorm sites, removes their
  beta tensors from the flat layout, and supplies cached full-sequence and decode math plus a
  closed-form backward. Its test covers the equation, numerical input gradients, gamma gradients,
  layout reduction, parameter tiling, cached/full equivalence, training accumulation, and pure
  RMSNorm copy/mmap persistence. GELU is independently enabled by `--gelu`, uses the exact erf form
  in full-prefix and cached decoding, and allocates its pre-activation cache only for GELU models.
  Its focused test covers scalar and end-to-end numerical gradients, cached/full equivalence, flat
  layout stability, and pure-GELU copy/mmap persistence. SwiGLU is independently enabled by
  `--swiglu` and implements `SiLU(xW₁+b₁) ⊙ (xW_gate+b_gate)` with a second projection,
  bias, activation cache, cached-decode scratch, and matched gradients for both branches. GELU and
  SwiGLU are mutually exclusive. Its focused test covers scalar SiLU derivatives, the integrated
  gate equation, numerical gradients for both projections, parameter layout/inventory, cached/full
  equivalence, invalid flag combinations, checkpoint continuation, and pure-SwiGLU copy/mmap
  plus quantized persistence. No build, test, or ablation has run before review.
- [~] Add padding and general attention masks before variable-length batching.
  `model_attention_mask_t` now combines a per-position padding mask with an optional general
  row-major edge mask, always intersected with causality. Mask-aware forward and all-position
  training APIs cache the effective mask for backward, select the last real inference position,
  exclude padded loss/gradients, and define empty attention rows as exact-zero context.
  `test_attention_mask.c` covers prefix equivalence, padding-token independence, future-edge
  rejection, finite empty rows, all-padding rejection, and padded gradient equivalence. Execution
  remains deferred to the final bundled validation pass before this changes to `[x]`.
- [ ] Consider grouped-query attention only after quality and decode benchmarks exist.
- [~] Expose opaque model, tokenizer, cache, and generation handles through a stable public C API.
  `src/include/dranzer.h` now defines API version 1 with opaque, reference-counted handles, one
  structured status space, copy/mmap bundle loading, binary-safe tokenizer buffers, full/cached
  inference, and a deterministic greedy generation session. Cache and generation handles retain
  their model/tokenizer dependencies. `test_public_api.c` covers the public ownership and data-path
  contract, including release-order safety. Source review is complete; execution is deferred to the
  final bundled validation before this changes to `[x]`.
- [~] Remove terminal output from library code and return structured errors to callers.
  Metrics are data-only, DEBUG output requires an application-provided sink, invalid CPU/GPU
  environment overrides are queryable configuration statuses, and CUDA failures retain a
  structured kind/operation/driver-code/message record. The CLI consumes and renders configuration
  diagnostics at its own boundary. `check_library_silence.sh` rejects terminal calls across the
  embeddable source set, while `test_library_diagnostics.c` captures both descriptors around invalid
  settings and requires zero bytes. Execution is deferred to the bundled validation pass.
- [~] Produce static/shared library targets and small embedding examples.
  `make public-libs` now creates self-contained `libdranzer.a` and `libdranzer.so` artifacts from
  isolated PIC objects. The shared artifact exposes only the explicit `DRANZER_1.0` symbol map.
  Public-header-only examples cover static full inference and shared incremental generation, while
  `public-api-check` verifies both link modes and the exact exported ABI. Execution is deferred to
  the final bundled validation pass before this changes to `[x]`.

Acceptance gate: architectural changes have quality, correctness, speed, and memory comparisons;
the CLI is a client of the same supported API available to external programs.

## v1.0 — Mature reference implementation

- [~] Publish small reference models, their corpus provenance, and expected evaluation metrics.
  `references/shakespeare-small/` now pins a complete training/evaluation recipe to the verified
  Project Gutenberg ebook 100 manifest rather than laundering the explicitly test-only smoke
  corpus into a quality result. One command downloads and verifies the exact bytes, applies the
  recorded newline-aligned split, performs a clean GCC build (or accepts a reviewed binary), trains,
  re-evaluates the saved bundle, and stages the bundle, tokenizer, resolved configuration, raw logs,
  provenance, hashes, and expected cross-entropy/perplexity. Static and replay verification reject
  payload corruption, path traversal, forged corpus identity, token-count drift, and metric drift;
  the release gate uploads the resulting directory. Execution and the resulting measured metric are
  deliberately deferred to the final bundled validation pass, so this remains partial until then.
- [~] Guarantee and test the documented model-format and public-API compatibility window.
  Bundle versions 1–3 and public API version 1 are now promised through the 1.x release line.
  Loaders report the actual wire version; fixed v1/v2/v3 header fields are pinned in bundle tests.
  The C ABI has explicit enum values, a size-negotiated 64-byte info record, runtime version
  reporting, compile-time layout/signature assertions, and a checked-in shared-symbol baseline.
  Execution remains deferred to the final bundled validation pass.
- [~] Run fuzzing, full leak detection, and deterministic GCC/Clang build checks in release gates.
  The tag/manual `release-gates.yml` promotes deterministic tokenizer/corpus and bundle mutation
  fuzzing to ASan/UBSan gates, removes the ordinary GPU-test leak exemption on driver-free hosted
  runners, and compares two clean builds under each compiler. GCC and Clang must each reproduce
  their own CLI/static/shared binaries and same-seed training artifacts; compiler-independent
  initial weights are compared across them. Execution is deferred to the final bundled pass.
- [~] Maintain semantic versions, a changelog, reproducible release artifacts, and migration notes.
  `VERSION` now defines project SemVer `0.5.0-dev`, mirrored by the public runtime version query and
  checked against the dated changelog. Migration notes separate project/API/bundle versioning and
  cover the pre-0.5 embedding changes. The release workflow normalizes and packages source plus a
  Linux C SDK twice, requires byte-identical archives/checksums, and uploads only after all release
  gates pass. Packaging execution remains deferred to the bundled validation pass.
- [x] Document supported operating systems, CPU features, and optional GPU behavior precisely.
  `docs/platform-support.md` now separates release-supported, feature-supported, experimental, and
  unsupported configurations. It pins the actual Ubuntu/x86-64 CI contract; exact AVX2/AVX-512
  plus OS-state checks; compile-only, unselected AArch64 NEON status; mmap constraints; OpenMP
  dependencies; and the optional device-0 CUDA path's PTX 7.0/`sm_75` minimum and CPU fallback.

Acceptance gate: a new user can reproduce a reference result, embed the library, upgrade within the
compatibility window, and diagnose failures using only released artifacts and documentation.

## Deliberately deferred

Distributed training, production-scale serving, multiple GPU-vendor backends, and very large model
support remain out of scope until the v0.2 workflow is reproducible and the v0.3 runtime has
measurable quality and performance baselines.
