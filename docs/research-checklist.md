# Research-grade checklist

[← Back to README](../README.md)

`docs/design-checklist.md` is the ordered maturity roadmap: what the software must *do*, version by
version. This file is the other axis, and it is not a roadmap. It asks a single question of the
codebase:

> If this project reported a result, what would stop an outside reader from believing it, and what
> would stop them from reproducing it?

Every item below is either a concrete answer to that question that is currently missing, or a defect
already in the tree that can silently corrupt a result. Items are ordered by how much damage their
absence does, not by how hard they are.

## How to use this file

- **P0** — a result produced today could be wrong or unreproducible because of this. Fix before
  running any experiment intended to be reported.
- **P1** — needed before a result is publishable or comparable to anyone else's.
- **P2** — raises the ceiling on what questions the project can answer at all.

The methodology rules in `docs/design-checklist.md` ("How this checklist is championed") apply here
unchanged, in particular: a difference is only a result if it exceeds the measured noise of the thing
being compared, and what a measurement *cannot* resolve is recorded as carefully as what it can.

Mark an item complete only when its acceptance gate is met and the evidence is written down where a
reader will find it.

**Status.** Every R0 defect that was in the tree when this file was written is fixed, and each one's
entry carries the evidence rather than a claim of completion. Two of them were understated here and
the corrected severity is recorded in place: the performance invariant was not merely flaky but was
reporting a ratio wrong by a factor of eight while passing, and the corpus item turned out to block
every quality result the project could report until the corpus provenance is established. One new R0
item was added, for a defect found by the fuzzing work rather than by inspection.

The measured findings all moved to [`docs/results.md`](results.md); the reproducibility contract is
[`docs/reproducibility.md`](reproducibility.md).

---

## R0 — Threats to results already in the tree

These are not hypothetical. Each one was found in the current source and each can produce a number
that looks fine and is not.

- [x] **P0. `-ffast-math` voids the numerics the causal mask depends on.** — resolved by removing
  the dependence, not the flag, because the flag was measured and earns its place.
  `core/transformer.c:47` writes `-INFINITY` into masked attention positions, and
  `src/Makefile:22` builds every translation unit with `-ffast-math`, which implies
  `-ffinite-math-only` — a promise to the compiler that no infinity will ever appear. Clang 18 warns
  about exactly this on every CI run (`-Wnan-infinity-disabled`), as it does for the `isfinite()`
  call in `tests/core/test_evaluation_state.c:48`.

  The masking works today. It works because no optimization has yet chosen to act on the permission
  it has been given, which is not the same as being correct, and the flag has already invalidated two
  results in this project: the bf16 conversion test's `isnan()`/`!= INFINITY` checks were folded away
  and reported two failures against a correct conversion (`docs/design-checklist.md`, part 3), and
  `test_quantize.c` failed in CI for 42 shapes because the same flag let two alias-conditioned code
  paths compute different sums. The correctness of every attention result rests on a flag whose
  contract the code violates.

  Decide it deliberately rather than by inheritance: either mask with a large finite sentinel
  (`-FLT_MAX/2` or a value derived from the score range), or drop `-ffast-math` from the files whose
  numerics depend on IEEE semantics, or drop it globally and measure what it was actually buying.
  That last measurement has never been made — the flag is justified in the Makefile comment as
  helping "weak FPUs", with no number beside it.

  *Acceptance gate:* no translation unit both relies on IEEE special values and is compiled with a
  flag that forbids them; the CI log is free of `-Wnan-infinity-disabled`; and the decision records
  the measured cost of whichever way it went.

  **The measurement that had never been taken.** Two builds from identical sources differing only in
  that flag, ABBA-interleaved, five paired repeats per tier (clang 14, i5-11320H, serial): inference
  at the small tier reads **3.02 ms/token with the flag against 3.76 without, 1.25x**, 5/5 paired
  readings favouring it and the two distributions not overlapping. Training reads 15.23 ms/step
  against 16.40, **1.09x, which is not resolved at N=5** — four of five pairs favour the flag and one
  inverts. Recorded in `src/Makefile` beside the flag and in `docs/results.md` (R6).

  So the flag stays and the code stops depending on IEEE special values. `attention_head_forward()`
  now writes exact `0.0f` into masked positions and normalizes only the `j <= i` prefix of each row,
  which is **bit-identical** to what the `-INFINITY` form produced — verified, not argued: the same
  fixed-seed 3-epoch run produces byte-identical `model.pth` under both versions
  (`8fe5eb266a163550…`). It is also less work, since both O(T²·d) loops now stop at `i`.
  `isfinite()`/`isnan()` at the edges became bit-pattern checks in a shared
  `include/common/fp_bits.h`, replacing two local copies, and `tools/bench_matmul.c`'s poisoned error
  value became `FLT_MAX` — it had been writing `INFINITY` and then comparing it against a finite
  tolerance under a flag that says infinities do not exist.

  **The gate is enforced structurally rather than by the compiler warning.**
  `tools/check_finite_math.sh` (`make finite-math-check`, wired into CI for both compilers) strips
  comments with `-fpreprocessed -dD -E` and greps every file built with the flag for `isnan`,
  `isinf`, `isfinite`, `fpclassify`, `INFINITY`, `NAN`, and `HUGE_VAL`, reading the exempt list out of
  the Makefile so a file that legitimately drops `-ffast-math` is exempted by the rule that drops it.
  130 files clean, one exempt (`core/quantize.c`). The warning could not have been the gate: clang 18
  emits it, clang 14 and gcc do not, so it fires on some supported toolchains and not others. Checked
  for teeth by adding a deliberate `isnan()` and confirming it fails.

- [x] **P0. The gradient check infers the gradient from an optimizer step, and passes at 100%
  relative error.** — rewritten; all three defects fixed together, and the replacement measures its
  own sensitivity.
  `tests/core/test_gradient_check.c:74` recovers the analytical gradient as
  `(original - new_value) / LR` after calling `model_train_step()`. That is only the gradient if the
  optimizer is plain SGD with no weight decay, no clipping, and no schedule — so the test cannot
  distinguish a wrong gradient from a wrong optimizer, and it silently stops testing gradients at all
  the moment the default optimizer path changes.

  The pass criterion at line 80 is `rel_error < 0.05 || diff < 1e-3`. The absolute escape hatch is
  load-bearing: the CI log shows `layer1.W_q[1] analytical=+0.000125 numerical=+0.000000
  rel_err=1.0000 PASS`. A gradient that disagrees completely passes because it is small. The floor
  exists because central differences in `float` at a fixed `EPS` have a cancellation error around
  `sqrt(machine_eps)` relative, which is the real limit being hit.

  Fix all three together: read gradients directly from `model->grads`, evaluate the perturbed losses
  in `double`, and check a *directional* derivative along a random unit vector over a whole tensor
  rather than 19 hand-picked scalars — one dot product against one finite difference, which tests
  every element at once and has no small-gradient corner.

  *Acceptance gate:* the check reads gradients without running the optimizer, has no absolute
  escape hatch, covers every parameter tensor rather than a sample of scalars, and the reported
  agreement is limited by the finite-difference method rather than by `float` precision.

  `tests/core/test_gradient_check.c` now reads `model->grads` after
  `model_accumulate_gradients_all()` — the path training actually uses — and never runs an optimizer.
  It compares a **directional derivative** `g·u` against `(L(p+hu) − L(p−hu))/2h` along a fixed
  random ±1 direction, one comparison per tensor, covering **all 27 tensors** of the fixture via the
  parameter inventory, so a tensor added to `model_new()` cannot escape. No absolute escape hatch;
  a tensor whose analytical and numerical derivatives are both exactly zero fails rather than passes.
  Losses accumulate in `double`, and the loss being differentiated is cross-checked against the loss
  the backward pass reported before anything is perturbed (gap 3.7e-08).

  **The limiting factor turned out to be ReLU kinks, and finding that out was the substance of the
  work.** A first version with a fixed step disagreed by 1e-2 to 8.8e-1 on most tensors while
  per-element checks on the same gradients agreed to 1e-5 — because perturbing a whole tensor at once
  moves some pre-activation across zero, and then the two sides of the difference sit on different
  linear pieces of the function, so the quotient is not an estimate of a derivative at all. The test
  now **counts switched ReLU units** from `cache_ff_hidden` and halves the step until there are none.
  The evidence is a clean split: before the halving loop, every tensor with ≥1 switched unit
  disagreed by ≥1e-2 and every tensor with none agreed to ≤1e-4, across all 27, worst reading `W_o`
  at 8.8e-1 with five switched units. The step is otherwise chosen per tensor by targeting a
  measurable loss change, since a layer-norm gain with a directional derivative of 0.009 needs a step
  a hundred times larger than `W_ff1`'s — and both stages look only at loss values, never at the
  analytical gradient, so neither can tune itself towards agreement.

  **The tolerance is measured, and the test proves it still has power.** Worst-tensor readings across
  six toolchains, zero switched units in all of them: gcc -O3 3.62e-03, gcc -O3 +OpenMP 3.62e-03,
  clang -Os 5.59e-03, clang -O3 8.07e-03, gcc -O2 1.10e-02, clang -O0 1.10e-02. That spread is float
  rounding in the forward pass being reassociated differently by each compiler, and it is the real
  floor: the loss is computed in `double` from logits that are not. The tolerance is 3e-2, 2.7x above
  the worst. Because a tolerance chosen for headroom might be too loose to catch a real error, the
  test then **corrupts a gradient by 5% and requires the identical comparison to reject it** — it
  does, at 4.76e-02, for both a projection and a norm tensor. Sensitivity is therefore measured
  rather than assumed: about 3% or larger is caught.

- [x] **P0. Weight initialization uses `rand()`, so "same seed" is only same-seed on the same libc.**
  — `rand()` is gone from every path that affects model state or generated tokens.
  `core/tensor_ops.c:19` (`xavier_init`) draws from the process-global C library RNG, seeded once at
  `cli/main.c:552`. `rand()` is implementation-defined: glibc, musl, macOS, and Windows all produce
  different sequences from the same seed. `cli/sampling.c:29` has the same dependency, so seeded
  generation is libc-bound too.

  This directly narrows a claim the project makes elsewhere. Exact-resume bit-identity is documented
  as scoped to "the same executable and execution backend"; the actual scope is narrower and
  undocumented — the same executable *and the same C library*. Nobody else can reproduce a run of
  this codebase from `--seed` alone, which is the single most basic reproducibility property a
  research artifact has.

  The project already has the right primitive: `dropout_forward_rng()` uses a model-owned
  `uint64_t rng_state` (`core/tensor_ops.c:191`) that is persisted by checkpoints. Initialization
  and sampling should draw from the same kind of stream, with the algorithm written down and pinned
  by a test against known vectors.

  *Acceptance gate:* no `rand()` remains on any path that affects model state or generated tokens; a
  test pins the first N draws of the project's RNG to fixed constants; and two builds against
  different C libraries produce byte-identical initial weights from the same seed.

  `core/rng.{c,h}` promotes the xorshift64* the dropout path already used into the project's one
  generator, with SplitMix64-derived **named streams** (`INIT`, `DROPOUT`, `SAMPLING`, `TESTING`) off
  a single user seed — so how often dropout draws can no longer shift the initial weights, and
  adjacent seeds give unrelated streams rather than nearly identical ones. `xavier_init()` and
  `dropout_forward()` take an explicit stream (the rand()-based `dropout_forward` overload is
  deleted rather than kept as a fallback: a fallback that silently degrades a reproducibility
  guarantee is worse than a missing argument). `model_new_seeded()` is the primary constructor and
  `model_new()` delegates with a documented default, which let all 23 `srand()` call sites in the
  suite become explicit per-test seeds instead of a hidden global. Sampling takes a stream through
  `generation_options_t`, and a NULL stream decodes greedily rather than reaching for a global.

  `tests/core/test_rng.c` pins the initial state and first six draws of each named stream against
  literals, plus stream disjointness, adjacent-seed independence, unit draws as **bit patterns**,
  200 000 range checks, and the zero-state fixed point.

  **The libc row is checked as a property, which is stronger than the build comparison the gate
  asked for.** No second C library was installable on the development machine, so instead
  `tests/integration/test_libc_independence.sh` replaces `rand`, `srand`, `random`, and `rand_r` with
  garbage-returning implementations, `LD_PRELOAD`s them ahead of libc, and requires the weight
  fingerprint and the generated text to be unchanged — across three architectures and seeds, and
  through a train-then-`generate` pair with sampling genuinely active. It verifies the preload takes
  effect first, so it cannot pass vacuously, and it was **checked for teeth** by reintroducing a
  `rand()` term into `xavier_init` and confirming all three fingerprints move. That covers the whole
  linked program rather than what a grep can see.

  `core/fingerprint.{c,h}` (extracted from the private copy in `core/bundle.c`, same constants and
  byte order) and `tools/fingerprint_model.c` provide the one-number primitive that check and
  `docs/reproducibility.md` are built on.

- [x] **P0. The training corpus is not part of the artifact.** — manifests, a verifier, and a rule
  that refuses; the provenance gap is now recorded rather than latent.
  `.gitignore:21` excludes `/tests/`, which is where the 178 MB `AllCombined.txt` corpus and its
  chunks live. The tracked corpus is `test.txt`, at 135 bytes. Every quality number this project
  could produce would be measured on data no reader can obtain, with no recorded provenance, no
  license, no train/validation split definition, and no content hash.

  Model bundles already record a "corpus fingerprint" (T2/T5), which pins *identity* but not
  *availability* — a fingerprint proves two runs used the same bytes; it does not let a third party
  get them.

  *Acceptance gate:* every corpus used in a reported result has a manifest under version control
  giving its source, license, byte count, SHA-256, and the exact split boundaries, plus a script that
  reconstructs it from the source; and no reported result uses a corpus without one.

  `data/corpora/` holds the format (`README.md`) and both corpora currently in play, and
  `scripts/corpus.sh` does `verify`, `split`, `create`, and `list`. Splits are **byte offsets, not
  fractions**, each with its own hash: recomputing a fraction on a corpus that grew by one byte would
  silently change which text was held out and every earlier held-out number would stop being
  comparable with nothing in the artifacts to show it.

  For the 178 MB corpus the identity fields are exact and verified — 178 723 194 bytes, 2 052 699
  lines, `be17913eb060032d…`, split at byte 169 787 047 with both sides hashed — and the 18
  `tests/chunk_a*` files were confirmed to concatenate to exactly those bytes. **Its provenance is
  `unverified` and stays that way**: nobody recorded where it came from. The manifest records what
  can be read off the bytes (English encyclopedia prose, consistent with Simple English Wikipedia)
  explicitly as an inference rather than a source, so it cannot be mistaken for one.

  The rule is enforced, not just stated: `scripts/corpus.sh verify --require-verified` **exits
  non-zero** on an unverified manifest, which is what a repro script for a reported result must use.
  So `docs/results.md` currently contains no quality number at all, and says why.

- [x] **P1. The performance invariant test is flaky and can fail a clean tree.** — and it was worse
  than flaky: it was reporting numbers that were not measurements.
  `tests/perf/test_perf_invariants.c:195` requires the AVX-512 `matmul_backward_weight` to beat the
  portable path by 1.20x. Observed on one machine across more than a dozen runs of an unmodified
  tree: every reading but one fell between 1.74x and 3.69x, and one collapsed to 0.67x. CI hit the
  same collapse at 0.74x on a different machine and a different compiler. A floor of
  1.20x sits inside that spread, so the check fails a correct tree at some rate and blocks merges for
  reasons unrelated to the change under review.

  This is a research-integrity item and not merely an annoyance: a suite that cries wolf teaches its
  maintainer to re-run until green, which is the same habit that turns a noisy experiment into a
  published effect.

  *Acceptance gate:* the invariant's own run-to-run spread is measured and recorded (as
  `tools/bench_parallel.c` already does for its noise floor), the threshold is set outside it, and
  the test reports the spread it saw so a future collapse is visible as a distribution change rather
  than a pass/fail flip.

  **The severity was understated in this item.** The AVX-512 `matmul_backward_input` comparison read
  **2.19x on one run of an untouched tree and 18.66x and 14.98x on the next two**; the true ratio on
  that machine is about 17x, confirmed by phase-separated best-of-10 timings of each side. A single
  reading was wrong by a factor of eight on the quantity it claimed to measure, and it **passed**,
  because the floor was 1.20x. The number was never evidence — a threshold anywhere near the real
  value would have flipped at random.

  `include/tools/timing_spread.h` now measures a speedup as a distribution: each replicate times both
  sides **ABBA** and keeps the faster of each side's two readings, the assertion is on the **median**
  of nine replicates, and the min and max are printed on every line so a change of shape is visible
  in the log. The ISA-capped comparisons set the cap inside each timed function so all four calls in
  a replicate run under the ISA they are meant to measure — timing the two arms in separate phases,
  as the old test did, lets drift between phases land entirely on one arm, which is exactly how 2.19x
  happened.

  Six consecutive runs on an idle machine, recorded in the test file: 2.02–2.21, 1.56–1.79,
  35.39–37.97, 3.76–3.82, 16.51–17.63, 2.05–2.15 — every median within 10% of its neighbours, while
  the worst single replicate inside those runs fell to 0.96x for a comparison whose median never left
  1.56–1.79. Every threshold now sits far outside the spread of the statistic it gates.

  A later hosted runner added a different kind of evidence: AMD EPYC 9V74 measured the AVX-512
  backward-weight median at a stable **1.18x [1.17x, 1.19x]**. The distribution was tight, so this
  was not the old timing defect; the original 1.20x floor simply did not transfer to that hardware.
  That one floor is now 1.10x, still separated from a collapsed dispatch near 1.0, while the forward
  and backward-input floors remain 1.20x. A ratio test can remove scheduler noise and still need its
  threshold falsified by a new microarchitecture — both layers matter.

  The next push supplied matching GCC evidence for the total-cost guard on the same hosted class:
  `matmul_backward_weight / matrix_multiply` was a stable **3.06x [2.93x, 3.10x]**, just over the
  old 3.00x ceiling. Its ceiling is now 3.50x; the known-bad loop order remains unambiguously caught
  at 24.7x, and the backward-input ceiling remains 3.00x.

  The same defect was found and fixed in `tests/gpu/test_gpu_latency_invariants.c`, which had a 1.00x
  floor against readings of 0.83x once and 1.01–1.42x on the eight runs after — it failed during this
  work, which is how it was noticed. Its median now reads 1.19–1.29x across five runs. The
  weight-cache assertion, the next-narrowest margin there, got the same treatment.

- [x] **P1. Weight decay is applied to every parameter, including LayerNorm gains and biases.** —
  now a policy over the tensor inventory, with both halves pinned.
  `core/optimizer.c:32` decays `param[i]` for the whole flat parameter vector. Standard practice —
  and what any baseline being compared against will do — excludes biases and normalization
  parameters, because decaying a LayerNorm gain toward zero is a different model, not a
  regularization strength.

  This matters more than its size suggests: it is exactly the kind of silent deviation that makes a
  "we match the reference architecture" claim false, and the parameter inventory needed to fix it
  already exists in `core/model_params.{c,h}` (built for quantization policy).

  *Acceptance gate:* the decay policy is expressed over the tensor inventory rather than the flat
  vector, its default matches the convention it is being compared against, and a test pins which
  tensors are excluded.

  `core/weight_decay.{c,h}` states the rule per `param_kind_t` — projections and embeddings decayed,
  biases and normalization parameters exempt, which is what PyTorch's transformer examples, nanoGPT,
  and the AdamW paper's setups do. The decay left `adam_update()` entirely: it is decoupled, so it
  does not belong inside the moment update, and it is selective, so it cannot be expressed over a
  flat range of floats at all. `model_optimizer_step()` applies it in its own pass first, which keeps
  each decayed tensor's update identical to what the fused version produced.

  `tests/core/test_weight_decay.c` pins the table for every kind in the enum (and fails if a kind is
  added without being pinned), then runs one step with **zero gradients** and `weight_decay=0.5` so
  that any movement is attributable: all 27 tensors are checked, every exempt tensor must not move by
  a single bit, every decayed tensor must move at every nonzero element, and one projection's value
  is checked against `p*(1-lr*wd)` so a decay applied twice or routed through the adaptive step would
  fail even though it would still "move everything".

- [ ] **P1. The tokenizer silently drops NUL bytes, so a corpus hash need not describe what was
  trained on.**
  Found by `tests/core/test_fuzz_tokenizer.c` while building it, not by inspection.
  `libs/src/byte_pair_encoding.c` represents every token as a NUL-terminated C string —
  `snprintf(token, MAX_PAIR_LEN, "%c", input[i])` to build one, `strlen()` to measure it — so an
  input byte of `0x00` becomes an empty token and disappears. A sweep over all 256 byte values
  establishes the boundary exactly: **that one value is lost and the other 255 round-trip byte for
  byte**.

  This is filed under threats to results rather than under hygiene because of what it does to the
  corpus manifests above. A manifest hashes the bytes on disk; if the tokenizer deletes some of them,
  the model was trained on different bytes than the hash identifies, and the chain from a published
  hash to a published number is broken without anything looking wrong. The corpus currently in the
  tree contains no NULs, so nothing measured so far is affected — which is exactly why this should be
  fixed while that is still true.

  The fix is to represent a token as a length plus bytes rather than as a C string. That touches the
  merge loop, the hashmap keys, and the portable serialization format, so it needs a format version
  bump and its own commit.

  *Acceptance gate:* every one of the 256 byte values survives an encode/decode round trip, the
  allowance in `test_fuzz_tokenizer.c` is removed rather than widened, and the serialized tokenizer
  format carries a version that distinguishes the two representations.

---

## R1 — Reproducibility of a single run

- [x] **P0. Define and test the reproducibility contract explicitly.** — `docs/reproducibility.md`,
  one row per axis, every cell measured.
  State, in one place, what is guaranteed to reproduce and under what conditions: same seed + same
  corpus + same config ⇒ identical weights, on (a) the same binary, (b) a rebuild with the same
  compiler and flags, (c) a different compiler, (d) a different CPU ISA, (e) a different libc, (f)
  serial vs OpenMP, (g) CPU vs GPU backend. Several of these are already known — OpenMP forking is
  bit-identical by design, and `DRANZER_CPU_ISA` must be pinned across machines — and several are
  currently unknown.

  *Acceptance gate:* a table in the docs with one row per axis and one of {bit-identical, identical
  within stated tolerance, not guaranteed} in each cell, every entry backed by a test or a recorded
  measurement rather than by reasoning.

  `docs/reproducibility.md`, generated from `scripts/repro/reproducibility_matrix.sh` (seven clean
  rebuilds). Two artifacts reported separately, because a seed names two things with different
  guarantees: **initial weights** are bit-identical on every axis measured — rebuild, gcc vs clang,
  `-Os`, OpenMP, `DRANZER_CPU_ISA` cap, and GPU backend, all `D0190BD0DA47CA15` — and **trained
  weights** are bit-identical for (a) the same binary twice, (b) a clean rebuild, (f) serial vs
  OpenMP, and (g) CPU vs GPU on the machine tested, and **not guaranteed** across compilers, `-Os`,
  or ISA caps, where float reassociation legitimately moves the last bits.

  Two rows are worth calling out. **Serial vs OpenMP is bit-identical end to end** — gcc and gcc with
  two threads produce the same trained hash through a full run with dropout and shuffling active,
  which confirms the design claim that no reduction crosses a thread. **CPU vs GPU is bit-identical**
  on this machine and driver, verified with `gpu_matmul_available()` returning true and the forward
  path dispatching unconditionally, but explicitly not guaranteed across GPUs since the PTX kernel's
  reduction order depends on the block decomposition.

  Row (a) is `tests/integration/test_determinism.sh`, new: the same binary, same seed, twice from
  scratch, with dropout at 0.2 and `--shuffle` on so the run actually consumes randomness. Six
  artifacts compared byte for byte — weights, tokenizer sidecar, every intermediate checkpoint, and
  the logged loss trajectory. Exact resume was tested; two independent runs never had been, and that
  is the property a reader checks first.

- [ ] **P1. One command reproduces one reported result.**
  `scripts/repro/<result-name>.sh` that takes a corpus manifest and a seed list, runs the
  experiment, and writes the exact table or figure that appears in the docs. A reader's first attempt
  to reproduce should not require them to reconstruct a command line from prose.

  *Acceptance gate:* every table in `docs/` that carries numbers names the script that regenerates
  it, and running that script on a clean checkout reproduces the numbers within their stated
  uncertainty.

  **In progress.** `scripts/repro/benchmark_all.sh` now creates one Markdown artifact containing
  the current whole-model throughput matrix: clean Clang and GCC builds, OpenMP compiled out, one
  thread and all visible cores, CPU and GPU modes, and every model tier. CPU and GPU sections are
  separate rows — the first parser mixed CPU inference with later GPU values and thereby described
  no real run — and every cell is a median with the observed training spread beside it. The script
  records the machine and toolchains, retains every raw report outside the checkout, validates its
  arguments, restores the default build, and has been exercised end to end with
  `--quick --repeats 1` from this checkout.

  This does not close the item: historical result tables for attention, quantization, threading,
  and the `-ffast-math` comparison still name tool commands or prose workflows rather than one
  `scripts/repro/<result>.sh` artifact generator. Leaving the box open makes that remaining scope
  visible instead of treating one successful benchmark wrapper as the acceptance gate.

- [ ] **P1. Record the full environment with every result, not just with benchmarks.**
  The benchmark CSVs already carry build command, compiler, OS, CPU, core count, OpenMP version,
  thread count, and detected SIMD (T9/T11). Quality runs carry a manifest of resolved flags (T4) but
  not the environment. Both need both, in the same schema, so a quality result and a speed result
  from the same session can be joined.

  *Acceptance gate:* one provenance record type, emitted by training, evaluation, and every
  benchmark tool, sufficient to identify the machine, build, and inputs of any recorded number.

- [ ] **P2. Make the run log a machine-readable artifact.**
  Per-step JSONL: step, tokens seen, learning rate, training loss, gradient norm before and after
  clipping, per-tensor update-to-weight ratio, wall time, and held-out loss where evaluated. Plots,
  regressions, and post-hoc questions all become queries against a file instead of a rerun.

  *Acceptance gate:* a training run emits a log from which its loss curve, LR schedule, and gradient
  norms can be reconstructed exactly, without re-running anything.

---

## R2 — The measurement harness

The project's methodology is already unusually disciplined — ABBA interleaving, best-of-N with the
median as a noise indicator, withdrawn claims when a larger sample failed to sharpen an estimate.
What is missing is that this discipline lives in prose and in individual tools, so each new
experiment re-implements it, and the one that forgets is the one that gets published.

- [ ] **P0. Measure the seed-variance floor, and make it a number the harness knows.**
  This is already the first unchecked item of v0.5 in `docs/design-checklist.md`, and it gates
  everything in R4 below. Train one configuration under N seeds; record the spread of held-out
  cross-entropy. Until that number exists, no architecture comparison this project makes means
  anything.

  *Acceptance gate:* a recorded floor per model tier, with N chosen from the observed spread rather
  than picked in advance, and the harness refusing to report a comparison narrower than it without
  labelling it unresolvable.

- [x] **P1. A shared statistics module, used by every experiment.**
  Paired comparisons where the design is paired (same seeds, both arms), bootstrap confidence
  intervals, effect size with its interval rather than a bare mean, and an explicit "unresolvable at
  this N" verdict. `tools/bench_quant.c` already computes paired differences with spreads by hand;
  that logic belongs in one place where it can be tested.

  *Acceptance gate:* no experiment tool computes its own error bars; a test pins the statistics
  against hand-computed values on fixed inputs.

  `src/include/tools/statistics.h` and `tools/statistics.c`: summary statistics with the n−1
  denominator, percentile bootstrap intervals (deterministic, seeded from `core/rng.h`, so a reported
  interval is reproducible from its seed rather than moving every time the analysis is re-run), paired
  comparison returning both a difference and a ratio with intervals, a sign count, and an estimate of
  how many pairs would resolve an unresolved effect.

  **The verdict is part of the return value**, which is the point of the module: `STAT_UNRESOLVED` is
  a first-class outcome, and a comparison is unresolved if its interval contains zero **or if the
  whole interval lies inside the caller's noise floor**. `tests/core/test_statistics.c` pins that
  second case explicitly — B is A plus 0.002 in every pair, so the interval is [0.002, 0.002] and
  excludes zero with total confidence, and against a floor of 0.01 the verdict must still be
  unresolved, with `stat_pairs_needed()` reporting that no N would help. That is the 12-seed→60-seed
  withdrawal made structural instead of remembered. Every other value in the test is worked out by
  hand in a comment, including a check that the standard deviation uses n−1 and not n, and that
  `stat_summarize()` does not reorder its caller's array (which would silently break pairing).

  Bootstrap rather than a t-test because the quantities compared here — throughput ratios, held-out
  cross-entropy across seeds, weight-space error — are not normally distributed and there is no reason
  to pretend otherwise.

  **Partially met:** `tools/bench_quant.c` now links the module but has not yet been rewritten to call
  it in place of its hand-rolled paired differences. The gate says no experiment tool computes its own
  error bars; that conversion is mechanical and is the remaining step.

- [ ] **P1. Pre-register the threshold before running the sweep.**
  The 12-seed → 60-seed quantization withdrawal (`docs/design-checklist.md`, part 1) is the exact
  failure this prevents, and the project caught it honestly after the fact. Make it structural: an
  experiment declares its comparison and its resolution threshold in a config file, the runner
  records that file's hash with the result, and a threshold changed after the fact is visible.

  *Acceptance gate:* every reported comparison names the pre-registered threshold it was judged
  against, and a post-hoc change of threshold is detectable from the artifacts alone.

- [ ] **P2. A results store, not a directory of CSVs.**
  `bench_results_v3.csv` and `matmul_results_v3.csv` have already been versioned twice by schema
  changes. One append-only table keyed by (experiment, config hash, seed, provenance hash) makes
  "has this cell already been measured" answerable, which is what turns a sweep into something that
  can be resumed and extended rather than re-run.

  *Acceptance gate:* a sweep interrupted halfway and restarted measures only the missing cells.

---

## R3 — Baselines and comparability

- [ ] **P0. An external correctness oracle for forward and backward.**
  Everything is currently checked against itself: numerical gradients against analytical ones, SIMD
  kernels against the portable kernel, cached decode against full-prefix decode, all-position loss
  against the mean of single-target passes. These are strong internal checks and they cannot catch a
  shared misconception — an architecture implemented consistently but differently from the thing it
  claims to be.

  Build the same architecture in PyTorch (or NumPy — no autodiff needed for the forward), load
  identical weights through a fixed-format dump, and require agreement to a stated tolerance on
  logits and on every parameter gradient for a fixed input. Once that exists, "DRANZER implements a
  decoder-only transformer" is a tested claim rather than a description.

  *Acceptance gate:* a checked-in reference script, a fixture of weights and inputs, and a test that
  fails if logits or gradients drift beyond tolerance from the reference.

- [ ] **P1. Report a standard metric on a standard corpus.**
  Bits-per-byte on enwik8, or perplexity on WikiText-103, or loss on TinyStories — the specific
  choice matters less than that a reader can put the number beside one they already know. Perplexity
  on a private corpus is uninterpretable outside this repository.

  *Acceptance gate:* at least one reported metric is directly comparable to a published number, with
  the tokenizer, context length, and evaluation protocol stated precisely enough to make the
  comparison fair.

- [ ] **P1. A parameter-matched external baseline.**
  Train a reference implementation at the same parameter count on the same corpus and the same token
  budget, and report both. Without it, a loss curve says nothing about whether this implementation is
  good — only that it descends.

  *Acceptance gate:* one reported result includes a same-budget external baseline, with the
  difference reported against the seed-variance floor from R2.

- [ ] **P2. Tokenizer quality as its own measurement.**
  Compression ratio (bytes per token) against vocabulary size, on held-out text, plus the cost of
  the BPE training pass. Vocabulary size is a hyperparameter that changes both model size and
  effective context, and it is currently unmeasured.

  *Acceptance gate:* a table of vocabulary size against held-out bytes-per-token and against
  held-out bits-per-byte, so the two effects can be separated.

---

## R4 — Studies the project is positioned to run

Each of these is gated on R2's seed-variance floor. Running any of them first produces a ranking
that the floor may erase.

- [ ] **P1. The architecture ablations already listed in v0.5** — tied embeddings, RoPE, RMSNorm,
  GELU/SwiGLU — one change at a time, each against the floor, reporting "no resolvable difference"
  where that is the honest answer. Listed here only to note that R2 and R3 are its prerequisites,
  not its follow-up.

- [ ] **P1. Does overlapping the training window buy held-out quality?**
  `--train-stride` exists precisely so this comparison is possible, and
  `docs/design-checklist.md` records it as unmeasured. The first position of a non-overlapping
  window has one token of context; a smaller stride fixes that at proportional cost. This is a
  cheap, self-contained experiment on infrastructure that already exists.

  *Acceptance gate:* held-out cross-entropy against stride at fixed token budget *and* at fixed wall
  time — the two answers can differ, and only reporting both is honest.

- [ ] **P2. A small scaling ladder.**
  Three or four (parameters × tokens) points with loss recorded against both, enough to say whether
  this implementation behaves like the literature's power laws at small scale or whether something
  in it does not. This is the study that most rewards a from-scratch implementation, because every
  variable is visible.

  *Acceptance gate:* loss against compute for at least four configurations, with seed variance shown
  per point, and an explicit statement of what the fit does and does not support at this scale.

- [ ] **P2. Precision as a research object, not only as an optimization.**
  The project already measures bf16 and INT4/INT8 *weight* error against accuracy with real rigour
  (part 1 and part 3). The unasked question is training: what breaks first when the forward, the
  backward, and the optimizer state each move to a narrower type, one at a time. The infrastructure
  — a quantization grid, an error report, a parameter inventory, a bf16 kernel — is already built.

  *Acceptance gate:* per-component precision ablation with held-out quality and step time for each,
  against the seed floor.

- [ ] **P2. Attention behaviour, given the project's name.**
  Attention entropy per head per layer over training, induction-head emergence on a synthetic copy
  task, and the fraction of heads that collapse to a single position. A from-scratch C
  implementation with every intermediate in an inspectable buffer is a better instrument for this
  than a framework, and `cache_probs` is already retained per layer.

  *Acceptance gate:* a synthetic task with a known solution, a metric that detects when the model has
  found it, and a recorded training-time trajectory of that metric.

---

## R5 — Numerics and performance as reportable objects

- [x] **P1. Attention is the last unvectorized hot path.** — measured, and it is not a minor share:
  it is most of the pass.
  `core/transformer.c:31-63` computes scores, mask, softmax, and the value-weighted sum in scalar
  triple loops, while the projections around them go through tuned AVX-512/AVX2 kernels. At
  sequence length T the scalar part is O(T²·d) and grows faster than everything that was optimized.
  Every kernel result in `docs/design-checklist.md` was measured against a model that spends part of
  its time here.

  *Acceptance gate:* attention's share of forward and backward time is measured and recorded before
  any kernel work begins, so the ceiling on that work is known in advance rather than discovered.

  `src/tools/bench_attention.c` (`make bench-attention`), recorded as R1 in `docs/results.md`. The
  scalar core — scores, mask, softmax, value-weighted sum, none of it vectorized — is **61–98% of
  forward-pass time** across the tiers, and attention is **43–96% of the backward**. At fixed d=64 the
  scalar share goes 61% → 85% → 98% as T goes 64 → 256 → 512, which is O(T²d) against O(Td²) showing
  up exactly where it should.

  **So the ceiling on further matmul tuning is the remaining 2–39% of the forward pass.** The 3.58x
  training improvement already banked (part 2 of the design checklist) is real, but the forward-side
  headroom left for kernel work is small and now known, and this is the number that should decide
  whether the next optimization is another matmul kernel or a vectorized attention.

  No instrumentation was added to the hot path; the decomposition times the whole stack, attention
  summed over layers, and the four projections at attention's own shapes, all through public entry
  points. Shares are formed per replicate and reported as a median with the observed range, because
  the method's precision is about ±10 points where attention approaches the whole pass.

  **A methodology failure, caught only because the number was impossible.** The first version timed
  each quantity in its own phase and reported attention as **141% of the forward pass it is part of** —
  drift between phases making two medians incomparable, the identical defect this checklist's own
  performance-invariant item describes. An impossible reading is a lucky failure; the same error at
  85% would have been published.

- [ ] **P1. Report model FLOPs utilization.**
  Tokens/second is not comparable across machines; MFU is. The FLOP count per token is derivable
  from the architecture the model already records, and peak throughput per machine is a constant
  that can be measured once and stored in the provenance record.

  *Acceptance gate:* every throughput number is accompanied by MFU against a stated peak, with how
  the peak was obtained recorded beside it.

- [ ] **P2. Memory as a first-class measurement.**
  Peak resident set, activation memory against sequence length and batch size, and the largest model
  that fits in a given budget. The benchmark records a memory figure; what is missing is the scaling
  relationship, which is what a reader actually needs.

  *Acceptance gate:* measured activation memory against T and B, compared against the analytic
  prediction from the architecture, with any gap explained.

---

## R6 — What a reader receives

- [x] **P1. A results document separate from the roadmap.** — `docs/results.md`.
  `docs/design-checklist.md` currently carries both the plan and the findings, and its measured
  results — the 3.58x training improvement, the 112x from all-position supervision, the bf16
  mechanism finding, the quantization withdrawal — are buried inside completion evidence for
  numbered goals. Those are the most valuable prose in the repository and they are filed under
  project management.

  *Acceptance gate:* a `docs/results.md` where each entry states the claim, the method, the
  uncertainty, the commit it was measured at, and the script that reproduces it — with the roadmap
  linking to it rather than containing it.

- [ ] **P1. Every claim carries its uncertainty in the same form.**
  The project is already careful about this in prose ("11.7x should not be quoted to two significant
  figures"). Make it uniform: point estimate, interval, N, and the noise floor it is being judged
  against, in that order, everywhere.

  *Acceptance gate:* no number in `docs/` appears without its uncertainty or an explicit statement of
  why it has none (an exact count, a pass/fail).

- [ ] **P2. Figures, generated by a script, regenerated by CI.**
  Loss curves, scaling plots, kernel comparisons. A figure checked in as a binary with no generator
  is a claim that cannot be audited.

  *Acceptance gate:* every figure in the docs is produced by a checked-in script from a checked-in
  results file.

---

## R7 — Hygiene that protects the results

- [x] **P1. UndefinedBehaviorSanitizer in the nightly matrix.** — clean over the whole suite.
  ASan runs nightly; UBSan does not. Signed overflow, misaligned loads, and invalid float-to-int
  conversions are the failure modes most likely to differ between the compilers and ISAs this
  project targets, and most likely to be mistaken for a numerical result.

  *Acceptance gate:* a clean UBSan run over the whole suite in the nightly matrix.

  `make UBSAN=1` (`-fsanitize=undefined -fno-sanitize-recover=undefined`, float-divide-by-zero
  excluded since `-ffast-math` already forbids infinities) and a separate nightly job with
  `UBSAN_OPTIONS=halt_on_error=1`, so the job cannot be green with diagnostics buried in its log. The
  whole suite passes clean. Confirmed the instrumentation is real rather than silently dropped: 107
  `__ubsan_*` symbols in the linked binary, and a deliberate signed-overflow probe aborts under the
  same flags.

- [x] **P1. Fuzz the tokenizer and the corpus reader.** — and it immediately found a data-loss
  defect, which is now pinned to its exact boundary.
  Bundle loading is already fuzzed with a deterministic mutation sweep (T5). The tokenizer and the
  training-data path take untrusted bytes and are not.

  *Acceptance gate:* a fuzz target for each, run in the nightly matrix, with a corpus of found
  crashes checked in as regression fixtures.

  `tests/core/test_fuzz_tokenizer.c`, deterministic rather than clock-seeded — a fuzz run that finds a
  crash once and cannot reproduce it is worse than none. 12 checked-in fixtures (embedded NULs,
  truncated UTF-8 sequences, lone continuation bytes, a surrogate encoded as UTF-8, one byte repeated
  for maximal merge pressure) plus 400 random cases over four byte distributions, since a
  uniform-random buffer has no repeated substrings and exercises none of the merge logic. The corpus
  reader is driven at chunk sizes 1, 3, and 8192 so no chunk boundary can be assumed to fall on a
  token or a line. Runs in the ordinary suite and therefore under both sanitizer nightlies.

  **It found that the tokenizer silently drops `0x00`.** `libs/src/byte_pair_encoding.c` represents
  every token as a NUL-terminated C string — `snprintf(..., "%c", input[i])` to build them, `strlen()`
  to measure them — so a NUL input byte becomes an empty token and vanishes. A sweep over all 256 byte
  values confirmed the boundary: **exactly one value is affected and the other 255 round-trip
  exactly**. That matters beyond tidiness — it breaks the chain between a corpus manifest's hash and
  what was actually trained on.

  It is pinned, not tolerated: the round-trip assertion predicts the loss precisely from the NUL
  count, checks that the surviving bytes are the input with its NULs deleted, and
  `check_every_byte_value()` fails if a second value ever starts vanishing *or* if NULs start
  surviving (in which case it says to remove the allowance and tick the item below). The fix is a
  change to the library's token representation and its serialized format, so it is listed separately
  rather than done here.

- [x] **P2. A determinism test that actually runs twice.**
  Exact resume is tested (T3). What is not tested is the simpler claim underneath it: the same
  binary, same seed, same corpus, run twice from scratch, produces byte-identical artifacts. That is
  the property a reader will check first.

  *Acceptance gate:* CI runs a short training twice and compares the artifacts byte for byte.

  `tests/integration/test_determinism.sh`, in the `make test` target and therefore in every CI job.
  Dropout and `--shuffle` are deliberately on: a run with them off would be trivially reproducible
  and would test almost nothing, since the interesting failure is a stream seeded per process rather
  than per run. Six artifacts compared, including every intermediate checkpoint (a divergence at step
  2 that later training washes out would otherwise go unseen) and the logged losses.

- [~] **P2. Pin the compiler versions CI uses.** — partially: runner images were already pinned, and
  the toolchain version is now recorded in every run. Installing an exact compiler version is not
  done.
  Both current CI failures were compiler- or hardware-dependent, and `ubuntu-24.04` silently moved
  clang from 14 to 18 relative to the development machine. An unpinned toolchain means a red suite
  can appear on a tree nobody touched — and, worse, a numerical result can move for the same reason.

  *Acceptance gate:* CI names its compiler versions, and a version bump is a commit that can be
  bisected to.

  Every workflow already targets `ubuntu-24.04` rather than `ubuntu-latest`, so the image only moves
  in a commit. Added a step that prints the compiler version, the runner image and version, and
  `uname -a` at the top of every CI job, so a run can be compared with an earlier one instead of the
  toolchain being invisible in the log.

  **Left open deliberately:** this records the version rather than fixing it. `ubuntu-24.04` can still
  ship a new clang patch release under an unchanged image tag. Actually pinning means installing a
  specific version (an apt pin or a container image), which changes how every job provisions itself;
  worth doing, but it is a CI-infrastructure change rather than the one-line addition the rest of this
  entry was, and it should be its own commit.
