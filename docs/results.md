# Results

[← Back to README](../README.md)

Every measured finding this project has, in one place. Each entry states the
claim, the method, the uncertainty, and what reproduces it.

This document exists because those findings were previously filed inside
completion evidence for numbered goals in
[`docs/design-checklist.md`](design-checklist.md). That made the most valuable
prose in the repository read as project management, and it made a result
impossible to cite: there was no stable place to point at. The roadmap now links
here rather than containing this.

## How to read an entry

Point estimate, then interval, then N, then the noise floor it was judged
against — in that order, every time. A number with no interval says why it has
none (an exact count, a pass/fail, a single deterministic measurement).

Two rules the project holds itself to, and which several entries below apply
against their own author:

- **A difference is only a result if it exceeds the measured noise of the thing
  being compared.** "No resolvable difference" is a finding and is reported as
  one.
- **What a measurement cannot resolve is recorded as carefully as what it can.**

---

## R1. Attention is where the time goes, not the matmuls

**Claim.** The scalar attention core — scores, causal mask, softmax, and the
value-weighted sum, none of which has a SIMD kernel — is **61–98% of forward-pass
time** across the project's model tiers, rising with context length at fixed
width. Attention's share of the backward pass is **43–96%**.

**Why it matters.** Every kernel result below it in this document was measured on
a model that spends most of its time in code those kernels never touch. That sets
a ceiling: further matmul tuning can address at most the remaining 2–39% of the
forward pass. Vectorizing attention addresses the rest.

**Method.** `src/tools/bench_attention.c`. Three quantities timed on the real
model through its public entry points — the whole layer stack
(`model_forward_hidden`), attention summed over layers
(`multihead_attention_forward`), and the four projections attention performs at
the same shapes (`model_dispatch_matmul`) — with the scalar core taken as the
difference. All measurements are interleaved within each replicate and the shares
are formed per replicate before being summarized.

**Uncertainty.** Medians of 7 replicates, with the observed range reported beside
each share. The decomposition's precision is about ±10 percentage points at shapes
where attention approaches the whole pass, because the isolated and in-context
measurements see different cache states; ranges touching 100% mean "essentially
all of it", not "more than the whole". Rows whose absolute forward time is under
1 ms sit near timer resolution and their ranges are correspondingly wide.

| config | d | T | forward | attention share | scalar core | backward | attention share |
|---|--:|--:|--:|---|--:|--:|---|
| tiny | 16 | 32 | 0.08 ms | 113% [69–117] | 108% | 0.49 ms | 43% [43–46] |
| tiny | 16 | 128 | 1.14 ms | 92% [75–105] | 91% | 5.39 ms | 80% [74–103] |
| tiny | 16 | 256 | 4.43 ms | 98% [80–102] | 97% | 19.62 ms | 88% [60–125] |
| small | 64 | 64 | 2.08 ms | 72% [62–86] | 61% | 11.03 ms | 60% [58–62] |
| small | 64 | 256 | 30.14 ms | 87% [80–104] | 85% | 146.38 ms | 80% [79–86] |
| small | 64 | 512 | 101.00 ms | 99% [89–118] | 98% | 632.54 ms | 96% [93–103] |
| medium | 128 | 256 | 97.97 ms | 78% [68–97] | 72% | 475.75 ms | 84% [76–101] |
| medium | 128 | 512 | 330.51 ms | 91% [78–113] | 88% | 1787.67 ms | 93% [84–95] |

**The trend is the finding, not any single cell.** At fixed d = 64 the scalar
share goes 61% → 85% → 98% as T goes 64 → 256 → 512, which is the O(T²d) against
O(Td²) scaling showing up exactly where it should. The absolute numbers are one
machine; the growth is structural.

**A methodology failure caught by an impossible number.** The first version of
this tool timed each quantity in its own phase — seven forwards, then seven
attentions — and reported attention as **141% of the forward pass it is part of**.
The cause was drift between phases making two medians incomparable; the fix was
interleaving. An impossible reading is a lucky failure. The same error at 85%
would have been believed, and this is the second time in this project that
phase-separated timing produced a wrong number (see R7).

**Reproduce.** `make -C src bench-attention && ./src/bench_attention.out`

---

## R2. All-position supervision: 112x, from deleting a `-1`

**Claim.** Supervising every position in the context window instead of only the
last improved medium-tier training throughput by **112x**.

**Method and uncertainty.** The causal mask already guarantees that position i's
final hidden state depends only on tokens 0..i, so the same forward pass yields
seq_len training signals rather than one. The change turns three m=1 matmuls into
m=seq_len; nothing about the model changes. Correctness is verified by
`tests/core/test_all_position_loss.c`, which requires the all-position loss to
equal the mean of single-target passes and separately tests the causality claim it
rests on.

The factor is a throughput ratio on one machine and tier; it is not an interval,
and it should not be read as one. See [`design-checklist.md`](design-checklist.md)
part 2 for the full record.

---

## R3. Kernel work: 3.58x on medium-tier training

**Claim.** The combined matmul work — a fixed loop order in
`matmul_backward_weight`, register-resident AVX-512/AVX2 kernels, the backward
AVX-512 paths, and GPU backward dispatch — multiplied out to **3.58x** on
medium-tier training (2.75x × 1.30x).

**Uncertainty.** Geometric means across 36 (compiler, tier, shape) combinations:
1.07x worst, 1.83x geometric mean, 3.02x best for the forward kernel against
`tiled_mr4`, and 8.78x geometric mean against the scalar reference. Both GCC and
Clang independently.

**Read this against R1.** These kernels operate on the 2–39% of forward time that
is not the scalar attention core. The 3.58x is a real training-throughput
improvement — it includes the backward pass, where the projections carry more of
the total — but the forward-side headroom left for further matmul tuning is small
and now measured.

**Reproduce.** `make -C src bench && ./src/bench.out --matmul-only --sweep`

---

## R4. bf16 weights are faster, and not for the predicted reason

**Claim.** bf16 weight storage with widening in the kernel is faster on every
shape measured: **geometric mean 1.21–1.24x**, worst single reading 1.06x, best
1.46x, reproduced across three sessions on portable, AVX2, and AVX-512 rungs.

**The mechanism is not the predicted one, and that matters more than the ratio.**
The hypothesis was cache capacity — halve the weights, they fit, the win appears
as B outgrows L2/L3. The measurements do not show that. The most reproducible wins
are on the *smallest* matrices (64 KB of weights, trivially L2-resident, reading
1.13x and 1.18x to three digits in three consecutive sessions), while the largest
(16 MB, past this machine's 8 MB L3) read only 1.11–1.20x. There is no monotonic
trend against B's size. The win is load bandwidth at every level of the hierarchy,
not weights newly fitting in cache — which predicts it transfers to small models
rather than only to large ones, the opposite of what the cache story implies.

**Accuracy, measured separately over 60 seeds** on a 3-layer 64-wide model:

| | bf16 | INT4 per-column |
|---|---:|---:|
| weight-space relative RMS | 0.001645 | 0.0907 |
| logit-space relative RMS movement | 0.003903 ± 0.000660 | 0.14–0.21 |
| top-1 flipped | 0.208% of windows | ~1% of windows |

**A methodology failure worth keeping.** The first bf16 kernel tiled (i, j) but
ran the full k inside each block, while the reference tiles (i, j, l). Measured
against it, bf16 read 0.83x and **0.34x** on the largest shapes — and the shape of
that result was seductive, because it looked exactly like "the widening cost
dominates once B leaves cache", which was the stated hypothesis. It was "a
two-level kernel loses to a three-level one". Rewritten to mirror the reference
instruction for instruction, the same three shapes read 1.36x, 1.23x, and 1.40x.
**A benchmark must vary the format and nothing else, or it measures the author** —
and a wrong result that confirms the stated hypothesis is the one least likely to
be questioned.

**Scope not claimed.** Forward-only; the backward pass is untouched. Nothing here
transfers to large models: a two-layer 64-wide model on a synthetic corpus has no
reason to carry the outlier structure the activation-outlier literature is about.

---

## R5. A withdrawal: per-tensor INT4 scaling

**Claim.** **None.** At 12 seeds, per-tensor quantization scaling looked
resolvably worse than per-column. At 60 seeds the effect did not sharpen, and the
claim was withdrawn.

This is the most important entry in this document, because it is the one that
shows the machinery working. An effect that appears at one sample size and fails
to survive a larger one is the standard way a wrong result enters the literature.
The response — running more seeds *before* publishing, and withdrawing rather than
reporting the smaller sample — is now structural rather than a matter of
discipline: `src/include/tools/statistics.h` returns "unresolvable at this N" as a
first-class verdict, and `tests/core/test_statistics.c` pins the case where an
interval excludes zero but sits inside the noise floor.

---

## R6. What `-ffast-math` is worth

**Claim.** `-ffast-math` buys **1.25x on inference** at the small tier (5/5 paired
readings favour it; the two distributions do not overlap) and about **1.09x on
training**, which is *not resolved* at N=5 — four of five paired readings favour
it and one inverts.

**Why the number exists.** The flag had been carried for years on the claim that
it "helps on weak FPUs", with no measurement beside it, while the causal mask
depended on `-INFINITY` — a value the same flag promises the compiler will never
appear. The measurement decided the question: the flag stays and the dependence
on IEEE special values goes.

**Method.** Two builds from identical sources differing only in that flag,
ABBA-interleaved, five paired repeats per tier, clang 14 on an i5-11320H, serial.

**A finding inside the finding.** The win is concentrated on inference, which is
where R1's scalar attention loops dominate. The flag is partly compensating for
the one hot path with no SIMD kernel — so vectorizing attention should shrink what
`-ffast-math` is worth.

---

## R7. The old timing invariants were not measuring anything

**Claim.** The AVX-512 `matmul_backward_input` performance invariant read
**2.19x on one run of an untouched tree and 18.66x and 14.98x on the next two**.
The true ratio on that machine is about 17x, confirmed by phase-separated
best-of-10 timings of each side. A single reading was wrong by a factor of eight
on the quantity it claimed to measure — and it passed, because the floor was
1.20x.

**The fix, and its evidence.** Every ratio is now a median over ABBA-interleaved
replicates with the spread printed beside it
(`src/include/tools/timing_spread.h`). Six consecutive runs on the same idle
machine:

| invariant | median across runs | threshold |
|---|---|---|
| `matmul_backward_input` / `matrix_multiply` | 2.02–2.21 | ≤ 3.00 |
| `matmul_backward_weight` / `matrix_multiply` | 1.56–1.79 | ≤ 3.50 |
| `matrix_multiply` vs scalar reference | 35.39–37.97 | ≥ 1.50 |
| AVX-512 forward matmul | 3.76–3.82 | ≥ 1.20 |
| AVX-512 `matmul_backward_input` | 16.51–17.63 | ≥ 1.20 |
| AVX-512 `matmul_backward_weight` | 2.05–2.15 | ≥ 1.10 |

Every median within 10% of its neighbours, while the worst single replicate inside
those runs fell to 0.96x for a comparison whose median never left 1.56–1.79. The
GPU kernel invariant behaved the same way: a 1.00x floor against readings of 0.83x
once and 1.01–1.42x on the eight runs after, now a median of 1.19–1.29x.

The backward-weight threshold was later falsified, rather than the measurement:
an AMD EPYC 9V74 runner produced a tight 1.18x [1.17x, 1.19x] distribution. Its
floor is therefore 1.10x; the other AVX-512 paths retain 1.20x. The revised guard
still separates a working dispatch from the near-1.0 collapse it exists to catch.
The same hosted GCC job measured backward-weight's total cost at 3.06x
[2.93x, 3.10x] versus forward, just beyond its old 3.00x ceiling. That ceiling is
now 3.50x, still far below the known-bad loop order's 24.7x result.

**This is a research-integrity item, not an annoyance.** A suite that cries wolf
teaches its maintainer to re-run until green, which is the same habit that turns a
noisy experiment into a published effect.

---

## R8. Reproducibility, measured per axis

See [`docs/reproducibility.md`](reproducibility.md) for the full table. The
summary: **initial weights are bit-identical across every axis measured** —
compiler, optimization level, OpenMP, ISA cap, and GPU backend — and trained
weights are bit-identical for a rebuild, for serial versus OpenMP, and for CPU
versus GPU on the machine tested, but not across compilers or ISA caps, where
float reassociation legitimately changes the last bits.

**Reproduce.** `scripts/repro/reproducibility_matrix.sh`

---

## Open, and why

Recorded here so a reader does not have to infer it from silence.

- **No quality result appears in this document.** The corpus this project trains
  on has unverified provenance
  ([`data/corpora/simple-wikipedia-178mb.manifest`](../data/corpora/simple-wikipedia-178mb.manifest)),
  so a held-out cross-entropy measured on it could not be reproduced by anyone
  else. Establishing the corpus is a prerequisite for every entry that R4's
  research checklist wants, including the seed-variance floor.
- **No seed-variance floor yet**, so no architecture comparison can be reported.
  The floor is cheap to measure and gated on the corpus above.
- **No external baseline and no standard-corpus metric**, so nothing here is
  comparable to a published number.
- **Every number above is from one machine** (i5-11320H, WSL2). Ratios are
  designed to survive that; absolute times do not.

## One deliberate behaviour change, and what it does not invalidate

Weight decay no longer applies to biases or normalization parameters
(`src/core/weight_decay.h`). That changes the trained weights of any run with
`--weight-decay` above zero, which is the CLI default of 0.01 — so a run repeated
after this change will not reproduce a weight hash from before it.

**No result in this document is affected.** The throughput and kernel results
(R1, R3, R6, R7) do not depend on the decay policy, and the quantization accuracy
work behind R4 and R5 set `weight_decay = 0.0f` explicitly
(`src/tools/bench_quant.c`) precisely so that the optimizer could not confound it.
Stated here rather than left to be discovered, because "the same seed no longer
gives the same weights" is exactly the kind of change that should never be
silent.
