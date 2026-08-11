# Weight-only quantization

[← Back to README](../README.md)

Quantization replaces float32 weights with values on a coarse integer grid. It is normally sold as
three benefits at once — smaller files, less memory bandwidth, and "negligible" accuracy loss — and
that bundling is the problem. When a speedup and an accuracy change arrive in the same commit,
neither can be attributed, and the accuracy claim becomes unfalsifiable.

Part 1 therefore did only the accuracy question, with no storage or kernel change: weights were
quantized to the grid and mapped straight back to float in place. Part 2 now provides an opt-in
versioned bundle representation for the same grid. It bit-packs codes and stores their float32
scales, then dequantizes while loading, so every runtime kernel still receives the same float
representation. Bandwidth kernels remain a separate goal in
[the checklist](design-checklist.md).

The grid lives in `src/core/quantize.c` behind `src/include/core/quantize.h`; the policy over which
tensors it applies to is `src/core/model_quantize.c`; the tensor inventory both depend on is
`src/core/model_params.c`. The storage path is `src/core/bundle.c`, and its version 2 contract is
documented in [model-bundle.md](model-bundle.md). The measurement tool is
`src/tools/bench_quant.c`.

## The scheme

Symmetric, no zero-point: a quantized value is exactly `scale * q`, with `scale` derived from the
largest magnitude in its group and `q` an integer in `[-qmax, qmax]` for `qmax = 2^(bits-1) - 1`.

Two consequences are worth stating because they are choices, not defaults:

- **The negative end gives up one level.** Two's complement runs to `-2^(bits-1)`, and dropping it
  costs 1 level in 128 at 8 bits but 1 in 8 at 4 bits. It is dropped anyway, so the grid is
  symmetric about zero and the reconstruction carries no sign-dependent bias.
- **A group of one value is exactly lossless**, at any bit width, because the scale *is* that value.
  This is not a happy accident — it is why per-column scales on a one-row bias vector achieve
  literally zero error while storing a 32-bit scale for every 4-bit value, which is worse than
  leaving the tensor alone.

## Which values share a scale, and why the axis is a trap

This project stores a weight matrix as `k x n` and computes `C = A @ W`. **A column of `W` is one
output channel; a row is one input channel.** PyTorch stores `nn.Linear` as `out x in` and quantizes
along dim 0 — per-output-channel, which corresponds to this project's *columns*. A per-row recipe
ported across that boundary silently quantizes the other axis.

There is also a mechanism that predicts columns should win. An output element sums over the
reduction axis, which is the rows. Give every row its own scale and one output element accumulates
products carrying `k` different scales, so their errors share no common factor. Give every column
its own scale and every product feeding an output element carries the same one.

That prediction is stated in `quantize.h` and **the measurement does not support it** — see
[What the measurements found](#what-the-measurements-found). Both axes are implemented so the
question could be asked rather than assumed.

## The policy

`model_quantize_weights()` applies a grid to the tensors a policy selects. Projections are always
included. Embeddings and the bias/norm parameters are flags, both off by default, and both defaults
are inherited conventions rather than measured facts — which is exactly why they are flags:

| Tensor kind | Default | Reasoning |
|---|---|---|
| Projections | quantized | The weights worth compressing, and what every scheme in the literature targets |
| Embeddings | left alone | Large enough to be worth compressing, but read by lookup rather than multiplied, so the scale-axis argument does not apply |
| Biases and norms | left alone | A fraction of a percent of the parameters, sitting directly on a normalized activation |

## Measurement: three levels, three different questions

Quantization error is reported at whichever level flatters the result. `bench_quant.out` reports all
three, because they answer different questions and only the first is portable.

1. **Weight space** — how far the reconstructed tensor is from the original. Exact, cheap, and
   independent of training quality, corpus, and model scale. The only level whose numbers mean
   anything outside this benchmark.
2. **Logit space** — how far outputs move on fixed inputs. Deterministic, and the first level where
   the *structure* of the error matters rather than its size: error that cancels across a reduction
   never arrives here.
3. **Held-out cross-entropy** — what a user would notice. Also where a difference is easiest to
   claim and hardest to justify.

### The corpus has a computable optimum

The benchmark trains on a first-order Markov process: with probability 0.9 the next token is a fixed
successor of the current one, otherwise uniform over a 32-token vocabulary. Training and held-out
windows are disjoint stretches of the same process.

That shape is not incidental. Its cross-entropy floor is computable in closed form (0.6508 nats
against a uniform model's 3.4657), so **"did this model learn anything" is a comparison rather than
an impression** — and the tool refuses to interpret any delta from a model that failed it, returning
non-zero.

That gate exists because this tool shipped without it once. An earlier corpus drew tokens from a
function of absolute position; the held-out stretch landed on pattern phases training had never
covered, the model memorized to a training loss of 0.02, and held-out cross-entropy came out *worse
than uniform*. Every quantization delta measured against it was noise on a model that predicted
nothing, and the table looked entirely reasonable.

### Comparisons are paired

For every seed, one model is trained, evaluated, then quantized and evaluated again. The *difference*
is the sample; the spread of those differences across seeds is the uncertainty.

This matters more than anything else in the methodology. The baseline's own seed-to-seed spread on
this setup is about **0.14 nats**. The accuracy costs being measured are between 0.0001 and 0.01
nats — one to three orders of magnitude smaller. An unpaired comparison would have to beat that 0.14
to say anything at all, and could not. The paired difference removes it, because both measurements
come from the same trained weights.

The same argument applies one level down, to the comparison *between schemes*: every scheme runs on
the same trained models, so granularity comparisons are paired too. Pairing sharpens those
comparisons considerably — but as
[the 12-seed run](#the-12-seed-run-said-something-the-60-seed-run-does-not) shows, sharpening an
estimate is not the same as having enough samples for it.

### "Not resolvable" is reported as a number

Where a delta does not clear twice its own spread, the tool prints how many seeds it *would* take at
the effect size and spread just measured. That turns an unsatisfying "no" into a decision: one more
afternoon of compute, or out of reach at this model size.

Treat that column as guidance and not a plan. It is computed from a noisy mean divided by a noisy
spread, and the estimates moved by more than an order of magnitude between the 12- and 60-seed runs
below.

## What the measurements found

```text
build:    gcc 11.4.0 -I include/ -I ../libs/include/ -O3 -ffast-math -lm
os:       Linux 5.15.167.4-microsoft-standard-WSL2 x86_64
cpu:      11th Gen Intel(R) Core(TM) i5-11320H @ 3.20GHz
corpus:   first-order Markov, vocab 32, P(successor) 0.9, floor 0.6508 nats
model:    emb 64, heads 4, layers 2, seq 4, 20000 steps, 128 held-out windows
seeds:    60
baseline: held-out cross-entropy 0.7031 mean, 0.1203 stddev (floor 0.6508)
```

| Scheme | weight rms.rel | eff. bits | levels | logit rms.rel | top-1 | ΔCE | ΔCE sd | seeds to resolve |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| int8/tensor | 0.00721 | 8.75 | 227 | 0.01101 | 0.999 | 0.00009 | 0.00073 | ~259 |
| int8/row | 0.00503 | 9.12 | 255 | 0.00758 | 1.000 | −0.00009 | 0.00059 | ~183 |
| int8/column | 0.00500 | 9.11 | 255 | 0.00821 | 0.999 | −0.00000 | 0.00059 | ~73000 |
| int8/col+emb | 0.00497 | 8.66 | 255 | 0.00858 | 0.999 | −0.00003 | 0.00064 | ~2600 |
| int8/col+all | 0.00453 | 8.75 | 255 | 0.00858 | 0.999 | −0.00003 | 0.00064 | ~2600 |
| int4/tensor | 0.13078 | 4.88 | 15 | 0.20905 | 0.986 | 0.00630 | 0.02519 | ~64 |
| int4/row | 0.09129 | 5.25 | 15 | 0.13639 | 0.991 | 0.00070 | 0.01148 | ~1100 |
| int4/column | 0.09068 | 5.24 | 15 | 0.15318 | 0.990 | 0.00251 | 0.01698 | ~183 |
| int4/col+emb | 0.09011 | 4.71 | 15 | 0.16050 | 0.990 | 0.00239 | 0.01818 | ~232 |
| int4/col+all | 0.08227 | 4.75 | 15 | 0.16050 | 0.990 | 0.00239 | 0.01818 | ~232 |

Paired granularity comparisons, same seeds, on logit rms.rel:

| Comparison | mean difference | spread | resolvable |
|---|---:|---:|---|
| int8: tensor − column | +0.00281 | 0.00172 | no |
| int8: row − column | −0.00063 | 0.00105 | no |
| int4: tensor − column | +0.05588 | 0.03078 | no |
| int4: row − column | −0.01679 | 0.02298 | no |

### The 12-seed run said something the 60-seed run does not

This table was first produced at 12 seeds. There, `int4: tensor − column` measured **+0.05628 ±
0.02216** and cleared the threshold: per-tensor scaling was resolvably worse than per-column.

At 60 seeds the same comparison is **+0.05588 ± 0.03078** and does not clear it. The mean barely
moved — 0.0563 to 0.0559, under 1% — while the spread grew by 39%. Five times the seeds did not
sharpen the estimate; it revealed that the 12-seed spread was an underestimate.

Nothing was fixed between the runs. The only change was more samples, and the conclusion reversed
from "resolvable" to "not". The effect is very likely real, since the mean is stable across a
fivefold increase in samples, but **the 12-seed run was not entitled to say so**, and neither is any
run that stops at the first seed count where its threshold happens to be met.

The `seeds to resolve` column moved the same way and is worth distrusting for the same reason: at 12
seeds it estimated ~34 seeds for int4/tensor and ~43 for int4/row. At 60 the estimates are ~64 and
~1100. They are computed from a noisy mean and a noisy spread, so they are order-of-magnitude
guidance and not a plan.

### What survives, and what does not

At 60 seeds, **no paired granularity comparison is resolvable in logit space**, and no ΔCE is
resolvable for any scheme. What remains solid is the level that does not depend on seeds at all:

- **Weight-space error separates the schemes cleanly.** Per-tensor scaling costs 0.1308 relative RMS
  at 4 bits against 0.0913 for per-row and 0.0907 for per-column — a 44% penalty, on a quantity
  computed directly from the weights rather than sampled through a model.
- **The `levels` column shows the mechanism as a count, not an inference.** At 8 bits, per-tensor
  scaling uses 227 of its 255 available levels while per-group scaling uses all 255: a single outlier
  has stretched the scale and left an eighth of the grid unreachable. That is the failure mode
  granularity exists to prevent, visible directly.
- **Top-1 agreement degrades measurably at 4 bits and not at 8.** INT4 changes the argmax on about
  1% of held-out windows; INT8 on essentially none.

So the defensible statement is: *per-group scaling is better than per-tensor scaling in weight space
and in grid utilization, by margins that are large and directly computed, and that advantage does
not translate into a resolvable accuracy difference at this model size.*

### Which axis the scale runs along is not measurably important here

This contradicts both the convention and this project's own stated mechanism. Paired, row minus
column is −0.00063 ± 0.00105 at 8 bits and −0.01679 ± 0.02298 at 4 bits: not resolvable at either
width, and *negative* both times, meaning per-input-channel was if anything marginally better than
the per-output-channel scheme the literature prefers. In weight space the two are within 0.7% of
each other.

The honest reading is that the prediction in `quantize.h` is unsupported at this scale, not that it
is wrong. A 64-wide reduction offers little room for the error-cancellation argument to bite. What
can be said is narrow: **at this model size, choosing the "correct" axis bought nothing measurable,
while choosing per-group over per-tensor bought something real in weight space.**

### The accuracy cost is smaller than this setup can see

No ΔCE is resolvable at 60 seeds. INT8's deltas would need hundreds to tens of thousands of seeds,
which is another way of saying they are indistinguishable from zero. INT4's need ~64 to ~1100 by an
estimate that is itself unstable.

Note what does *not* follow from a small ΔCE. INT4 moves the logits by 14-21% relative and changes
the top-1 prediction on about 1% of windows. Both are large, unambiguous, and measured. They simply
do not translate into a cross-entropy change this setup can resolve — cross-entropy depends on the
target token's relative position, not on the logit vector's magnitude. Reporting only ΔCE would have
called INT4 free; reporting only logit divergence would have called it expensive. Neither alone is
the answer, which is the reason all three levels are printed.

### Effective bits, not stated bits

The `eff. bits` column is bits per parameter the policy would actually store, counting float32 scales
and the tensors left at full width. It is reported because the stated width is not the storage cost,
and comparing schemes at equal *stated* width silently credits a finer granularity for accuracy it
bought with space.

The gap is large. "INT4" ranges from 4.71 to 5.25 effective bits depending on granularity and
policy, and "INT8" from 8.66 to 9.12 — never the nominal number. The `col+all` rows are the sharpest
illustration: quantizing biases and norms alongside everything else *raises* effective bits from
4.71 to 4.75 while changing the logits by exactly nothing, because those tensors are one row each,
so per-column scaling gives every value its own 32-bit scale. That is the degenerate corner the
default policy avoids, and the per-tensor breakdown (`--per-tensor`) shows it directly: every bias
and norm tensor reports 0.00000 relative error, 1-2 levels used, and every value clipped.

## Reproducing the measurements

```bash
cd src
make bench-quant CC=gcc
./bench_quant.out --seeds 12
./bench_quant.out --seeds 12 --per-tensor      # add the per-tensor breakdown
./bench_quant.out --seeds 60 --steps 20000     # enough to resolve the INT4 deltas
```

Every row is appended to `quant_results_v1.csv` with the same build/OS/CPU/SIMD/thread provenance
columns every results file in this project carries. The tool exits non-zero if the trained model did
not beat a uniform predictor, so a broken training configuration fails rather than producing a
plausible-looking table.

Unlike this project's timing benchmarks, **these results are deterministic**: the corpus is a fixed
constant, the model seeds are fixed, and quantization is arithmetic. Two runs of the same command
produce byte-identical tables, and there is no noise floor to establish because the only variance is
the seed-to-seed variance being deliberately sampled. Results should therefore reproduce across
machines, which the timing files explicitly do not.

## Limitations

These are load-bearing. The numbers above are a methodology demonstration on a small model, and the
gap between that and a claim about real models is the most important thing on this page.

- **The model is two layers of width 64 on a synthetic Markov corpus.** Quantization difficulty is
  driven by outlier structure in trained weights, and there is no reason to expect a model this
  small on a corpus this simple to have the outlier structure of a real language model — which is
  precisely the phenomenon that motivated LLM.int8() and SmoothQuant at scale. **Nothing here
  transfers to large models by default**, and the level-1 weight-space numbers are the only ones
  that are even scale-independent in principle.
- **Only weight-only quantization is measured.** Activations stay float32 throughout. Activation
  outliers are the harder half of the real problem and are not addressed.
- **The grid is symmetric and per-group only.** No affine zero-point, no group-of-N-within-a-row
  blocking, no error-compensating schemes (GPTQ/AWQ-style). Those are the interesting comparisons
  and none of them were run.
- **Nothing in the accuracy table is faster.** By construction: those weights are float32 again the
  instant they are quantized. Runtime INT8/INT4 kernels now exist separately, but their benchmark is
  deliberately outside this accuracy experiment and has not yet supplied a result.
- **Even 60 seeds is a coarse spread estimate**, and the reported figure is a sample standard
  deviation rather than a confidence interval. "Resolvable" is a stated rule of thumb
  (|mean| > 2x spread), not a hypothesis test, with no correction for the ten schemes and four
  comparisons being examined at once — which, given that a 12-seed run already produced a
  conclusion the 60-seed run withdrew, is a reason to treat every "yes" in this document as
  provisional.
- **One corpus, one architecture, one optimizer.** No claim is made that the granularity ordering
  survives any of those changing.
