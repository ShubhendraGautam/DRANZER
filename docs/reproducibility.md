# The reproducibility contract

[← Back to README](../README.md)

What is guaranteed to reproduce, under what conditions, and how each cell was
established. Every entry below is backed by a test or a recorded measurement.
None of it is reasoning about what the code ought to do — that distinction is the
point of the document, because the previous version of this claim ("bit-for-bit
on the same executable and execution backend", in `docs/design-checklist.md`) was
both true and narrower than it sounded, and nothing said which parts had been
checked.

## The two artifacts

A seed names two different things and they have different guarantees, so they are
reported separately throughout.

**Initial weights** — what `--seed N` produces before any arithmetic touches the
data. These come from the project's own generator (`core/rng.h`), which is
integer arithmetic, and one conversion to float. Nothing in that path depends on
the toolchain, so these are bit-identical everywhere, and every row below
confirms it.

**Trained weights** — the same after a fixed short training run. This is where
kernel selection, threading, and compiler reassociation can change the last bits.
A difference here is a fact about float arithmetic, not a defect.

## The contract

| axis | initial weights | trained weights | established by |
|---|---|---|---|
| (a) the same binary, run twice | bit-identical | **bit-identical** | `tests/integration/test_determinism.sh` — two independent runs, 6 artifacts compared byte for byte, dropout on and `--shuffle` on |
| (b) a rebuild with the same compiler and flags | bit-identical | **bit-identical** | reproducibility matrix, row 2 |
| (c) a different compiler (gcc 11.4 vs clang 14) | bit-identical | **not guaranteed** | reproducibility matrix, row 3 |
| (d) a different optimization level (`-Os` vs `-O3`) | bit-identical | **not guaranteed** | reproducibility matrix, row 5 |
| (e) a different CPU ISA (`DRANZER_CPU_ISA` cap) | bit-identical | **not guaranteed** | reproducibility matrix, rows 6–7 |
| (f) serial vs OpenMP, same compiler | bit-identical | **bit-identical** | reproducibility matrix, row 4 — gcc serial and gcc with 2 OpenMP threads produce the same hash |
| (g) CPU vs GPU backend | bit-identical | **bit-identical on this machine and driver; not guaranteed across GPUs** | measured directly (see below) |
| (h) a different C library | bit-identical | inherits whichever of the above applies | by construction plus a property test (see below) |

### Rows (c), (d), and (e): why "not guaranteed" is the right answer

These are not bugs to be fixed. A different compiler reassociates float
reductions differently; `-Os` vectorizes differently; a capped ISA dispatches to a
different matmul kernel with a different accumulation order. Float addition is not
associative, so the last bits move. The project's own kernel tests check that
every path agrees with the portable reference to a stated tolerance, which is the
correct guarantee for this axis.

The consequence for results is concrete and it is why the row exists: **comparing
trained artifacts across machines requires pinning `DRANZER_CPU_ISA` on both**,
and comparing them across compilers is not meaningful at all. Any result that
quotes a weight hash has to name its compiler and its ISA cap.

### Row (f): OpenMP is bit-identical, and that is by design

`core/parallel.c` parallelizes only over independent output rows and attention
heads. No reduction crosses a thread, so there is no summation order for the
thread count to change. `tests/core/test_parallel.c` asserts this at the kernel
level; the matrix confirms it end to end, through a whole training run with
dropout and shuffling active.

### Row (g): how CPU vs GPU was measured

The same fixed run, once without `--gpu` and once with it, on a machine where
`gpu_matmul_available()` returns true. Both produced
`6e721cb1564985b6…` — byte-identical. Forward matmuls dispatch to the GPU
unconditionally when one is usable (`core/matmul_dispatch.c`), so the GPU path was
genuinely exercised; the GPU *backward* is gated on a work threshold and at this
model size stayed on the CPU.

Not guaranteed across GPUs: the hand-written PTX kernel's reduction order depends
on the block decomposition, so a device with a different SM configuration may
differ in the last bits. `tests/gpu/test_gpu_matmul_backward.c` checks agreement
against the CPU reference to a tolerance rather than exactly, which is the honest
guarantee for that axis.

### Row (h): the C library

There is no second libc on the development machine, so this row is not backed by
a build comparison. It is backed by two things that together are stronger than
one comparison would be:

1. **No libc generator remains on any path that affects model state or generated
   tokens.** `tests/integration/test_libc_independence.sh` proves the property
   directly rather than by inspection: it replaces `rand`, `srand`, `random`, and
   `rand_r` with implementations returning deliberate garbage, preloads them ahead
   of libc, and requires the weight fingerprint and the generated text to be
   unchanged. It first verifies that the preload actually takes effect, so it
   cannot pass vacuously.
2. **The generator's output is pinned to literals.** `tests/core/test_rng.c`
   asserts the exact initial state and the first six draws of each named stream
   against constants. That test fails on any platform whose arithmetic differs.

This used to be the largest hole in the project's reproducibility story:
initialization drew from `rand()`, which is implementation-defined, so `--seed 42`
named a run *on one C library* and nobody else could reproduce it from the seed.

## Regenerating this table

```sh
scripts/repro/reproducibility_matrix.sh docs/generated/repro-matrix.md
```

Seven clean rebuilds, a few minutes. The measured table as of the commit that
added this document:

| build / runtime | initial weights | matches | trained weights | matches |
|---|---|:-:|---|:-:|
| clang -O3 -ffast-math (reference) | `D0190BD0DA47CA15` | yes | `6e721cb1564985b6` | yes |
| clang, rebuilt from clean | `D0190BD0DA47CA15` | yes | `6e721cb1564985b6` | yes |
| gcc -O3 -ffast-math | `D0190BD0DA47CA15` | yes | `527584a7929531ea` | no |
| gcc, OpenMP (2 threads) | `D0190BD0DA47CA15` | yes | `527584a7929531ea` | no |
| clang, size-optimized (-Os) | `D0190BD0DA47CA15` | yes | `e6971e976386d520` | no |
| clang, `DRANZER_CPU_ISA=baseline` | `D0190BD0DA47CA15` | yes | `1b9e919fd51b3811` | no |
| clang, `DRANZER_CPU_ISA=avx2` | `D0190BD0DA47CA15` | yes | `68ede483c25891bd` | no |

Measured on Linux 5.15 (WSL2), i5-11320H, clang 14.0.0 and gcc 11.4.0. The
"matches" columns compare against the reference row. Note that gcc and
gcc+OpenMP share a trained hash — that is row (f) — and that all seven share an
initial-weight fingerprint.

## What a seed does not fix

Being explicit about the boundary, since a reader will assume more than is true:

- **The corpus.** A seed does not name the data. That is what the manifests in
  [`data/corpora/`](../data/corpora/README.md) are for, and the 178 MB corpus this
  project has trained on has unverified provenance, so no reported quality number
  may currently rest on it.
- **The tokenizer vocabulary**, unless it is frozen or loaded. A vocabulary
  trained on different bytes assigns different ids, and every token in a
  checkpoint then means something else.
- **Wall time, throughput, or memory.** Nothing about a seed makes a timing
  reproducible; those need the provenance record the benchmarks already carry.

## Pending verification: NUL bytes in a corpus

`tests/core/test_fuzz_tokenizer.c` found that the C-string tokenizer dropped
`0x00`, making the trained byte stream differ from the file its manifest hashed.
The implementation now uses length-delimited token bytes and versioned binary-safe
serialization, but that change is awaiting code review and has deliberately not
been built or run. The acceptance state remains tracked in
[`docs/research-checklist.md`](research-checklist.md).
