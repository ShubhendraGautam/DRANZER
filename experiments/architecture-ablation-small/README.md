# Paired small-tier architecture ablations

Each run changes exactly one architecture flag and pairs its held-out loss with
the same seed from a completed `seed-floor-small-v1` artifact. Corpus, split,
tokenizer, compiler, model dimensions, optimizer, schedule, and training
controls are inherited from that artifact. The baseline app hash must match, so
a rebuilt or changed executable cannot silently enter one arm.

The analyzer bootstraps feature-minus-baseline paired differences and compares
the entire 95% interval with the baseline's measured between-seed floor. Its
verdict is structural: `feature_better`, `feature_worse`, or `unresolved`.

Run one feature at a time:

```sh
scripts/research/measure_architecture_ablation.sh \
  --seed-floor dist/research/seed-floor-small-v1 \
  --architecture tied \
  --output dist/research/architecture-tied-small-v1
```

Accepted architecture names are `tied`, `rope`, `rmsnorm`, `gelu`, and
`swiglu`. Use `--corpus FILE` to reuse a local manifest-pinned corpus, and
`--app FILE --analyzer FILE` together to use reviewed binaries. The output
keeps every feature model, replay, resolved config, model hash, paired loss,
and the deterministic comparison manifest.
