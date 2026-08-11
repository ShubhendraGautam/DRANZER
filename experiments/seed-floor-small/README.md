# Small-tier seed-variance floor

This experiment measures how much held-out cross-entropy moves when only model
initialization and training randomness change. It uses the same architecture,
corpus, tokenizer, and training controls as the small Shakespeare reference
model; `seed` is the only changing input.

The committed [`recipe.env`](recipe.env) is the pre-run contract:

- collect at least 8 seeds and no more than 20;
- define the architecture-comparison noise floor as the sample standard
  deviation of held-out cross-entropy across those seeds;
- after the minimum, stop only when the deterministic 95% bootstrap interval
  for the mean has half-width no greater than 0.60 times that observed spread;
- if the criterion is not met at 20, report `limit_reached` and do not promote
  the floor as ready.

This separates two quantities that are often confused. `noise_floor` is
between-seed spread and is passed to `stat_paired_compare()`. The bootstrap
interval measures how precisely the experiment has estimated the mean and is
used only to choose N.

After source review, run:

```sh
scripts/research/measure_seed_floor.sh \
  --output dist/research/seed-floor-small-v1
```

Use `--corpus FILE` to avoid downloading the manifest-pinned corpus. Use
`--app FILE --analyzer FILE` together to run the reviewed binaries from the
final bundled build instead of rebuilding. The output keeps every seed/model
hash and resolved configuration, plus `seed-floor.manifest`; interrupted or
failed work is removed with its temporary directory and is never presented as
a completed result.

The command exits nonzero if the pre-registered maximum is reached without the
precision target. That outcome is evidence that more seeds or a revised,
separately versioned protocol is required—not permission to loosen the target
after seeing the result.
