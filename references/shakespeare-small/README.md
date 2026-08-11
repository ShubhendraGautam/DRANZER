# Shakespeare small reference model

[← Reference implementation checklist](../../docs/design-checklist.md)

This recipe publishes DRANZER's small, reproducible reference artifact. It is a
two-layer, 32-wide decoder trained for one pass over the first 90% of Project
Gutenberg ebook 100 and evaluated on the final 10%. The tokenizer is the exact
byte vocabulary plus PAD, UNK, BOS, and EOS (`vocab_size = 260`), so there are
no learned BPE merges whose training could obscure the protocol.

This is a release smoke/reference model, not a competitive Shakespeare result.
The split is a byte boundary, not a work-aware split, and the corpus contains
the Project Gutenberg header and license. The expected cross-entropy and
perplexity are useful for reproducing this artifact; they are not directly
comparable to Tiny Shakespeare, enwik8, WikiText, or any other published
benchmark.

## Publish

From a clean checkout:

```sh
scripts/reference/publish_shakespeare_small.sh publish \
  --output dist/reference/shakespeare-small-v1
```

The command verifies or downloads the exact corpus, materializes its recorded
split, performs a clean GCC build, trains with every trajectory-affecting flag
spelled out in [`recipe.env`](recipe.env), evaluates the saved bundle again,
and writes a release directory containing:

- the self-contained model bundle and compatibility tokenizer sidecar;
- the resolved training configuration and raw evaluation output;
- the corpus manifest and recipe used;
- `reference-model.manifest`, which records hashes and expected held-out token
  count, cross-entropy, perplexity, and their absolute replay tolerance.

Nothing is published merely because training exited successfully. The command
checks the second evaluation against the metrics saved during training before
making the staged directory visible at the requested output path.

To use an already downloaded corpus or a reviewed executable:

```sh
scripts/reference/publish_shakespeare_small.sh publish \
  --corpus /path/to/pg100.txt \
  --app /path/to/app.out \
  --output dist/reference/shakespeare-small-v1
```

`--app` deliberately skips the build. This makes it possible to review the
source first and then run the final bundled validation with the exact binary
that passed that review.

## Verify a published directory

Static verification needs no executable:

```sh
scripts/reference/publish_shakespeare_small.sh verify \
  --package dist/reference/shakespeare-small-v1
```

Supplying both the corpus and executable additionally replays evaluation and
requires the observed metrics to match the package's expected values:

```sh
scripts/reference/publish_shakespeare_small.sh verify \
  --package dist/reference/shakespeare-small-v1 \
  --corpus /path/to/pg100.txt \
  --app src/app.out
```

The source text is public domain in the USA according to the [Project Gutenberg
catalog](https://www.gutenberg.org/ebooks/100). The downloaded file remains
subject to the embedded [Project Gutenberg license and trademark
terms](https://www.gutenberg.org/policy/license.html), including the instruction
to check local law outside the USA. The release package carries the unmodified
corpus manifest but not the corpus itself.
