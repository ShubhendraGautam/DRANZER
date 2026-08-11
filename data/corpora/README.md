# Corpus manifests

[← Back to README](../../README.md)

A manifest is what makes a corpus citable. Without one, a quality number is
measured on bytes no reader can obtain, and there is no way to tell whether two
runs used the same data or whether a third party could get it.

The rule this directory exists to enforce:

> **No reported result may name a corpus that has no manifest here, and no
> result may be reported against a manifest whose `provenance` is not
> `verified`.**

That second clause is doing real work right now: the 178 MB corpus this project
has been training on has an unverified origin (see
[`simple-wikipedia-178mb.manifest`](simple-wikipedia-178mb.manifest)), so it
cannot back a reported quality number until someone establishes where it came
from. The manifest exists so that gap is written down rather than discovered by a
reader.

## What a manifest is

A plain-text file of `key = value` lines. `#` begins a comment. Every key below
is required; a manifest missing one is invalid and
`scripts/corpus.sh verify` says so.

| key | meaning |
|---|---|
| `name` | Stable identifier, used in result records and repro scripts. |
| `description` | What the text is, in one line. |
| `provenance` | `verified` or `unverified`. See below. |
| `source_url` | Where the bytes came from, or `UNKNOWN`. |
| `license` | SPDX identifier or a URL, or `UNKNOWN`. |
| `retrieved` | ISO date the bytes were obtained, or `UNKNOWN`. |
| `encoding` | Byte encoding, e.g. `utf-8`. |
| `bytes` | Exact byte count of the whole corpus. |
| `lines` | Exact count of `\n` bytes. |
| `sha256` | Hash of the whole corpus. |
| `split_train_bytes` | Byte offset where the training split ends and validation begins. |
| `split_train_sha256` | Hash of bytes `[0, split_train_bytes)`. |
| `split_validation_sha256` | Hash of bytes `[split_train_bytes, bytes)`. |
| `reconstruct` | A command that produces the exact bytes, or `UNAVAILABLE`. |

## `provenance = verified` means

Someone has confirmed that `reconstruct` produces bytes matching `sha256` from
`source_url`, under the stated `license`. Not that the file exists locally, and
not that the hash matches a local copy - a hash proves two runs used the same
bytes, which is identity. Availability is a separate property and it is the one a
reader needs.

## Why splits are byte offsets

A split defined as "the last 5%" is a description; a split defined as "byte
169787047" is a fact, and the two sides can be hashed independently. The offsets
here are placed at the first newline at or after the intended fraction, so
neither side begins mid-line, and the resulting number is then fixed forever -
recomputing the fraction on a corpus that grew by one byte would silently change
which text was held out.

Every held-out number this project reports has to name the manifest and the split
hash it was measured against, so a later reader can tell whether two results are
comparable at all.

## Usage

```sh
scripts/corpus.sh verify data/corpora/tiny-smoke.manifest path/to/corpus
scripts/corpus.sh split  data/corpora/tiny-smoke.manifest path/to/corpus outdir/
scripts/corpus.sh create --name my-corpus --path file.txt   # a manifest skeleton
```

The verified `gutenberg-shakespeare-100.manifest` backs the small release
reference model. `tiny-smoke.manifest` remains test-only, and the unverified
178 MB corpus remains barred from reported quality results.
