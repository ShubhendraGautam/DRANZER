# Public C embedding API

[← Back to README](../README.md)

`src/include/dranzer.h` is the supported embedding boundary. It includes only standard size/integer
headers and exposes opaque `dranzer_model_t`, `dranzer_tokenizer_t`, `dranzer_cache_t`, and
`dranzer_generation_t` handles. Applications do not depend on the layout of `neural_model_t`, the
BPE hash maps, activation caches, or CLI configuration structures.

The project release is reported by `DRANZER_VERSION_STRING` and `dranzer_version_string()`. The
first independent API contract version is `DRANZER_API_VERSION == 1`; `dranzer_api_version()` lets an
application compare its header with the loaded library. Every fallible call returns one
`dranzer_status_t`; `dranzer_status_string()` supplies a stable diagnostic phrase without printing
from the wrapper. Handles are stateful and are not safe for concurrent calls. Separate handles may
be used by separate threads when the underlying model is not being mutated.

API version 1 inference is CPU-only; it does not expose the CLI's optional `--gpu` request. This is
an API-surface limit, not a link dependency—the packaged library remains usable on hosts with or
without an NVIDIA driver.

No embeddable runtime module writes to stdout or stderr. Model metrics are returned as structured
fields internally, invalid CPU/GPU environment overrides have queryable configuration statuses,
and the CUDA wrapper retains its last operation/driver-code/message tuple. The standalone CLI and
GPU probe remain executable frontends and choose how to render those values. A DEBUG build is also
silent unless the embedding application supplies `DRANZER_DEBUG_SINK` at compile time.

## Loading and ownership

```c
#include "dranzer.h"

dranzer_model_t *model = NULL;
dranzer_tokenizer_t *tokenizer = NULL;
dranzer_bundle_info_t info = DRANZER_BUNDLE_INFO_INIT;

dranzer_status_t rc = dranzer_bundle_load(
    "model.bin", DRANZER_LOAD_MMAP, &model, &tokenizer, &info);
if (rc != DRANZER_OK) {
    /* dranzer_status_string(rc) */
}

/* use handles */

dranzer_tokenizer_free(tokenizer);
dranzer_model_free(model);
```

`DRANZER_LOAD_COPY` accepts bundle versions 1 through 3 and produces ordinary writable internal
parameters. `DRANZER_LOAD_MMAP` accepts lossless versions 1 and 3 and uses the checked read-only mapping described in
[Model bundle format](model-bundle.md). The public API currently exposes inference, so both modes
have the same public operations.

Model and tokenizer handles returned by `dranzer_bundle_load()` are independently owned. Cache and
generation handles retain their dependencies internally: it is safe to release the caller's
original model/tokenizer handles first, and the underlying objects live until the last dependent
handle is freed.

`dranzer_bundle_info_t` is a fixed 64-byte API-v1 record. Callers initialize `struct_size` through
`DRANZER_BUNDLE_INFO_INIT`; the library reports how many bytes it understands, the loaded wire
format version, provenance fields, and zeroed reserved space. This permits a newer caller to detect
an older library without letting either side write past the other's record.

## Token and logits buffers

Tokenizer strings are byte spans, not C strings. They may contain NUL. `dranzer_tokenize()` and
`dranzer_detokenize()` use a two-call buffer convention: pass a null output with a zero capacity to
receive the required count/size and `DRANZER_BUFFER_TOO_SMALL`, allocate, then call again. No partial
result is written when capacity is short.

`dranzer_model_forward()` accepts a token span and a logits buffer of at least
`dranzer_model_vocab_size()` floats. `dranzer_cache_create()` plus `dranzer_cache_forward()` is the
incremental equivalent; reset begins a new sequence without reallocating.

## Generation sessions

`dranzer_generation_create()` owns an incremental cache and logits workspace. Reset tokenizes a
prompt, prepends BOS for a special-aware tokenizer, truncates from the left when necessary, and
prefills the cache. `dranzer_generation_next_greedy()` returns one token and its exact decoded byte
piece. PAD, UNK, BOS, and unassigned vocabulary slots are excluded; EOS returns
`DRANZER_FINISHED` with an empty piece. A short piece buffer returns `DRANZER_BUFFER_TOO_SMALL`
without consuming the token, so the caller can resize and retry.

The initial stable surface deliberately exposes deterministic greedy stepping first. Sampling,
callbacks, and stop-sequence orchestration remain available to the repository CLI but are not yet
promised as public ABI.

## Building and linking

From the repository root, build the self-contained static and shared libraries plus both examples:

```bash
make -C src public-libs examples
```

Both libraries contain the model runtime, runtime-selected CPU kernels, optional dynamically loaded
CUDA path, and tokenizer; an embedding application does not link the internal `libattention.a`.
The only system link dependencies are the math and dynamic-loader libraries:

```bash
cc -Isrc/include app.c src/libdranzer.a -lm -ldl -o app-static
cc -Isrc/include app.c -Lsrc -ldranzer -lm -ldl -o app-shared
```

The shared object exports only the documented version-1 functions under the `DRANZER_1.0` symbol
version. Internal model, tokenizer, and backend symbols remain local. An application using the
shared form must install `libdranzer.so` on its loader path or provide an application-relative
rpath. See [`src/examples`](../src/examples/README.md) for a static full-forward example and a
shared incremental-generation example.

## Compatibility window

API version 1 is source- and ABI-compatible for the full 1.x release line. Existing function
signatures, enum values, status phrases, structure offsets, and exported symbols are not removed or
reinterpreted; compatible additions use new symbols and reserved structure space. An incompatible
change requires `DRANZER_API_VERSION == 2`, a new shared-object symbol namespace, and migration
notes. Opaque handle layouts are never part of the contract.

The checked-in `tests/compat/public-api-v1.symbols` baseline, compile-time layout/signature checks
in `test_public_compatibility.c`, and `public-api-check` shared-export inspection enforce that
window. Compatibility is promised within one operating-system/architecture ABI; a shared object is
not portable between, for example, x86-64 and AArch64.
