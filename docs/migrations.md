# Migration notes

[← Back to README](../README.md)

## Pre-0.5 embedding code to API version 1

External code should stop including `core/model.h`, CLI headers, or tokenizer implementation
headers. Include only `dranzer.h`, load a self-contained bundle with `dranzer_bundle_load()`, and
hold opaque handles. Cache and generation handles retain their dependencies, but every acquired
handle still needs its matching free function.

Initialize metadata before loading:

```c
dranzer_bundle_info_t info = DRANZER_BUNDLE_INFO_INIT;
dranzer_status_t rc = dranzer_bundle_load(
    path, DRANZER_LOAD_COPY, &model, &tokenizer, &info);
```

An all-zero or pre-0.5 metadata record is intentionally rejected: `struct_size` is what prevents a
new library from overrunning an older caller's record. Tokenizer text is now a byte span and may
contain NUL; use the documented two-call buffer convention instead of `strlen()` on decoded data.

## Internal archive to supported libraries

`libs/libattention.a` remains an internal tokenizer/hashmap archive. Embedders link the
self-contained `src/libdranzer.a` or `src/libdranzer.so` and the system math/dynamic-loader
libraries. The shared form exports only the `DRANZER_1.0` namespace.

## Legacy weights and tokenizer sidecars

The CLI still reads the old host-native weight file plus optional `.tokenizer` sidecar, but new
output is a canonical bundle containing architecture, weights, tokenizer, and provenance. There is
no in-place conversion command because legacy files do not contain all bundle provenance; keep the
matching sidecar when retaining one, and publish newly trained artifacts as bundles. Bundle
versions 1 through 3 are the portable compatibility window; checkpoint files are exact-resume state
and have a separate, narrower contract.

## Diagnostics

Embeddable code no longer writes errors or metrics to stdout/stderr. Check `dranzer_status_t`, use
`dranzer_status_string()` for its stable short phrase, and choose logging/rendering in the host
application. DEBUG builds remain silent unless the application supplies `DRANZER_DEBUG_SINK`.

## Version meanings

- Project SemVer (`DRANZER_VERSION_STRING`) describes the release as a whole.
- `DRANZER_API_VERSION` changes only for incompatible C API changes.
- Bundle format versions describe bytes on disk and are reported through
  `dranzer_bundle_info_t.format_version`.

Do not infer one version from another. A compatible project release can add a bundle reader or a
new public function without changing the API major.
