# Embedding examples

These programs include only the stable `dranzer.h` header. `embed_infer.c`
links the static library and prints the highest-logit next token for a prompt.
`embed_generate.c` links the shared library and writes exact, length-delimited
token bytes from deterministic greedy generation.

Build both libraries and examples from the repository root:

```bash
make -C src public-libs examples
./src/examples/embed_infer.out model.bin "hello"
./src/examples/embed_generate.out model.bin "hello" 20
```

The generated shared-library example carries relative development and SDK
runtime search paths, so it finds either `src/libdranzer.so` or an installed
adjacent `lib/libdranzer.so` without changing `LD_LIBRARY_PATH`.
