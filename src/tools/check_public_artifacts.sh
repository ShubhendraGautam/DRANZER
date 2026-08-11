#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
static="$root/libdranzer.a"
shared="$root/libdranzer.so"
symbol_baseline="$root/tests/compat/public-api-v1.symbols"

for artifact in "$static" "$shared" \
                "$root/examples/embed_infer.out" \
                "$root/examples/embed_generate.out"; do
    if [[ ! -f "$artifact" ]]; then
        echo "public artifact missing: $artifact" >&2
        exit 1
    fi
done

if [[ ! -f "$symbol_baseline" ]]; then
    echo "public ABI symbol baseline missing: $symbol_baseline" >&2
    exit 1
fi
mapfile -t required_symbols < <(
    sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$symbol_baseline"
)

static_symbols="$(nm -g --defined-only "$static" | awk 'NF >= 3 { print $3 }')"
for symbol in "${required_symbols[@]}"; do
    if ! grep -qx "$symbol" <<<"$static_symbols"; then
        echo "static library is missing public symbol: $symbol" >&2
        exit 1
    fi
done

shared_symbols="$(nm -D --defined-only "$shared" | awk 'NF >= 3 { print $3 }' | sed 's/@.*//')"
for symbol in "${required_symbols[@]}"; do
    if ! grep -qx "$symbol" <<<"$shared_symbols"; then
        echo "shared library is missing public symbol: $symbol" >&2
        exit 1
    fi
done

while IFS= read -r symbol; do
    [[ -z "$symbol" || "$symbol" == "DRANZER_1.0" ]] && continue
    matched=0
    for expected in "${required_symbols[@]}"; do
        if [[ "$symbol" == "$expected" ]]; then
            matched=1
            break
        fi
    done
    if [[ $matched -eq 0 ]]; then
        echo "shared library exposes undocumented symbol: $symbol" >&2
        exit 1
    fi
done <<<"$shared_symbols"

echo "public artifacts: static/shared linkage and DRANZER_1.0 exports verified"
