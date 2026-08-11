#!/usr/bin/env bash
# Release determinism gate. A compiler must reproduce its own binaries across
# clean builds and its own training artifacts across fresh runs. Trained
# artifacts are deliberately not compared between GCC and Clang: floating-point
# reassociation makes that a non-contract (docs/reproducibility.md).
set -euo pipefail

src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d /tmp/dranzer-release-repro.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

if [[ $# -eq 0 ]]; then
    set -- gcc clang
fi

artifacts=(app.out libdranzer.a libdranzer.so fingerprint_model.out)
reference_fingerprint=""

for compiler in "$@"; do
    if ! command -v "$compiler" >/dev/null 2>&1; then
        echo "release reproducibility: compiler not found: $compiler" >&2
        exit 1
    fi
    "$compiler" --version | sed -n '1p'

    for pass in first second; do
        make -C "$src_dir" clean
        make -C "$src_dir" all public-libs fingerprint CC="$compiler"

        for artifact in "${artifacts[@]}"; do
            if [[ ! -f "$src_dir/$artifact" ]]; then
                echo "release reproducibility: missing $artifact ($compiler, $pass)" >&2
                exit 1
            fi
        done
        (
            cd "$src_dir"
            sha256sum "${artifacts[@]}"
        ) > "$work_dir/$compiler.$pass.sha256"
    done

    if ! diff -u "$work_dir/$compiler.first.sha256" \
                  "$work_dir/$compiler.second.sha256"; then
        echo "release reproducibility: $compiler clean builds differ" >&2
        exit 1
    fi
    echo "release reproducibility: $compiler binaries are byte-identical"

    fingerprint="$("$src_dir/fingerprint_model.out" --seed 42 --vocab 260 \
        --embedding-dim 16 --heads 2 --layers 2 --max-seq-len 32 --quiet)"
    if [[ -z "$reference_fingerprint" ]]; then
        reference_fingerprint="$fingerprint"
    elif [[ "$fingerprint" != "$reference_fingerprint" ]]; then
        echo "release reproducibility: initial weights differ across compilers" >&2
        echo "  reference: $reference_fingerprint" >&2
        echo "  $compiler: $fingerprint" >&2
        exit 1
    fi

    DRANZER_CPU_ISA=baseline \
        bash "$src_dir/tests/integration/test_determinism.sh" \
             "$src_dir/app.out"
done

echo "RELEASE REPRODUCIBILITY CHECK PASSED"
