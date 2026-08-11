#!/usr/bin/env bash
# Mutation checks for a generated Shakespeare reference-model package.
set -euo pipefail

package="${1:-}"
publisher="${2:-}"
if [ ! -d "$package" ] || [ ! -x "$publisher" ]; then
    echo "Usage: test_reference_package.sh <package-dir> <publisher-script>" >&2
    exit 2
fi

work="$(mktemp -d /tmp/dranzer-reference-package-test.XXXXXX)"
trap "rm -rf '$work'" EXIT

expect_rejected() {
    local label="$1"
    if "$publisher" verify --package "$work/package" >/dev/null 2>&1; then
        echo "FAIL: $label was accepted" >&2
        exit 1
    fi
    echo "PASS: $label rejected"
}

fresh_copy() {
    rm -rf "$work/package"
    cp -a "$package" "$work/package"
}

"$publisher" verify --package "$package"

fresh_copy
model_file="$(sed -n 's/^model_file[[:space:]]*=[[:space:]]*//p' \
    "$work/package/reference-model.manifest" | head -n 1)"
printf '\0' >> "$work/package/$model_file"
expect_rejected "model payload corruption"

fresh_copy
sed -i 's|^model_file[[:space:]]*=.*|model_file = ../escape.pth|' \
    "$work/package/reference-model.manifest"
expect_rejected "package path traversal"

fresh_copy
sed -i 's/^sha256[[:space:]]*=.*/sha256 = 0000000000000000000000000000000000000000000000000000000000000000/' \
    "$work/package/corpus.manifest"
# Update the outer file hash so this reaches the semantic corpus-identity
# comparison instead of stopping at the ordinary package checksum.
corpus_hash="$(sha256sum "$work/package/corpus.manifest" | cut -d' ' -f1)"
sed -i "s/^corpus_manifest_sha256[[:space:]]*=.*/corpus_manifest_sha256 = $corpus_hash/" \
    "$work/package/reference-model.manifest"
expect_rejected "forged corpus identity"

echo "reference package mutation checks passed"
