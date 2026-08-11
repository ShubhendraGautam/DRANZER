#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work_dir="$(mktemp -d /tmp/dranzer-package-check.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/first" "$work_dir/second"

bash "$repo_root/scripts/release/package.sh" "$work_dir/first" >/dev/null
bash "$repo_root/scripts/release/package.sh" "$work_dir/second" >/dev/null

if ! diff -u \
        <(cd "$work_dir/first" && find . -maxdepth 1 -type f -printf '%f\n' | sort) \
        <(cd "$work_dir/second" && find . -maxdepth 1 -type f -printf '%f\n' | sort); then
    echo "release package runs produced different file sets" >&2
    exit 1
fi

for first in "$work_dir/first"/*; do
    name="$(basename "$first")"
    if ! cmp -s "$first" "$work_dir/second/$name"; then
        echo "release package is not reproducible: $name" >&2
        exit 1
    fi
done

for archive in "$work_dir/first"/*.tar.gz; do
    if ! members="$(tar -tzf "$archive")"; then
        echo "release archive cannot be listed: $(basename "$archive")" >&2
        exit 1
    fi
    if grep -Eq '(^/|(^|/)\.\.(/|$))' <<< "$members"; then
        echo "release archive has an unsafe member path: $(basename "$archive")" >&2
        exit 1
    fi
done

(cd "$work_dir/first" && sha256sum -c SHA256SUMS)

echo "release packages: repeated archives and checksums are byte-identical"
