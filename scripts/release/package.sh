#!/usr/bin/env bash
# Produce normalized source and Linux embedding-SDK archives from one commit.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_dir="${1:-$repo_root/dist}"
version="$(tr -d '\r\n' < "$repo_root/VERSION")"
source_epoch="${SOURCE_DATE_EPOCH:-$(git -C "$repo_root" show -s --format=%ct HEAD)}"
arch="${DRANZER_RELEASE_ARCH:-$(uname -m)}"
release_cc="${DRANZER_RELEASE_CC:-cc}"

bash "$repo_root/scripts/release/check_version.sh" >/dev/null
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
    echo "release packaging requires a clean tracked worktree" >&2
    exit 1
fi
if [[ ! "$source_epoch" =~ ^[0-9]+$ || ! "$arch" =~ ^[0-9A-Za-z_.-]+$ ]]; then
    echo "invalid SOURCE_DATE_EPOCH or release architecture" >&2
    exit 1
fi
if ! command -v "$release_cc" >/dev/null 2>&1; then
    echo "release compiler not found: $release_cc" >&2
    exit 1
fi

required=(
    src/libdranzer.a
    src/libdranzer.so
    src/examples/embed_infer.out
    src/examples/embed_generate.out
)
for path in "${required[@]}"; do
    if [[ ! -f "$repo_root/$path" ]]; then
        echo "release artifact not built: $path" >&2
        exit 1
    fi
done

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
work_dir="$(mktemp -d /tmp/dranzer-package.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

source_root="dranzer-$version"
mkdir -p "$work_dir/source"
git -C "$repo_root" archive --format=tar --prefix="$source_root/" HEAD |
    tar -xf - -C "$work_dir/source"

sdk_root="dranzer-$version-linux-$arch"
mkdir -p "$work_dir/$sdk_root/include" "$work_dir/$sdk_root/lib" \
         "$work_dir/$sdk_root/bin" "$work_dir/$sdk_root/share/doc" \
         "$work_dir/$sdk_root/share/examples"
install -m 0644 "$repo_root/src/include/dranzer.h" \
    "$work_dir/$sdk_root/include/dranzer.h"
install -m 0644 "$repo_root/src/libdranzer.a" "$repo_root/src/libdranzer.so" \
    "$work_dir/$sdk_root/lib/"
install -m 0755 "$repo_root/src/examples/embed_infer.out" \
    "$repo_root/src/examples/embed_generate.out" "$work_dir/$sdk_root/bin/"
install -m 0644 "$repo_root/src/examples/embed_infer.c" \
    "$repo_root/src/examples/embed_generate.c" "$work_dir/$sdk_root/share/examples/"
install -m 0644 "$repo_root/README.md" "$repo_root/LICENSE" \
    "$repo_root/CHANGELOG.md" "$repo_root/docs/public-api.md" \
    "$repo_root/docs/model-bundle.md" "$repo_root/docs/migrations.md" \
    "$repo_root/docs/platform-support.md" \
    "$work_dir/$sdk_root/share/doc/"

commit="$(git -C "$repo_root" rev-parse HEAD)"
compiler_version="$("$release_cc" --version | sed -n '1p')"
printf 'version=%s\ncommit=%s\nsource_date_epoch=%s\narchitecture=%s\ncompiler=%s\n' \
    "$version" "$commit" "$source_epoch" "$arch" "$compiler_version" \
    > "$work_dir/$sdk_root/BUILD-INFO.txt"
chmod 0644 "$work_dir/$sdk_root/BUILD-INFO.txt"

normalize_tree() {
    local tree="$1"
    find "$tree" -type d -exec chmod 0755 {} +
    while IFS= read -r -d '' path; do
        touch -h -d "@$source_epoch" "$path"
    done < <(find "$tree" -print0)
}

write_archive() {
    local parent="$1" root="$2" destination="$3"
    normalize_tree "$parent/$root"
    tar --sort=name --format=posix \
        --pax-option=delete=atime,delete=ctime \
        --mtime="@$source_epoch" --owner=0 --group=0 --numeric-owner \
        -C "$parent" -cf - "$root" | gzip -n -9 > "$destination"
}

source_archive="$source_root-source.tar.gz"
sdk_archive="$sdk_root.tar.gz"
write_archive "$work_dir/source" "$source_root" "$output_dir/$source_archive"
write_archive "$work_dir" "$sdk_root" "$output_dir/$sdk_archive"
(
    cd "$output_dir"
    LC_ALL=C sha256sum "$source_archive" "$sdk_archive" > SHA256SUMS
)

echo "$output_dir/$source_archive"
echo "$output_dir/$sdk_archive"
echo "$output_dir/SHA256SUMS"
