#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
version="$(tr -d '\r\n' < "$repo_root/VERSION")"
header="$repo_root/src/include/dranzer.h"

if [[ ! "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$ ]]; then
    echo "VERSION is not SemVer: $version" >&2
    exit 1
fi

without_build="${version%%+*}"
prerelease=""
if [[ "$without_build" == *-* ]]; then
    prerelease="${without_build#*-}"
    IFS='.' read -r -a prerelease_parts <<< "$prerelease"
    for identifier in "${prerelease_parts[@]}"; do
        if [[ "$identifier" =~ ^[0-9]+$ &&
              ${#identifier} -gt 1 && "$identifier" == 0* ]]; then
            echo "VERSION has a zero-prefixed numeric prerelease identifier" >&2
            exit 1
        fi
    done
fi
core_version="${without_build%%-*}"
IFS='.' read -r version_major version_minor version_patch <<< "$core_version"

header_version="$(sed -n 's/^#define DRANZER_VERSION_STRING "\([^"]*\)"$/\1/p' "$header")"
if [[ "$header_version" != "$version" ]]; then
    echo "VERSION/header mismatch: $version != $header_version" >&2
    exit 1
fi
header_major="$(sed -n 's/^#define DRANZER_VERSION_MAJOR UINT32_C(\([0-9][0-9]*\))$/\1/p' "$header")"
header_minor="$(sed -n 's/^#define DRANZER_VERSION_MINOR UINT32_C(\([0-9][0-9]*\))$/\1/p' "$header")"
header_patch="$(sed -n 's/^#define DRANZER_VERSION_PATCH UINT32_C(\([0-9][0-9]*\))$/\1/p' "$header")"
header_prerelease="$(sed -n 's/^#define DRANZER_VERSION_PRERELEASE "\([^"]*\)"$/\1/p' "$header")"
if [[ "$header_major.$header_minor.$header_patch" != "$core_version" ||
      "$header_prerelease" != "$prerelease" ]]; then
    echo "VERSION/header component macros disagree" >&2
    exit 1
fi
if ! grep -Fqx "## [$version] - 2026-08-12" "$repo_root/CHANGELOG.md"; then
    echo "CHANGELOG.md has no dated entry for $version" >&2
    exit 1
fi
if [[ ! -s "$repo_root/docs/migrations.md" || ! -s "$repo_root/LICENSE" ]]; then
    echo "release migration or license document is missing" >&2
    exit 1
fi

if [[ "${GITHUB_REF_TYPE:-}" == "tag" &&
      "${GITHUB_REF_NAME:-}" != "v$version" ]]; then
    echo "tag ${GITHUB_REF_NAME:-<unset>} does not match VERSION v$version" >&2
    exit 1
fi

echo "version contract: $version"
