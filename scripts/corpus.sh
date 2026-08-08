#!/usr/bin/env bash
#
# Corpus manifests: verify a corpus against one, materialize its splits, or draft
# a new one. See data/corpora/README.md for the format and for why the rule
# exists.
#
# The point of this being a script rather than a convention is that a convention
# does not fail a build. `verify` exits non-zero on a hash mismatch, a missing
# key, or - deliberately - on a manifest whose provenance is unverified when
# --require-verified is passed, which is what a repro script for a reported
# result should use.
set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  corpus.sh verify <manifest> <corpus-file> [--require-verified]
      Check byte count, line count, whole-file hash, and both split hashes.

  corpus.sh split <manifest> <corpus-file> <output-dir>
      Write train.txt and validation.txt at the manifest's byte offset, and
      verify each side against its recorded hash before declaring success.

  corpus.sh create --name <id> --path <corpus-file> [--fraction 0.95]
      Print a manifest skeleton with the identity fields filled in and the
      provenance fields left as UNKNOWN for a human to complete.

  corpus.sh list
      Every manifest in data/corpora, with its provenance status.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- manifest reading -------------------------------------------------------
# Values are read one key at a time rather than sourced: a manifest is data, and
# `source`ing it would make an untrusted file executable.
manifest_get() { # $1 = manifest, $2 = key
    sed -n "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*\(.*[^[:space:]]\)[[:space:]]*$/\1/p" "$1" |
        head -n 1
}

REQUIRED_KEYS=(name description provenance source_url license retrieved encoding
               bytes lines sha256 split_train_bytes split_train_sha256
               split_validation_sha256 reconstruct)

check_required_keys() { # $1 = manifest
    local missing=0 key
    for key in "${REQUIRED_KEYS[@]}"; do
        if [ -z "$(manifest_get "$1" "$key")" ]; then
            echo "  missing required key: $key"
            missing=1
        fi
    done
    return $missing
}

sha256_of_stdin() { sha256sum | cut -d' ' -f1; }

# --- verify -----------------------------------------------------------------
cmd_verify() {
    local manifest="${1:-}" corpus="${2:-}" require_verified=0
    shift 2 2>/dev/null || { usage; exit 2; }
    [ "${1:-}" = "--require-verified" ] && require_verified=1

    if [ ! -f "$manifest" ] || [ ! -f "$corpus" ]; then
        echo "corpus.sh verify: need an existing manifest and corpus file" >&2
        exit 2
    fi

    echo "manifest: $manifest"
    echo "corpus:   $corpus"
    local failed=0
    check_required_keys "$manifest" || failed=1

    local want_bytes want_lines want_sha provenance
    want_bytes="$(manifest_get "$manifest" bytes)"
    want_lines="$(manifest_get "$manifest" lines)"
    want_sha="$(manifest_get "$manifest" sha256)"
    provenance="$(manifest_get "$manifest" provenance)"

    local got_bytes got_lines got_sha
    got_bytes="$(wc -c < "$corpus" | tr -d ' ')"
    got_lines="$(wc -l < "$corpus" | tr -d ' ')"
    got_sha="$(sha256_of_stdin < "$corpus")"

    compare() { # $1 label, $2 want, $3 got
        if [ "$2" = "$3" ]; then
            printf '  %-16s %s\n' "$1" "$3"
        else
            printf '  %-16s MISMATCH: manifest says %s, file has %s\n' "$1" "$2" "$3"
            failed=1
        fi
    }
    compare bytes "$want_bytes" "$got_bytes"
    compare lines "$want_lines" "$got_lines"
    compare sha256 "$want_sha" "$got_sha"

    # Split hashes, read directly out of the file rather than by writing the
    # splits to disk - verifying should not need 170 MB of scratch space.
    local split_bytes want_train_sha want_val_sha got_train_sha got_val_sha
    split_bytes="$(manifest_get "$manifest" split_train_bytes)"
    want_train_sha="$(manifest_get "$manifest" split_train_sha256)"
    want_val_sha="$(manifest_get "$manifest" split_validation_sha256)"
    if [ -n "$split_bytes" ]; then
        got_train_sha="$(head -c "$split_bytes" "$corpus" | sha256_of_stdin)"
        got_val_sha="$(tail -c "+$((split_bytes + 1))" "$corpus" | sha256_of_stdin)"
        compare "train split" "$want_train_sha" "$got_train_sha"
        compare "valid split" "$want_val_sha" "$got_val_sha"
    fi

    printf '  %-16s %s\n' provenance "$provenance"
    if [ "$require_verified" -eq 1 ] && [ "$provenance" != "verified" ]; then
        echo ""
        echo "REFUSED: this corpus has provenance='$provenance'. A reported result"
        echo "may not be measured on bytes a reader cannot obtain - see"
        echo "data/corpora/README.md. Establish the source, license, and a"
        echo "reconstruct command, then set provenance = verified."
        failed=1
    fi

    echo ""
    if [ "$failed" -eq 0 ]; then
        echo "CORPUS VERIFIED: $(manifest_get "$manifest" name)"
    else
        echo "CORPUS VERIFICATION FAILED: $(manifest_get "$manifest" name)"
    fi
    return $failed
}

# --- split ------------------------------------------------------------------
cmd_split() {
    local manifest="${1:-}" corpus="${2:-}" outdir="${3:-}"
    if [ ! -f "$manifest" ] || [ ! -f "$corpus" ] || [ -z "$outdir" ]; then
        usage
        exit 2
    fi
    mkdir -p "$outdir" || exit 1

    local split_bytes want_train_sha want_val_sha
    split_bytes="$(manifest_get "$manifest" split_train_bytes)"
    want_train_sha="$(manifest_get "$manifest" split_train_sha256)"
    want_val_sha="$(manifest_get "$manifest" split_validation_sha256)"
    if [ -z "$split_bytes" ]; then
        echo "corpus.sh split: manifest has no split_train_bytes" >&2
        exit 1
    fi

    head -c "$split_bytes" "$corpus" > "$outdir/train.txt"
    tail -c "+$((split_bytes + 1))" "$corpus" > "$outdir/validation.txt"

    local got_train got_val failed=0
    got_train="$(sha256_of_stdin < "$outdir/train.txt")"
    got_val="$(sha256_of_stdin < "$outdir/validation.txt")"
    # Verified after writing, not assumed: a truncated write or a full disk would
    # otherwise produce a training run on silently different data.
    [ "$got_train" = "$want_train_sha" ] || { echo "train.txt hash mismatch" >&2; failed=1; }
    [ "$got_val" = "$want_val_sha" ] || { echo "validation.txt hash mismatch" >&2; failed=1; }

    if [ "$failed" -eq 0 ]; then
        echo "wrote $outdir/train.txt ($split_bytes bytes) and $outdir/validation.txt"
        echo "both match the manifest's recorded split hashes"
    fi
    return $failed
}

# --- create -----------------------------------------------------------------
cmd_create() {
    local name="" path="" fraction="0.95"
    while [ $# -gt 0 ]; do
        case "$1" in
            --name) name="${2:-}"; shift 2 ;;
            --path) path="${2:-}"; shift 2 ;;
            --fraction) fraction="${2:-}"; shift 2 ;;
            *) usage; exit 2 ;;
        esac
    done
    if [ -z "$name" ] || [ ! -f "$path" ]; then usage; exit 2; fi

    # The split offset is computed in python rather than in shell because it has
    # to land on a newline at or after a fraction of the file, and the two split
    # hashes have to be taken in one pass over a file that may be gigabytes.
    python3 - "$path" "$name" "$fraction" <<'PYTHON'
import hashlib, sys
path, name, fraction = sys.argv[1], sys.argv[2], float(sys.argv[3])

total = 0
lines = 0
whole = hashlib.sha256()
with open(path, 'rb') as f:
    while True:
        block = f.read(1 << 20)
        if not block:
            break
        total += len(block)
        lines += block.count(b'\n')
        whole.update(block)

target = int(total * fraction)
with open(path, 'rb') as f:
    f.seek(target)
    tail = f.read(1 << 20)
newline = tail.find(b'\n')
split = total if newline == -1 else target + newline + 1

train = hashlib.sha256()
valid = hashlib.sha256()
with open(path, 'rb') as f:
    remaining = split
    while remaining > 0:
        block = f.read(min(1 << 20, remaining))
        remaining -= len(block)
        train.update(block)
    while True:
        block = f.read(1 << 20)
        if not block:
            break
        valid.update(block)

print(f"""# Manifest drafted by scripts/corpus.sh create. The identity fields below are
# measured; every UNKNOWN is a fact about this corpus that nobody has recorded
# yet, and each one has to be filled in before a result may be reported against
# it. See data/corpora/README.md.

name = {name}
description = TODO one line: what this text is
provenance = unverified
source_url = UNKNOWN
license = UNKNOWN
retrieved = UNKNOWN
encoding = utf-8
bytes = {total}
lines = {lines}
sha256 = {whole.hexdigest()}

split_train_bytes = {split}
split_train_sha256 = {train.hexdigest()}
split_validation_sha256 = {valid.hexdigest()}

reconstruct = UNAVAILABLE""")
PYTHON
}

# --- list -------------------------------------------------------------------
cmd_list() {
    local any=0
    for manifest in "$repo_root"/data/corpora/*.manifest; do
        [ -f "$manifest" ] || continue
        any=1
        printf '%-28s %-12s %14s bytes  %s\n' \
            "$(manifest_get "$manifest" name)" \
            "$(manifest_get "$manifest" provenance)" \
            "$(manifest_get "$manifest" bytes)" \
            "$(basename "$manifest")"
    done
    [ "$any" -eq 1 ] || echo "no manifests in data/corpora"
}

case "${1:-}" in
    verify) shift; cmd_verify "$@" ;;
    split)  shift; cmd_split "$@" ;;
    create) shift; cmd_create "$@" ;;
    list)   shift; cmd_list "$@" ;;
    *)      usage; exit 2 ;;
esac
