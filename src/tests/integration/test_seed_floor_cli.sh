#!/usr/bin/env bash
# Strict-input and adaptive-status checks for seed_floor.out.
set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -x "$1" ]; then
    echo "Usage: test_seed_floor_cli.sh <seed_floor.out>" >&2
    exit 2
fi
tool="$(realpath "$1")"
work="$(mktemp -d /tmp/dranzer-seed-floor-test.XXXXXX)"
trap "rm -rf '$work'" EXIT
hash=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

cat > "$work/constant.csv" <<EOF
seed,validation_cross_entropy,model_sha256
1,2.5,$hash
2,2.5,$hash
3,2.5,$hash
4,2.5,$hash
EOF
"$tool" "$work/constant.csv" --minimum 4 --maximum 8 \
    --target-ratio 0.25 --resamples 2000 --output "$work/constant.floor"
grep -qx 'status = ready' "$work/constant.floor"
grep -qx 'noise_floor = 0.000000000000' "$work/constant.floor"

head -n 4 "$work/constant.csv" > "$work/early.csv"
"$tool" "$work/early.csv" --minimum 4 --maximum 8 \
    --target-ratio 0.25 --resamples 2000 --output "$work/early.floor"
grep -qx 'status = collect_more' "$work/early.floor"
grep -qx 'recommended_total = 4' "$work/early.floor"

cat > "$work/duplicate.csv" <<EOF
seed,validation_cross_entropy,model_sha256
1,2.5,$hash
1,2.4,$hash
EOF
if "$tool" "$work/duplicate.csv" >/dev/null 2>&1; then
    echo "FAIL: duplicate seed accepted" >&2
    exit 1
fi

cat > "$work/nonfinite.csv" <<EOF
seed,validation_cross_entropy,model_sha256
1,nan,$hash
2,2.4,$hash
EOF
if "$tool" "$work/nonfinite.csv" >/dev/null 2>&1; then
    echo "FAIL: non-finite loss accepted" >&2
    exit 1
fi

sed 's/0123/bad!/' "$work/constant.csv" > "$work/bad-hash.csv"
if "$tool" "$work/bad-hash.csv" >/dev/null 2>&1; then
    echo "FAIL: malformed model hash accepted" >&2
    exit 1
fi

echo "seed-floor CLI checks passed"
