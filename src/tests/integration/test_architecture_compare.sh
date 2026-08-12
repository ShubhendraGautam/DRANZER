#!/usr/bin/env bash
set -euo pipefail

analyzer="${1:-./architecture_compare.out}"
work="$(mktemp -d /tmp/dranzer-architecture-compare-test.XXXXXX)"
trap 'rm -rf "$work"' EXIT

cat > "$work/resolved.csv" <<'EOF'
seed,baseline_validation_cross_entropy,feature_validation_cross_entropy,baseline_model_sha256,feature_model_sha256
1,2.0,1.8,0000000000000000000000000000000000000000000000000000000000000000,1111111111111111111111111111111111111111111111111111111111111111
2,2.2,1.9,2222222222222222222222222222222222222222222222222222222222222222,3333333333333333333333333333333333333333333333333333333333333333
3,2.1,1.85,4444444444444444444444444444444444444444444444444444444444444444,5555555555555555555555555555555555555555555555555555555555555555
EOF

"$analyzer" "$work/resolved.csv" --noise-floor 0.05 \
    --resamples 2000 --bootstrap-seed 7 --output "$work/resolved.manifest"
grep -qx 'direction = feature_better' "$work/resolved.manifest"
grep -qx 'pair_count = 3' "$work/resolved.manifest"

sed 's/1.85/2.10/' "$work/resolved.csv" > "$work/unresolved.csv"
"$analyzer" "$work/unresolved.csv" --noise-floor 1.0 \
    --resamples 2000 --bootstrap-seed 7 --output "$work/unresolved.manifest"
grep -qx 'direction = unresolved' "$work/unresolved.manifest"
grep -qx 'recommended_total = unbounded' "$work/unresolved.manifest"

if "$analyzer" "$work/resolved.csv" > /dev/null 2>&1; then
    echo 'architecture compare accepted a missing noise floor' >&2
    exit 1
fi
sed 's/^2,/1,/' "$work/resolved.csv" > "$work/duplicate.csv"
if "$analyzer" "$work/duplicate.csv" --noise-floor 0.05 > /dev/null 2>&1; then
    echo 'architecture compare accepted a duplicate seed' >&2
    exit 1
fi

echo 'ARCHITECTURE COMPARE CHECK PASSED'
