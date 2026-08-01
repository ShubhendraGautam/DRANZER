#!/usr/bin/env bash
set -euo pipefail

app_path="${1:-./app.out}"
app_path="$(cd "$(dirname "$app_path")" && pwd)/$(basename "$app_path")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
work_dir="$(mktemp -d /tmp/dranzer-eval-integration.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

cd "$work_dir"
"$app_path" train \
  --input "$repo_root/test.txt" \
  --validation "$repo_root/test.txt" \
  --model model.pth \
  --vocab-size 260 \
  --embedding-dim 8 \
  --num-heads 2 \
  --num-layers 1 \
  --max-seq-len 16 \
  --train-window 4 \
  --epochs 1 \
  --seed 42 > train.log

grep -q "Cross-entropy:" train.log
grep -q "Perplexity:" train.log
grep -q "validation_fingerprint_fnv1a" config.json
grep -q "validation_cross_entropy" config.json
manifest="$(find checkpoints -maxdepth 1 -name '*.manifest' | head -n 1)"
test -n "$manifest"
grep -q 'explicit_options = .*--input' "$manifest"
grep -q 'gradient_accumulation_steps = 1' "$manifest"
grep -q 'tokenizer_mode = special-v1' "$manifest"
grep -q 'bos_token_id = 258' "$manifest"
grep -q 'eos_token_id = 259' "$manifest"
grep -q 'tokenizer_has_special_tokens = 1' config.json
test ! -w "$manifest"

# The deployable artifact is self-contained: evaluation must not depend on
# the compatibility tokenizer sidecar emitted during training.
test "$(dd if=model.pth bs=1 count=8 2>/dev/null)" = "DRNZBNDL"
rm model.pth.tokenizer

hash_before="$(sha256sum model.pth)"
"$app_path" eval \
  --model model.pth \
  --input "$repo_root/test.txt" \
  --eval-window 4 > eval.log
hash_after="$(sha256sum model.pth)"

test "$hash_before" = "$hash_after"
grep -q "Tokens: 137 (136 next-token predictions)" eval.log
grep -q "Cross-entropy:" eval.log
grep -q "Perplexity:" eval.log
grep -q "Versioned model bundle loaded" eval.log
grep -q "Model and optimizer state were not updated" eval.log

if "$app_path" eval \
    --model model.pth \
    --tokenizer overridden.tokenizer \
    --input "$repo_root/test.txt" > tokenizer-override.log 2>&1; then
  echo "bundle unexpectedly accepted a tokenizer override" >&2
  exit 1
fi
grep -q "embeds its frozen tokenizer" tokenizer-override.log

if "$app_path" eval --model model.pth > missing-input.log 2>&1; then
  echo "eval unexpectedly accepted a missing explicit --input" >&2
  exit 1
fi
grep -q "requires an explicit --input" missing-input.log

echo "EVALUATION CLI INTEGRATION CHECK PASSED"
