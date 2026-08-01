#!/usr/bin/env bash
set -euo pipefail

app_path="${1:-./app.out}"
app_path="$(cd "$(dirname "$app_path")" && pwd)/$(basename "$app_path")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
work_dir="$(mktemp -d /tmp/dranzer-resume-integration.XXXXXX)"
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
  --epochs 2 \
  --learning-rate 0.002 \
  --batch-size 4 \
  --gradient-accumulation 2 \
  --shuffle \
  --dropout 0.2 \
  --warmup-steps 5 \
  --total-steps 300 \
  --checkpoint-interval 3 \
  --keep-checkpoints 100 \
  --seed 42 > uninterrupted.log

mid_checkpoint="checkpoints/checkpoint_epoch_0_step_3.ckpt"
test -f "$mid_checkpoint"
final_checkpoint="$(sed -n 's/^.*Final checkpoint saved: //p' uninterrupted.log | tail -n 1)"
test -n "$final_checkpoint"
test -f "$final_checkpoint"

model_hash="$(sha256sum model.pth | awk '{print $1}')"
checkpoint_hash="$(sha256sum "$final_checkpoint" | awk '{print $1}')"
cp "$final_checkpoint" uninterrupted-final.ckpt

if "$app_path" train --resume "$mid_checkpoint" --epochs 2 \
    --batch-size 5 > conflicting-resume.log 2>&1; then
  echo "resume unexpectedly accepted a trajectory-changing override" >&2
  exit 1
fi
grep -q "cannot change an exact-resume trajectory" conflicting-resume.log

# Resume must be self-contained: neither the model artifact nor tokenizer
# sidecar is needed because the checkpoint embeds both tokenizer and model
# training state.
rm -f model.pth model.pth.tokenizer
"$app_path" train --resume "$mid_checkpoint" --epochs 2 > resumed.log

test -f model.pth
test -f model.pth.tokenizer
test "$model_hash" = "$(sha256sum model.pth | awk '{print $1}')"
test "$checkpoint_hash" = "$(sha256sum "$final_checkpoint" | awk '{print $1}')"
cmp -s uninterrupted-final.ckpt "$final_checkpoint"
grep -Eq "Resumed .* at epoch 1 after 24 predictions" resumed.log
grep -q "Corpus tokens streamed:" resumed.log
grep -q "Total optimizer steps: 34" resumed.log

# `latest` resolves the terminal checkpoint and is a valid no-op resume. It
# still restores the tokenizer sidecar and validation provenance.
rm -f model.pth.tokenizer
"$app_path" train --resume latest --checkpoint-dir checkpoints > latest.log
test -f model.pth.tokenizer
test "$model_hash" = "$(sha256sum model.pth | awk '{print $1}')"
grep -Eq "Resumed .* at epoch 3 after 0 predictions" latest.log
grep -q "Cross-entropy:" latest.log

echo "EXACT RESUME CLI INTEGRATION CHECK PASSED"
