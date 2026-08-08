#!/usr/bin/env bash
#
# The simplest reproducibility claim there is: the same binary, the same seed, the
# same corpus, run twice from scratch, produces byte-identical artifacts.
#
# Exact resume is already tested (tests/cli/test_checkpoint_resume.c and
# tests/integration/test_resume_cli.sh), which is a stronger property in one
# direction - it requires the optimizer, scheduler, and RNG position to survive a
# round trip through a file - and no test at all of this one. Resume compares a
# run against its own continuation. Nothing compared two independent runs.
#
# That is the property a reader checks first, and it is the one that catches a
# whole class of defect the resume test cannot: anything that leaks
# non-determinism into a fresh run. Uninitialized memory that happens to be
# consistent within a process. A hash iteration order that depends on allocator
# addresses. A time-seeded fallback. A thread schedule that reaches a reduction.
# All of those pass a resume test and fail this one.
#
# Three artifacts are compared, because they fail differently:
#   - the model weights, which is the result;
#   - the tokenizer sidecar, since a vocabulary that differs run to run makes
#     every token id mean something else;
#   - an intermediate checkpoint, which catches a divergence that later training
#     happens to wash out.
set -uo pipefail

app_path="${1:-./app.out}"
app_path="$(cd "$(dirname "$app_path")" && pwd)/$(basename "$app_path")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
work_dir="$(mktemp -d /tmp/dranzer-determinism.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

if [ ! -x "$app_path" ]; then
    echo "DETERMINISM CHECK SKIPPED: $app_path is not executable" >&2
    exit 0
fi

train_into() { # $1 = subdirectory
    mkdir -p "$work_dir/$1" && cd "$work_dir/$1" || return 1
    "$app_path" train \
        --input "$repo_root/test.txt" \
        --validation "$repo_root/test.txt" \
        --model model.pth \
        --vocab-size 280 \
        --embedding-dim 16 \
        --num-heads 2 \
        --num-layers 2 \
        --max-seq-len 16 \
        --train-window 8 \
        --epochs 3 \
        --learning-rate 0.002 \
        --batch-size 4 \
        --gradient-accumulation 2 \
        --shuffle \
        --dropout 0.2 \
        --warmup-steps 3 \
        --total-steps 100 \
        --checkpoint-interval 2 \
        --keep-checkpoints 100 \
        --seed 1234 > train.log 2>&1
}

# Dropout is deliberately on and --shuffle is deliberately set. Both consume
# randomness, so a run with them off would be trivially reproducible and would
# test almost nothing - the interesting failure is a stream that is seeded per
# process rather than per run.
if ! train_into first || ! train_into second; then
    echo "DETERMINISM CHECK FAILED: a training run did not complete" >&2
    tail -n 20 "$work_dir/first/train.log" "$work_dir/second/train.log" 2>/dev/null >&2
    exit 1
fi

failed=0
compare() { # $1 = label, $2 = relative path
    local a="$work_dir/first/$2" b="$work_dir/second/$2"
    if [ ! -f "$a" ] || [ ! -f "$b" ]; then
        echo "FAIL $1: $2 missing from one or both runs"
        failed=1
        return
    fi
    local ha hb
    ha="$(sha256sum "$a" | cut -d' ' -f1)"
    hb="$(sha256sum "$b" | cut -d' ' -f1)"
    if [ "$ha" = "$hb" ]; then
        printf '  %-24s %s  identical (%s bytes)\n' "$1" "${ha:0:16}" \
            "$(wc -c < "$a" | tr -d ' ')"
    else
        echo "FAIL $1: $2 differs between two runs of the same seed"
        echo "     first  $ha"
        echo "     second $hb"
        cmp "$a" "$b" | head -3
        failed=1
    fi
}

echo "two independent training runs, --seed 1234, dropout 0.2, shuffle on:"
compare "model weights" model.pth
compare "tokenizer sidecar" model.pth.tokenizer

# Every checkpoint, not just the last: a divergence at step 2 that later training
# pulls back together would otherwise go unseen.
checkpoints=0
for checkpoint in "$work_dir/first/checkpoints"/*.ckpt; do
    [ -f "$checkpoint" ] || continue
    compare "checkpoint $(basename "$checkpoint" .ckpt | sed 's/checkpoint_//')" \
        "checkpoints/$(basename "$checkpoint")"
    checkpoints=$((checkpoints + 1))
done
if [ "$checkpoints" -eq 0 ]; then
    echo "FAIL: no checkpoints were written, so the intermediate state went unchecked"
    failed=1
fi

# The reported losses too. Identical weights with different logged losses would
# mean the log is not describing the run.
if ! diff -q <(grep -oE 'loss[=: ][0-9.]+' "$work_dir/first/train.log") \
             <(grep -oE 'loss[=: ][0-9.]+' "$work_dir/second/train.log") >/dev/null; then
    echo "FAIL: the two runs logged different losses"
    failed=1
else
    echo "  loss trajectory         identical across both logs"
fi

if [ "$failed" -eq 0 ]; then
    echo "DETERMINISM CHECK PASSED ($((checkpoints + 2)) artifacts byte-identical)"
else
    echo "DETERMINISM CHECK FAILED"
fi
exit $failed
