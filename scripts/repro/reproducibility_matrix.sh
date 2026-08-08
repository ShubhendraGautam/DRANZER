#!/usr/bin/env bash
#
# Regenerates the table in docs/reproducibility.md.
#
# One row per axis along which two runs can differ - rebuild, compiler,
# instruction set, threading, backend - and for each, whether the same seed still
# produces the same bytes. Every cell in that document is produced by this script;
# none of it is reasoning about what the code ought to do.
#
# Two artifacts are compared per axis, because they answer different questions:
#
#   initial weights   what `--seed N` names before any arithmetic on the data.
#                     Should be identical everywhere: the generator is integer
#                     arithmetic (core/rng.h) and the conversion to float is one
#                     expression, so nothing here depends on the toolchain.
#
#   trained weights   the same after a short fixed training run. This is where
#                     kernel selection, threading, and compiler reassociation can
#                     legitimately change the last bits, so a difference here is
#                     a fact about float arithmetic rather than a defect - which
#                     is exactly why the two are reported separately.
#
# Usage: scripts/repro/reproducibility_matrix.sh [output-file]
#        Writes a markdown table to stdout, or to the given file.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
src_dir="$repo_root/src"
work_dir="$(mktemp -d /tmp/dranzer-repro-matrix.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

output="${1:-/dev/stdout}"

# --- the fixed experiment ----------------------------------------------------
# Small enough to run every variant in a few minutes, large enough that the
# forward pass exercises attention, the FFN, layer norm, dropout, and more than
# one optimizer step. A single-step run would hide a divergence that only appears
# once Adam's moments are non-zero.
FINGERPRINT_ARGS=(--seed 4242 --vocab 260 --embedding-dim 16 --heads 2
                  --layers 2 --max-seq-len 24 --quiet)

train_args() { # $1 = model path
    printf '%s\n' train \
        --input "$repo_root/test.txt" --validation "$repo_root/test.txt" \
        --model "$1" --vocab-size 260 --embedding-dim 16 --num-heads 2 \
        --num-layers 2 --max-seq-len 24 --train-window 12 --epochs 4 \
        --learning-rate 0.002 --batch-size 4 --gradient-accumulation 2 \
        --dropout 0.2 --shuffle --warmup-steps 3 --total-steps 200 --seed 4242
}

# --- one variant -------------------------------------------------------------
# Builds the tree with the given compiler and flags into its own directory, then
# reports the two hashes. Everything is rebuilt from clean: reusing objects across
# variants is how a "different compiler" row ends up measuring one compiler.
measure_variant() { # $1 = label, $2 = CC, $3 = extra make args, $4 = env prefix
    local label="$1" cc="$2" make_args="$3" env_spec="$4"
    local variant_dir="$work_dir/$(echo "$label" | tr -c 'a-zA-Z0-9' '_')"
    mkdir -p "$variant_dir"

    cd "$src_dir" || return 1
    make clean >/dev/null 2>&1
    if ! make all fingerprint -j2 CC="$cc" $make_args >"$variant_dir/build.log" 2>&1; then
        echo "$label|BUILD FAILED|BUILD FAILED"
        return 0
    fi
    cp "$src_dir/app.out" "$variant_dir/app.out"
    cp "$src_dir/fingerprint_model.out" "$variant_dir/fingerprint.out"

    local initial trained
    initial=$(cd "$variant_dir" && env $env_spec ./fingerprint.out "${FINGERPRINT_ARGS[@]}" 2>/dev/null)
    (cd "$variant_dir" && mapfile -t args < <(train_args model.pth) &&
        env $env_spec ./app.out "${args[@]}" >train.log 2>&1)
    if [ -f "$variant_dir/model.pth" ]; then
        trained=$(sha256sum "$variant_dir/model.pth" | cut -c1-16)
    else
        trained="RUN FAILED"
    fi
    echo "$label|${initial:-ERROR}|$trained"
}

echo "measuring; each variant is a full clean rebuild" >&2

results=()
# The reference: the default build, twice, in two separate clean rebuilds. The
# second is the "rebuild with the same compiler and flags" axis, and it has to be
# a genuinely separate build for that to mean anything.
results+=("$(measure_variant 'clang -O3 -ffast-math (reference)' clang '' '')")
results+=("$(measure_variant 'clang, rebuilt from clean' clang '' '')")
results+=("$(measure_variant 'gcc -O3 -ffast-math' gcc '' '')")
results+=("$(measure_variant 'gcc, OpenMP (2 threads)' gcc 'OMP=1' 'OMP_NUM_THREADS=2')")
results+=("$(measure_variant 'clang, size-optimized (-Os)' clang 'SIZE=1' '')")
# Runtime ISA caps: same binary, different kernels selected at dispatch.
results+=("$(measure_variant 'clang, DRANZER_CPU_ISA=baseline' clang '' 'DRANZER_CPU_ISA=baseline')")
results+=("$(measure_variant 'clang, DRANZER_CPU_ISA=avx2' clang '' 'DRANZER_CPU_ISA=avx2')")

reference_initial=$(echo "${results[0]}" | cut -d'|' -f2)
reference_trained=$(echo "${results[0]}" | cut -d'|' -f3)

{
    echo "| build / runtime | initial weights | matches | trained weights | matches |"
    echo "|---|---|:-:|---|:-:|"
    for row in "${results[@]}"; do
        IFS='|' read -r label initial trained <<< "$row"
        init_match="no"; [ "$initial" = "$reference_initial" ] && init_match="yes"
        train_match="no"; [ "$trained" = "$reference_trained" ] && train_match="yes"
        printf '| %s | `%s` | %s | `%s` | %s |\n' \
            "$label" "$initial" "$init_match" "$trained" "$train_match"
    done
    echo ""
    echo "Reference: initial \`$reference_initial\`, trained \`$reference_trained\`."
    echo "Generated by \`scripts/repro/reproducibility_matrix.sh\` on"
    echo "$(uname -sr), $(clang --version | head -1), $(gcc --version | head -1)."
} > "$output"

cd "$src_dir" && make clean >/dev/null 2>&1 && make -j2 >/dev/null 2>&1
echo "done" >&2
