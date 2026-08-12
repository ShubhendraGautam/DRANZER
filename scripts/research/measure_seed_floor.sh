#!/usr/bin/env bash
# Pre-registered adaptive seed sweep for held-out cross-entropy.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
recipe="$repo_root/experiments/seed-floor-small/recipe.env"
default_output="$repo_root/dist/research/seed-floor-small-v1"

usage() {
    cat <<'EOF'
Usage:
  measure_seed_floor.sh [--output DIR] [--corpus FILE]
                        [--app FILE --analyzer FILE]

Without --corpus, download the exact source_url in the corpus manifest.
Without --app/--analyzer, perform one clean recipe-selected compiler build.
The two reviewed-binary options must be supplied together.
EOF
}

kv_get() { # $1 file, $2 key
    sed -n "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*\(.*[^[:space:]]\)[[:space:]]*$/\1/p" \
        "$1" | head -n 1
}

recipe_get() {
    kv_get "$recipe" "$1"
}

sha256_file() {
    sha256sum "$1" | cut -d' ' -f1
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "error: not a regular file: $1" >&2
        exit 2
    fi
}

require_uint() {
    if ! awk -v value="$2" 'BEGIN { exit !(value ~ /^(0|[1-9][0-9]*)$/) }'; then
        echo "error: $1 is not a canonical unsigned integer: '$2'" >&2
        exit 2
    fi
}

require_decimal() {
    if ! awk -v value="$2" \
        'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/) }'; then
        echo "error: $1 is not a non-negative decimal: '$2'" >&2
        exit 2
    fi
}

require_close() { # $1 label, $2 expected, $3 observed, $4 tolerance
    require_decimal "$1 expected" "$2"
    require_decimal "$1 observed" "$3"
    if ! awk -v want="$2" -v got="$3" -v tolerance="$4" \
        'BEGIN { difference=got-want; if (difference < 0) difference=-difference;
                 exit !(difference <= tolerance) }'; then
        echo "error: $1 differs: config=$2 replay=$3 tolerance=$4" >&2
        exit 1
    fi
}

validate_recipe() {
    local keys key value
    keys="experiment_version experiment_id corpus_manifest compiler first_seed minimum_samples maximum_samples target_precision_ratio bootstrap_resamples bootstrap_seed vocab_size embedding_dim num_heads num_layers max_seq_len train_window train_stride epochs batch_size gradient_accumulation learning_rate optimizer dropout grad_clip weight_decay warmup_steps total_steps eval_window"
    for key in $keys; do
        value="$(recipe_get "$key")"
        if [ -z "$value" ]; then
            echo "error: recipe is missing '$key'" >&2
            exit 2
        fi
    done
    require_uint first_seed "$(recipe_get first_seed)"
    require_uint minimum_samples "$(recipe_get minimum_samples)"
    require_uint maximum_samples "$(recipe_get maximum_samples)"
    if [ "$(recipe_get minimum_samples)" -lt 2 ] ||
       [ "$(recipe_get maximum_samples)" -lt "$(recipe_get minimum_samples)" ]; then
        echo "error: invalid adaptive sample bounds in recipe" >&2
        exit 2
    fi
    require_decimal target_precision_ratio "$(recipe_get target_precision_ratio)"
}

output="$default_output"
corpus=""
app=""
analyzer=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output) output="${2:-}"; shift 2 ;;
        --corpus) corpus="${2:-}"; shift 2 ;;
        --app) app="${2:-}"; shift 2 ;;
        --analyzer) analyzer="${2:-}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "error: unknown option '$1'" >&2; usage; exit 2 ;;
    esac
done

validate_recipe
if [ -z "$output" ]; then
    echo "error: --output requires a directory" >&2
    exit 2
fi
if { [ -n "$app" ] && [ -z "$analyzer" ]; } ||
   { [ -z "$app" ] && [ -n "$analyzer" ]; }; then
    echo "error: --app and --analyzer must be supplied together" >&2
    exit 2
fi
output="$(realpath -m "$output")"
if [ -e "$output" ]; then
    echo "error: refusing to overwrite existing output: $output" >&2
    exit 2
fi

work="$(mktemp -d /tmp/dranzer-seed-floor.XXXXXX)"
trap "rm -rf '$work'" EXIT
stage="$work/result"
mkdir -p "$stage/models" "$stage/runs"

manifest_relative="$(recipe_get corpus_manifest)"
source_manifest="$repo_root/$manifest_relative"
require_file "$source_manifest"
cp "$source_manifest" "$work/corpus.manifest"
if [ -z "$corpus" ]; then
    corpus="$work/corpus.txt"
    source_url="$(kv_get "$source_manifest" source_url)"
    echo "downloading pinned corpus from $source_url" >&2
    curl -fL --retry 3 --connect-timeout 15 --max-time 300 \
        --proto '=https' --tlsv1.2 "$source_url" -o "$corpus"
else
    corpus="$(realpath "$corpus")"
fi
require_file "$corpus"
"$repo_root/scripts/corpus.sh" verify "$source_manifest" "$corpus" --require-verified
"$repo_root/scripts/corpus.sh" split "$source_manifest" "$corpus" "$work/split"

compiler="$(recipe_get compiler)"
if [ -z "$app" ]; then
    echo "building reviewed app and analyzer with $compiler" >&2
    make -C "$repo_root/src" clean all seed-floor-tool CC="$compiler" \
        DEBUG= ASAN= UBSAN= OMP=
    app="$repo_root/src/app.out"
    analyzer="$repo_root/src/seed_floor.out"
else
    app="$(realpath "$app")"
    analyzer="$(realpath "$analyzer")"
fi
require_file "$app"
require_file "$analyzer"

samples="$stage/seed-losses.csv"
printf '%s\n' 'seed,validation_cross_entropy,model_sha256' > "$samples"
shared_tokenizer="$work/shared.tokenizer"
minimum_samples="$(recipe_get minimum_samples)"
maximum_samples="$(recipe_get maximum_samples)"
first_seed="$(recipe_get first_seed)"
status=collect_more
count=0

while [ "$count" -lt "$maximum_samples" ]; do
    seed=$((first_seed + count))
    count=$((count + 1))
    run="$work/run-$seed"
    mkdir -p "$run/checkpoints"
    model="$run/model.pth"
    echo "seed $seed ($count/$maximum_samples)" >&2

    (cd "$run" && "$app" train \
        --input "$work/split/train.txt" \
        --validation "$work/split/validation.txt" \
        --model "$model" \
        --tokenizer "$shared_tokenizer" \
        --checkpoint-dir "$run/checkpoints" \
        --checkpoint-interval 0 \
        --keep-checkpoints 1 \
        --seed "$seed" \
        --vocab-size "$(recipe_get vocab_size)" \
        --embedding-dim "$(recipe_get embedding_dim)" \
        --num-heads "$(recipe_get num_heads)" \
        --num-layers "$(recipe_get num_layers)" \
        --max-seq-len "$(recipe_get max_seq_len)" \
        --train-window "$(recipe_get train_window)" \
        --train-stride "$(recipe_get train_stride)" \
        --epochs "$(recipe_get epochs)" \
        --batch-size "$(recipe_get batch_size)" \
        --gradient-accumulation "$(recipe_get gradient_accumulation)" \
        --learning-rate "$(recipe_get learning_rate)" \
        --optimizer "$(recipe_get optimizer)" \
        --dropout "$(recipe_get dropout)" \
        --grad-clip "$(recipe_get grad_clip)" \
        --weight-decay "$(recipe_get weight_decay)" \
        --warmup-steps "$(recipe_get warmup_steps)" \
        --total-steps "$(recipe_get total_steps)") > "$run/training.txt"

    config="$run/config.json"
    require_file "$model"
    require_file "$shared_tokenizer"
    require_file "$config"
    loss="$(kv_get "$config" validation_cross_entropy)"
    require_decimal "seed $seed held-out loss" "$loss"

    (cd "$run" && "$app" eval \
        --model "$model" \
        --input "$work/split/validation.txt" \
        --eval-window "$(recipe_get eval_window)") > "$run/evaluation.txt"
    replay_loss="$(awk '/- Cross-entropy:/ { value=$3 } END { print value }' \
        "$run/evaluation.txt")"
    require_close "seed $seed held-out replay" "$loss" "$replay_loss" 0.000001

    model_hash="$(sha256_file "$model")"
    printf '%s,%s,%s\n' "$seed" "$loss" "$model_hash" >> "$samples"
    cp "$model" "$stage/models/seed-$seed.pth"
    sed "s|$work|\$WORKDIR|g" "$config" > "$stage/runs/seed-$seed.config"
    sed "s|$work|\$WORKDIR|g" "$run/training.txt" > "$stage/runs/seed-$seed.training.txt"
    sed "s|$work|\$WORKDIR|g" "$run/evaluation.txt" > "$stage/runs/seed-$seed.evaluation.txt"

    "$analyzer" "$samples" \
        --minimum "$minimum_samples" \
        --maximum "$maximum_samples" \
        --target-ratio "$(recipe_get target_precision_ratio)" \
        --resamples "$(recipe_get bootstrap_resamples)" \
        --bootstrap-seed "$(recipe_get bootstrap_seed)" \
        --output "$work/seed-floor.next"
    mv "$work/seed-floor.next" "$stage/seed-floor.manifest"
    status="$(kv_get "$stage/seed-floor.manifest" status)"
    if [ "$status" = ready ]; then
        break
    fi
done

cp "$recipe" "$stage/recipe.env"
cp "$source_manifest" "$stage/corpus.manifest"
cp "$shared_tokenizer" "$stage/shared.tokenizer"
(cd "$stage" && find models runs -type f -print0 | sort -z | xargs -0 sha256sum) \
    > "$stage/run-artifacts.sha256"
source_revision="$(git -C "$repo_root" rev-parse HEAD)"
project_version="$(tr -d '[:space:]' < "$repo_root/VERSION")"
compiler_version="$($compiler --version | sed -n '1p')"
cpu_model="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p; T; q' /proc/cpuinfo)"
host_os="$(uname -srm)"

cat > "$stage/experiment.manifest" <<EOF
format_version = 1
experiment_id = $(recipe_get experiment_id)
project_version = $project_version
source_revision = $source_revision
compiler = $compiler_version
host_os = $host_os
cpu = $cpu_model
app_sha256 = $(sha256_file "$app")
analyzer_sha256 = $(sha256_file "$analyzer")
recipe_file = recipe.env
recipe_sha256 = $(sha256_file "$stage/recipe.env")
corpus_manifest_file = corpus.manifest
corpus_manifest_sha256 = $(sha256_file "$stage/corpus.manifest")
corpus_sha256 = $(kv_get "$source_manifest" sha256)
tokenizer_file = shared.tokenizer
tokenizer_sha256 = $(sha256_file "$stage/shared.tokenizer")
samples_file = seed-losses.csv
samples_sha256 = $(sha256_file "$samples")
floor_file = seed-floor.manifest
floor_sha256 = $(sha256_file "$stage/seed-floor.manifest")
run_artifact_hashes_file = run-artifacts.sha256
run_artifact_hashes_sha256 = $(sha256_file "$stage/run-artifacts.sha256")
status = $status
EOF

mkdir -p "$(dirname "$output")"
mv "$stage" "$output"
echo "seed-floor result written to $output"
if [ "$status" != ready ]; then
    echo "error: adaptive precision target was not met (status=$status)" >&2
    exit 1
fi
