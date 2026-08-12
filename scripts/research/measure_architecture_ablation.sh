#!/usr/bin/env bash
# One-change-at-a-time held-out quality comparison against a completed seed floor.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ablation_recipe="$repo_root/experiments/architecture-ablation-small/recipe.env"

usage() {
    cat <<'EOF'
Usage:
  measure_architecture_ablation.sh --seed-floor DIR
      --architecture tied|rope|rmsnorm|gelu|swiglu [--output DIR]
      [--corpus FILE] [--app FILE --analyzer FILE]

The seed-floor artifact supplies the exact baseline seeds, losses, tokenizer,
corpus manifest, training recipe, noise floor, and reviewed app hash.
EOF
}

kv_get() {
    sed -n "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*\(.*[^[:space:]]\)[[:space:]]*$/\1/p" \
        "$1" | head -n 1
}

sha256_file() { sha256sum "$1" | cut -d' ' -f1; }

require_file() {
    if [ ! -f "$1" ]; then
        echo "error: not a regular file: $1" >&2
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

require_close() {
    require_decimal "$1 expected" "$2"
    require_decimal "$1 observed" "$3"
    if ! awk -v want="$2" -v got="$3" -v tolerance="$4" \
        'BEGIN { d=got-want; if (d < 0) d=-d; exit !(d <= tolerance) }'; then
        echo "error: $1 differs: config=$2 replay=$3 tolerance=$4" >&2
        exit 1
    fi
}

seed_floor=""
architecture=""
output=""
corpus=""
app=""
analyzer=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --seed-floor) seed_floor="${2:-}"; shift 2 ;;
        --architecture) architecture="${2:-}"; shift 2 ;;
        --output) output="${2:-}"; shift 2 ;;
        --corpus) corpus="${2:-}"; shift 2 ;;
        --app) app="${2:-}"; shift 2 ;;
        --analyzer) analyzer="${2:-}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "error: unknown option '$1'" >&2; usage; exit 2 ;;
    esac
done

if [ -z "$seed_floor" ] || [ -z "$architecture" ]; then
    usage
    exit 2
fi
case "$architecture" in
    tied) feature_flag=--tie-embeddings ;;
    rope) feature_flag=--rope ;;
    rmsnorm) feature_flag=--rmsnorm ;;
    gelu) feature_flag=--gelu ;;
    swiglu) feature_flag=--swiglu ;;
    *) echo "error: unsupported architecture '$architecture'" >&2; exit 2 ;;
esac
if { [ -n "$app" ] && [ -z "$analyzer" ]; } ||
   { [ -z "$app" ] && [ -n "$analyzer" ]; }; then
    echo "error: --app and --analyzer must be supplied together" >&2
    exit 2
fi

seed_floor="$(realpath "$seed_floor")"
baseline_recipe="$seed_floor/recipe.env"
baseline_manifest="$seed_floor/experiment.manifest"
floor_manifest="$seed_floor/seed-floor.manifest"
baseline_samples="$seed_floor/seed-losses.csv"
baseline_tokenizer="$seed_floor/shared.tokenizer"
source_manifest="$seed_floor/corpus.manifest"
for required in "$ablation_recipe" "$baseline_recipe" "$baseline_manifest" \
                "$floor_manifest" "$baseline_samples" "$baseline_tokenizer" \
                "$source_manifest"; do
    require_file "$required"
done
if [ "$(kv_get "$floor_manifest" status)" != ready ]; then
    echo "error: seed-floor status is not ready" >&2
    exit 2
fi
if [ "$(kv_get "$baseline_manifest" experiment_id)" != \
     "$(kv_get "$ablation_recipe" baseline_experiment_id)" ]; then
    echo "error: seed-floor experiment identity does not match ablation recipe" >&2
    exit 2
fi
noise_floor="$(kv_get "$floor_manifest" noise_floor)"
require_decimal noise_floor "$noise_floor"
sample_count="$(kv_get "$floor_manifest" sample_count)"
observed_rows="$(awk 'NR > 1 { count++ } END { print count + 0 }' "$baseline_samples")"
if [ "$observed_rows" != "$sample_count" ] || [ "$sample_count" -lt 2 ]; then
    echo "error: baseline sample count does not match its floor manifest" >&2
    exit 2
fi

if [ -z "$output" ]; then
    output="$repo_root/dist/research/architecture-$architecture-small-v1"
fi
output="$(realpath -m "$output")"
if [ -e "$output" ]; then
    echo "error: refusing to overwrite existing output: $output" >&2
    exit 2
fi

work="$(mktemp -d /tmp/dranzer-architecture-ablation.XXXXXX)"
trap 'rm -rf "$work"' EXIT
stage="$work/result"
mkdir -p "$stage/models" "$stage/runs"

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

compiler="$(kv_get "$baseline_recipe" compiler)"
if [ -z "$app" ]; then
    make -C "$repo_root/src" clean all architecture-compare-tool \
        CC="$compiler" DEBUG= ASAN= UBSAN= OMP=
    app="$repo_root/src/app.out"
    analyzer="$repo_root/src/architecture_compare.out"
else
    app="$(realpath "$app")"
    analyzer="$(realpath "$analyzer")"
fi
require_file "$app"
require_file "$analyzer"
expected_app_hash="$(kv_get "$baseline_manifest" app_sha256)"
if [ "$(sha256_file "$app")" != "$expected_app_hash" ]; then
    echo "error: feature arm must use the exact app binary recorded by the baseline" >&2
    exit 2
fi

shared_tokenizer="$work/shared.tokenizer"
cp "$baseline_tokenizer" "$shared_tokenizer"
pairs="$stage/paired-losses.csv"
printf '%s\n' 'seed,baseline_validation_cross_entropy,feature_validation_cross_entropy,baseline_model_sha256,feature_model_sha256' > "$pairs"

while IFS=, read -r seed baseline_loss baseline_hash; do
    [ "$seed" = seed ] && continue
    run="$work/run-$seed"
    mkdir -p "$run/checkpoints"
    model="$run/model.pth"
    echo "$architecture seed $seed" >&2
    (cd "$run" && "$app" train \
        --input "$work/split/train.txt" \
        --validation "$work/split/validation.txt" \
        --model "$model" \
        --tokenizer "$shared_tokenizer" \
        --checkpoint-dir "$run/checkpoints" \
        --checkpoint-interval 0 \
        --keep-checkpoints 1 \
        --seed "$seed" \
        --vocab-size "$(kv_get "$baseline_recipe" vocab_size)" \
        --embedding-dim "$(kv_get "$baseline_recipe" embedding_dim)" \
        --num-heads "$(kv_get "$baseline_recipe" num_heads)" \
        --num-layers "$(kv_get "$baseline_recipe" num_layers)" \
        --max-seq-len "$(kv_get "$baseline_recipe" max_seq_len)" \
        --train-window "$(kv_get "$baseline_recipe" train_window)" \
        --train-stride "$(kv_get "$baseline_recipe" train_stride)" \
        --epochs "$(kv_get "$baseline_recipe" epochs)" \
        --batch-size "$(kv_get "$baseline_recipe" batch_size)" \
        --gradient-accumulation "$(kv_get "$baseline_recipe" gradient_accumulation)" \
        --learning-rate "$(kv_get "$baseline_recipe" learning_rate)" \
        --optimizer "$(kv_get "$baseline_recipe" optimizer)" \
        --dropout "$(kv_get "$baseline_recipe" dropout)" \
        --grad-clip "$(kv_get "$baseline_recipe" grad_clip)" \
        --weight-decay "$(kv_get "$baseline_recipe" weight_decay)" \
        --warmup-steps "$(kv_get "$baseline_recipe" warmup_steps)" \
        --total-steps "$(kv_get "$baseline_recipe" total_steps)" \
        "$feature_flag") > "$run/training.txt"

    config="$run/config.json"
    require_file "$model"
    require_file "$config"
    feature_loss="$(kv_get "$config" validation_cross_entropy)"
    require_decimal "seed $seed feature held-out loss" "$feature_loss"
    (cd "$run" && "$app" eval --model "$model" \
        --input "$work/split/validation.txt" \
        --eval-window "$(kv_get "$baseline_recipe" eval_window)") \
        > "$run/evaluation.txt"
    replay_loss="$(awk '/- Cross-entropy:/ { value=$3 } END { print value }' \
        "$run/evaluation.txt")"
    require_close "seed $seed feature held-out replay" "$feature_loss" \
        "$replay_loss" 0.000001
    feature_hash="$(sha256_file "$model")"
    printf '%s,%s,%s,%s,%s\n' "$seed" "$baseline_loss" "$feature_loss" \
        "$baseline_hash" "$feature_hash" >> "$pairs"
    cp "$model" "$stage/models/seed-$seed.pth"
    sed "s|$work|\$WORKDIR|g" "$config" > "$stage/runs/seed-$seed.config"
    sed "s|$work|\$WORKDIR|g" "$run/training.txt" \
        > "$stage/runs/seed-$seed.training.txt"
    sed "s|$work|\$WORKDIR|g" "$run/evaluation.txt" \
        > "$stage/runs/seed-$seed.evaluation.txt"
done < "$baseline_samples"

"$analyzer" "$pairs" --noise-floor "$noise_floor" \
    --resamples "$(kv_get "$ablation_recipe" bootstrap_resamples)" \
    --bootstrap-seed "$(kv_get "$ablation_recipe" bootstrap_seed)" \
    --output "$stage/comparison.manifest"
cp "$ablation_recipe" "$stage/recipe.env"
cp "$source_manifest" "$stage/corpus.manifest"
cp "$baseline_tokenizer" "$stage/shared.tokenizer"
(cd "$stage" && find models runs -type f -print0 | sort -z | xargs -0 sha256sum) \
    > "$stage/run-artifacts.sha256"

cat > "$stage/experiment.manifest" <<EOF
format_version = 1
experiment_id = $(kv_get "$ablation_recipe" experiment_id)
architecture = $architecture
feature_flag = $feature_flag
baseline_experiment = $seed_floor
baseline_experiment_sha256 = $(sha256_file "$baseline_manifest")
baseline_source_revision = $(kv_get "$baseline_manifest" source_revision)
project_version = $(tr -d '[:space:]' < "$repo_root/VERSION")
compiler = $($compiler --version | sed -n '1p')
host_os = $(uname -srm)
cpu = $(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p; T; q' /proc/cpuinfo)
app_sha256 = $(sha256_file "$app")
analyzer_sha256 = $(sha256_file "$analyzer")
noise_floor = $noise_floor
pair_count = $sample_count
pairs_file = paired-losses.csv
pairs_sha256 = $(sha256_file "$pairs")
comparison_file = comparison.manifest
comparison_sha256 = $(sha256_file "$stage/comparison.manifest")
run_artifact_hashes_file = run-artifacts.sha256
run_artifact_hashes_sha256 = $(sha256_file "$stage/run-artifacts.sha256")
EOF

mkdir -p "$(dirname "$output")"
mv "$stage" "$output"
echo "architecture ablation result written to $output"
