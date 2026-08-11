#!/usr/bin/env bash
# Build, train, package, and verify the small Shakespeare reference model.
# See references/shakespeare-small/README.md for the protocol and limitations.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
recipe="$repo_root/references/shakespeare-small/recipe.env"
default_package="$repo_root/dist/reference/shakespeare-small-v1"

usage() {
    cat <<'EOF'
Usage:
  publish_shakespeare_small.sh publish [--output DIR] [--corpus FILE] [--app FILE]
  publish_shakespeare_small.sh verify  [--package DIR] [--corpus FILE --app FILE]

publish downloads the pinned corpus when --corpus is omitted. It performs a
clean recipe-selected compiler build when --app is omitted. verify always
checks package hashes; with both --corpus and --app it also replays evaluation.
EOF
}

recipe_get() { # $1 = key
    sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\(.*[^[:space:]]\)[[:space:]]*$/\1/p" \
        "$recipe" | head -n 1
}

manifest_get() { # $1 = manifest, $2 = key
    sed -n "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*\(.*[^[:space:]]\)[[:space:]]*$/\1/p" \
        "$1" | head -n 1
}

config_get() { # $1 = config, $2 = key
    manifest_get "$1" "$2"
}

sha256_file() {
    sha256sum "$1" | cut -d' ' -f1
}

require_value() { # $1 = description, $2 = value
    if [ -z "$2" ]; then
        echo "error: missing $1" >&2
        exit 1
    fi
}

require_regular_file() {
    if [ ! -f "$1" ]; then
        echo "error: not a regular file: $1" >&2
        exit 2
    fi
}

require_safe_package_name() { # package filenames may not escape their directory
    case "$1" in
        ""|.|..|*/*)
            echo "error: unsafe package filename '$1'" >&2
            exit 1
            ;;
    esac
}

require_equal() { # $1 label, $2 expected, $3 actual
    if [ "$2" != "$3" ]; then
        echo "error: $1 mismatch: expected '$2', observed '$3'" >&2
        exit 1
    fi
}

require_decimal() {
    if ! awk -v value="$2" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/) }'; then
        echo "error: $1 is not a non-negative decimal: '$2'" >&2
        exit 1
    fi
}

require_uint() {
    if ! awk -v value="$2" 'BEGIN { exit !(value ~ /^(0|[1-9][0-9]*)$/) }'; then
        echo "error: $1 is not a canonical unsigned integer: '$2'" >&2
        exit 1
    fi
}

require_within() { # $1 label, $2 expected, $3 observed, $4 absolute tolerance
    require_decimal "$1 expected value" "$2"
    require_decimal "$1 observed value" "$3"
    require_decimal "$1 tolerance" "$4"
    if ! awk -v want="$2" -v got="$3" -v tolerance="$4" \
        'BEGIN { difference = got - want; if (difference < 0) difference = -difference;
                 exit !(difference <= tolerance) }'; then
        echo "error: $1 $3 is outside $2 +/- $4" >&2
        exit 1
    fi
}

corpus_manifest_path() {
    local relative
    relative="$(recipe_get corpus_manifest)"
    require_value "recipe corpus_manifest" "$relative"
    printf '%s/%s\n' "$repo_root" "$relative"
}

download_corpus() { # $1 destination
    local source manifest
    manifest="$(corpus_manifest_path)"
    source="$(manifest_get "$manifest" source_url)"
    require_value "corpus source_url" "$source"
    echo "downloading corpus from $source" >&2
    curl -fL --retry 3 --connect-timeout 15 --max-time 300 \
        --proto '=https' --tlsv1.2 "$source" -o "$1"
}

verify_corpus() { # $1 manifest, $2 corpus
    "$repo_root/scripts/corpus.sh" verify "$1" "$2" --require-verified
}

materialize_splits() { # $1 manifest, $2 corpus, $3 destination
    "$repo_root/scripts/corpus.sh" split "$1" "$2" "$3"
}

evaluation_values() { # $1 evaluation output; prints tokens predictions loss perplexity
    awk '
        /- Tokens:/ {
            tokens=$3
            predictions=$4
            gsub(/[()]/, "", predictions)
        }
        /- Cross-entropy:/ { loss=$3 }
        /- Perplexity:/ { perplexity=$3 }
        END {
            if (tokens == "" || predictions == "" || loss == "" || perplexity == "") exit 1
            print tokens, predictions, loss, perplexity
        }
    ' "$1"
}

verify_static_package() { # $1 package directory
    local package="$1" manifest="$1/reference-model.manifest"
    require_regular_file "$manifest"

    local required key value
    required="format_version reference_id project_version source_revision compiler recipe_file recipe_sha256 corpus_manifest_file corpus_manifest_sha256 corpus_sha256 model_file model_sha256 tokenizer_file tokenizer_sha256 config_file config_sha256 training_log_file training_log_sha256 evaluation_file evaluation_sha256 validation_bytes validation_split_sha256 validation_tokens validation_predictions validation_cross_entropy validation_perplexity metric_absolute_tolerance"
    for key in $required; do
        value="$(manifest_get "$manifest" "$key")"
        require_value "reference-model.manifest key '$key'" "$value"
    done

    require_equal "reference manifest format" "1" "$(manifest_get "$manifest" format_version)"
    require_equal "reference id" "$(recipe_get reference_id)" \
        "$(manifest_get "$manifest" reference_id)"

    local file_key hash_key filename expected_hash
    for file_key in recipe_file corpus_manifest_file model_file tokenizer_file config_file training_log_file evaluation_file; do
        hash_key="${file_key%_file}_sha256"
        filename="$(manifest_get "$manifest" "$file_key")"
        expected_hash="$(manifest_get "$manifest" "$hash_key")"
        require_safe_package_name "$filename"
        require_regular_file "$package/$filename"
        require_equal "$filename SHA-256" "$expected_hash" \
            "$(sha256_file "$package/$filename")"
    done

    require_equal "packaged recipe" "$(sha256_file "$recipe")" \
        "$(manifest_get "$manifest" recipe_sha256)"

    local packaged_corpus_manifest corpus_bytes train_bytes validation_bytes
    packaged_corpus_manifest="$package/$(manifest_get "$manifest" corpus_manifest_file)"
    require_equal "corpus SHA-256" "$(manifest_get "$manifest" corpus_sha256)" \
        "$(manifest_get "$packaged_corpus_manifest" sha256)"
    require_equal "validation split SHA-256" \
        "$(manifest_get "$manifest" validation_split_sha256)" \
        "$(manifest_get "$packaged_corpus_manifest" split_validation_sha256)"
    corpus_bytes="$(manifest_get "$packaged_corpus_manifest" bytes)"
    train_bytes="$(manifest_get "$packaged_corpus_manifest" split_train_bytes)"
    validation_bytes="$(manifest_get "$manifest" validation_bytes)"
    require_uint "corpus byte count" "$corpus_bytes"
    require_uint "training split byte count" "$train_bytes"
    require_uint "validation byte count" "$validation_bytes"
    if [ "$train_bytes" -gt "$corpus_bytes" ]; then
        echo "error: training split exceeds corpus byte count" >&2
        exit 1
    fi
    require_equal "validation byte count" \
        "$((corpus_bytes - train_bytes))" "$validation_bytes"

    echo "reference package hashes verified: $package"
}

replay_evaluation() { # $1 package, $2 corpus, $3 app, $4 scratch directory
    local package="$1" corpus="$2" app="$3" scratch="$4"
    local reference_manifest="$package/reference-model.manifest"
    local packaged_corpus_manifest
    packaged_corpus_manifest="$package/$(manifest_get "$reference_manifest" corpus_manifest_file)"

    require_regular_file "$corpus"
    require_regular_file "$app"
    verify_corpus "$packaged_corpus_manifest" "$corpus"
    materialize_splits "$packaged_corpus_manifest" "$corpus" "$scratch/split"

    local model_file eval_window replay_output
    model_file="$(manifest_get "$reference_manifest" model_file)"
    eval_window="$(recipe_get eval_window)"
    replay_output="$scratch/replay-evaluation.txt"
    (cd "$scratch" && "$app" eval \
        --model "$package/$model_file" \
        --input "$scratch/split/validation.txt" \
        --eval-window "$eval_window") > "$replay_output"

    local observed_tokens observed_predictions observed_loss observed_perplexity
    read -r observed_tokens observed_predictions observed_loss observed_perplexity \
        < <(evaluation_values "$replay_output")

    local expected_tokens expected_predictions expected_loss expected_perplexity tolerance
    expected_tokens="$(manifest_get "$reference_manifest" validation_tokens)"
    expected_predictions="$(manifest_get "$reference_manifest" validation_predictions)"
    expected_loss="$(manifest_get "$reference_manifest" validation_cross_entropy)"
    expected_perplexity="$(manifest_get "$reference_manifest" validation_perplexity)"
    tolerance="$(manifest_get "$reference_manifest" metric_absolute_tolerance)"

    require_equal "validation token count" "$expected_tokens" "$observed_tokens"
    require_equal "validation prediction count" "$expected_predictions" "$observed_predictions"
    require_within "validation cross-entropy" "$expected_loss" "$observed_loss" "$tolerance"
    require_within "validation perplexity" "$expected_perplexity" "$observed_perplexity" "$tolerance"
    echo "reference evaluation replay verified"
}

publish() {
    local output="$default_package" corpus="" app=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --output) output="${2:-}"; shift 2 ;;
            --corpus) corpus="${2:-}"; shift 2 ;;
            --app) app="${2:-}"; shift 2 ;;
            --help|-h) usage; exit 0 ;;
            *) echo "error: unknown publish option '$1'" >&2; usage; exit 2 ;;
        esac
    done
    require_value "--output value" "$output"
    output="$(realpath -m "$output")"
    if [ -e "$output" ]; then
        echo "error: output already exists; refusing to overwrite: $output" >&2
        exit 2
    fi

    local work
    work="$(mktemp -d /tmp/dranzer-reference.XXXXXX)"
    trap "rm -rf '$work'" EXIT
    mkdir -p "$work/run" "$work/package"

    local source_manifest="$work/source-corpus.manifest"
    cp "$(corpus_manifest_path)" "$source_manifest"
    if [ -z "$corpus" ]; then
        corpus="$work/corpus.txt"
        download_corpus "$corpus"
    else
        corpus="$(realpath "$corpus")"
    fi
    verify_corpus "$source_manifest" "$corpus"
    materialize_splits "$source_manifest" "$corpus" "$work/split"

    local compiler
    compiler="$(recipe_get compiler)"
    if [ -z "$app" ]; then
        echo "building reviewed source with $compiler" >&2
        make -C "$repo_root/src" clean all CC="$compiler"
        app="$repo_root/src/app.out"
    else
        app="$(realpath "$app")"
    fi
    require_regular_file "$app"

    local reference_id model tokenizer checkpoint_dir
    reference_id="$(recipe_get reference_id)"
    model="$work/run/$reference_id.pth"
    tokenizer="$work/run/$reference_id.tokenizer"
    checkpoint_dir="$work/run/checkpoints"

    echo "training $reference_id" >&2
    (cd "$work/run" && "$app" train \
        --input "$work/split/train.txt" \
        --validation "$work/split/validation.txt" \
        --model "$model" \
        --tokenizer "$tokenizer" \
        --checkpoint-dir "$checkpoint_dir" \
        --checkpoint-interval 0 \
        --keep-checkpoints 1 \
        --seed "$(recipe_get seed)" \
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
        --total-steps "$(recipe_get total_steps)") > "$work/train.txt"

    echo "re-evaluating saved bundle" >&2
    (cd "$work/run" && "$app" eval \
        --model "$model" \
        --input "$work/split/validation.txt" \
        --eval-window "$(recipe_get eval_window)") > "$work/evaluation.txt"

    local config="$work/run/config.json"
    require_regular_file "$model"
    require_regular_file "$tokenizer"
    require_regular_file "$config"

    local validation_tokens validation_predictions validation_loss validation_perplexity
    read -r validation_tokens validation_predictions validation_loss validation_perplexity \
        < <(evaluation_values "$work/evaluation.txt")
    local tolerance
    tolerance="$(recipe_get metric_absolute_tolerance)"
    require_equal "saved validation token count" \
        "$(config_get "$config" validation_tokens)" "$validation_tokens"
    require_within "saved validation cross-entropy" \
        "$(config_get "$config" validation_cross_entropy)" "$validation_loss" "$tolerance"
    require_within "saved validation perplexity" \
        "$(config_get "$config" validation_perplexity)" "$validation_perplexity" "$tolerance"

    local package="$work/package"
    local model_name="$reference_id.pth"
    local tokenizer_name="$reference_id.tokenizer"
    cp "$model" "$package/$model_name"
    cp "$tokenizer" "$package/$tokenizer_name"
    cp "$recipe" "$package/recipe.env"
    cp "$source_manifest" "$package/corpus.manifest"
    # Paths contain the random scratch directory and are not trajectory inputs.
    # Normalize only those paths so the resolved configuration remains useful.
    sed "s|$work|\$WORKDIR|g" "$config" > "$package/training-config.txt"
    sed "s|$work|\$WORKDIR|g" "$work/evaluation.txt" > "$package/evaluation.txt"
    sed "s|$work|\$WORKDIR|g" "$work/train.txt" > "$package/training.txt"

    local source_revision project_version compiler_version corpus_bytes train_bytes
    source_revision="$(git -C "$repo_root" rev-parse HEAD)"
    project_version="$(tr -d '[:space:]' < "$repo_root/VERSION")"
    compiler_version="$($compiler --version | head -n 1)"
    corpus_bytes="$(manifest_get "$source_manifest" bytes)"
    train_bytes="$(manifest_get "$source_manifest" split_train_bytes)"
    require_uint "corpus byte count" "$corpus_bytes"
    require_uint "training split byte count" "$train_bytes"

    cat > "$package/reference-model.manifest" <<EOF
# Generated by scripts/reference/publish_shakespeare_small.sh.
# Values are expected replay results for the exact bundle named below.
format_version = 1
reference_id = $reference_id
project_version = $project_version
source_revision = $source_revision
compiler = $compiler_version
recipe_file = recipe.env
recipe_sha256 = $(sha256_file "$package/recipe.env")
corpus_manifest_file = corpus.manifest
corpus_manifest_sha256 = $(sha256_file "$package/corpus.manifest")
corpus_sha256 = $(manifest_get "$source_manifest" sha256)
model_file = $model_name
model_sha256 = $(sha256_file "$package/$model_name")
tokenizer_file = $tokenizer_name
tokenizer_sha256 = $(sha256_file "$package/$tokenizer_name")
config_file = training-config.txt
config_sha256 = $(sha256_file "$package/training-config.txt")
evaluation_file = evaluation.txt
evaluation_sha256 = $(sha256_file "$package/evaluation.txt")
training_log_file = training.txt
training_log_sha256 = $(sha256_file "$package/training.txt")
validation_bytes = $((corpus_bytes - train_bytes))
validation_split_sha256 = $(manifest_get "$source_manifest" split_validation_sha256)
validation_tokens = $validation_tokens
validation_predictions = $validation_predictions
validation_cross_entropy = $validation_loss
validation_perplexity = $validation_perplexity
metric_absolute_tolerance = $tolerance
EOF

    verify_static_package "$package"
    replay_evaluation "$package" "$corpus" "$app" "$work/replay"

    mkdir -p "$(dirname "$output")"
    mv "$package" "$output"
    echo "reference model published to $output"
}

verify() {
    local package="$default_package" corpus="" app=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --package) package="${2:-}"; shift 2 ;;
            --corpus) corpus="${2:-}"; shift 2 ;;
            --app) app="${2:-}"; shift 2 ;;
            --help|-h) usage; exit 0 ;;
            *) echo "error: unknown verify option '$1'" >&2; usage; exit 2 ;;
        esac
    done
    package="$(realpath "$package")"
    verify_static_package "$package"

    if [ -n "$corpus" ] || [ -n "$app" ]; then
        if [ -z "$corpus" ] || [ -z "$app" ]; then
            echo "error: replay verification requires both --corpus and --app" >&2
            exit 2
        fi
        local work
        work="$(mktemp -d /tmp/dranzer-reference-verify.XXXXXX)"
        trap "rm -rf '$work'" EXIT
        replay_evaluation "$package" "$(realpath "$corpus")" "$(realpath "$app")" "$work"
    fi
}

command="${1:-}"
[ "$#" -gt 0 ] && shift
case "$command" in
    publish) publish "$@" ;;
    verify) verify "$@" ;;
    --help|-h|help) usage ;;
    *) usage; exit 2 ;;
esac
