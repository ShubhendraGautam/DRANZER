#!/usr/bin/env bash
# Repeatable cross-compiler matmul sweep: the procedure behind every kernel
# and tile-size claim in docs/matmul.md. Candidates are ranked by their
# fastest round, since contention can only ever make a round slower.
#
# For each requested compiler it does a clean build of bench.out, runs the
# isolated matmul sweep over the requested tiers, and then prints, per shape
# and compiler, which candidate won and by how much over the portable scalar
# reference. Every row it summarises is also appended to the results CSV with
# full build/host provenance, so a claim can always be traced back.
#
# Usage:
#   tools/matmul_sweep.sh                       # gcc + clang, all tiers
#   tools/matmul_sweep.sh --compilers clang     # one compiler
#   tools/matmul_sweep.sh --tiers "small medium"
#   tools/matmul_sweep.sh --repeats 5 --omp     # more rounds, OpenMP build
#   tools/matmul_sweep.sh --quick               # smoke test, NOT for selection
#
# Run it on an otherwise idle machine: the summary reports the spread between
# the best and median round so a noisy session is visible rather than silent.

set -u

compilers="gcc clang"
tiers="tiny small medium"
repeats=3
quick=""
omp=""
csv="matmul_results_v2.csv"

while [ $# -gt 0 ]; do
    case "$1" in
        --compilers) compilers="$2"; shift 2 ;;
        --tiers)     tiers="$2"; shift 2 ;;
        --repeats)   repeats="$2"; shift 2 ;;
        --csv)       csv="$2"; shift 2 ;;
        --quick)     quick="--quick"; shift ;;
        --omp)       omp="OMP=1"; shift ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Error: unknown option '$1'" >&2; exit 2 ;;
    esac
done

script_dir="$(cd "$(dirname "$0")" && pwd)"
src_dir="$(dirname "$script_dir")"
cd "$src_dir" || exit 1

# A fresh file per sweep keeps one session's rows separable from the running
# local history the benchmark otherwise appends to.
session_csv="$csv"
started_at="$(date +%Y-%m-%dT%H:%M:%S)"
echo "Matmul sweep started $started_at"
echo "  compilers: $compilers"
echo "  tiers:     $tiers"
echo "  repeats:   $repeats ${quick:+(quick mode - smoke test only)}"
echo "  results:   $src_dir/$session_csv"
echo

failed=0
for cc in $compilers; do
    if ! command -v "$cc" >/dev/null 2>&1; then
        echo "Skipping $cc: not installed"
        continue
    fi
    echo "=== $cc ==="
    # shellcheck disable=SC2086
    if ! make clean >/dev/null 2>&1 || ! make bench CC="$cc" $omp >/dev/null 2>&1; then
        echo "BUILD FAILED for $cc" >&2
        failed=1
        continue
    fi
    for tier in $tiers; do
        # shellcheck disable=SC2086
        if ! ./bench.out --matmul-only --sweep --tier "$tier" \
                --repeats "$repeats" $quick --csv-path "$session_csv"; then
            echo "SWEEP FAILED for $cc tier $tier" >&2
            failed=1
        fi
    done
    echo
done

if [ ! -f "$session_csv" ]; then
    echo "No results file produced." >&2
    exit 1
fi

echo "=== summary: fastest candidate per shape (this sweep only) ==="
awk -v started="$started_at" '
    # Quote-aware CSV split: the provenance columns hold free-form build
    # commands and CPU model strings, so a plain -F"," would misalign the
    # moment one of them contains a comma.
    function csv_split(line, out,   i, n, field, inq, ch) {
        n = 0; field = ""; inq = 0
        for (i = 1; i <= length(line); i++) {
            ch = substr(line, i, 1)
            if (inq) {
                if (ch != "\"") { field = field ch }
                else if (substr(line, i + 1, 1) == "\"") { field = field "\""; i++ }
                else { inq = 0 }
            } else if (ch == "\"") { inq = 1 }
            else if (ch == ",") { out[++n] = field; field = "" }
            else { field = field ch }
        }
        out[++n] = field
        delete out[n + 1]
        return n
    }
    NR == 1 { next }
    {
        if (csv_split($0, f) < 19) next
        if (f[1] < started) next
        tier = f[2]; shape = f[3]; kernel = f[7]; tile = f[8] + 0
        rounds = f[10] + 0; best = f[12] + 0; median = f[13] + 0
        if (best <= 0) next
        split(f[19], parts, " ")
        key = parts[1] "|" tier "|" shape
        label = kernel
        if (kernel == "tiled" || kernel == "tiled_mr4") label = kernel "/tile" tile
        if (kernel == "scalar") scalar_us[key] = best
        else if (kernel == "auto") auto_us[key] = best
        if (kernel != "auto" && kernel != "scalar" && \
            (!(key in best_us) || best < best_us[key])) {
            best_us[key] = best
            best_label[key] = label
            best_noise[key] = (median - best) / best * 100.0
            best_rounds[key] = rounds
        }
    }
    END {
        printf "%-8s %-7s %-30s %-18s %10s %9s %9s %7s\n", \
               "compiler", "tier", "shape", "fastest", "us", "vs scalar", \
               "auto gap", "noise"
        for (key in best_us) {
            split(key, parts, "|")
            # A single round has no spread to report; saying "0.0%" there
            # would read as a quiet machine rather than as no evidence.
            noise = best_rounds[key] > 1 ? sprintf("%6.1f%%", best_noise[key]) \
                                         : "   n/a"
            printf "%-8s %-7s %-30s %-18s %10.3f %8.2fx %8.2fx %s\n", \
                   parts[1], parts[2], parts[3], best_label[key], best_us[key], \
                   (key in scalar_us && best_us[key] > 0 ? scalar_us[key] / best_us[key] : 0), \
                   (key in auto_us && best_us[key] > 0 ? auto_us[key] / best_us[key] : 0), \
                   noise
        }
    }
' "$session_csv" | { read -r header; echo "$header"; sort -k1,1 -k2,2 -k3,3; }

echo
echo "\"auto gap\" is how much slower the shipped default policy is than the"
echo "fastest explicit candidate for that shape - 1.00x means the policy"
echo "already picks the best kernel there. \"noise\" is the median-to-best"
echo "spread of the winning candidate's rounds: treat anything above roughly"
echo "10% as a session too noisy to retune from."
exit $failed
