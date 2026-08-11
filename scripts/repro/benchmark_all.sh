#!/usr/bin/env bash
#
# One command, one file: every throughput number this project can produce, across
# the axes that actually change it.
#
# Why a script rather than a note saying "run bench.out"
# -----------------------------------------------------
# `bench.out` measures one build. The interesting question is not "how fast is
# this build" but "which of the available choices matter", and the answer turned
# out to be surprising enough that guessing was not safe:
#
#   - the default build has NO OpenMP compiled in, so it is single-threaded no
#     matter how many cores the machine has;
#   - gcc and clang differ by about 3x on the same source and the same flags;
#   - threading helps attention at long context and does almost nothing to a
#     training step at short context, because the optimizer and the norms are
#     serial.
#
# A reader comparing two numbers from this project has to know which of those
# three they differ by, so all of them are measured together, in one file, with
# the provenance attached.
#
# Each configuration is a full clean rebuild. Reusing objects between compilers
# is how a "compiler effect" row ends up measuring one compiler.
#
# Every cell is a MEDIAN over repeats, with the spread reported beside the
# training column. One reading per configuration is not enough here and that is
# measured, not assumed: repeating the identical configuration back to back on
# this machine gave 6.81, 7.02, 7.21, 7.44, and 9.31 ms/step - a 1.37x spread.
# Any single-reading comparison of two configurations closer than that is
# reporting machine noise. An earlier version of this script did exactly that and
# produced an apparent 2.95x compiler difference that a paired interleaved
# comparison then put at 1.10x.
#
# Usage: scripts/repro/benchmark_all.sh [output.md] [--quick] [--repeats N]
#          --quick     tiny and small tiers only (medium is most of the wall time)
#          --repeats N readings per cell, median reported (default 3)
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
src_dir="$repo_root/src"
work_dir="$(mktemp -d /tmp/dranzer-benchall.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

output="$repo_root/docs/generated/benchmark-all.md"
output_set=0
quick=0
repeats=3
failures=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --quick)
            quick=1
            shift
            ;;
        --repeats)
            if [ "$#" -lt 2 ]; then
                echo "error: --repeats needs a positive integer" >&2
                exit 2
            fi
            repeats="$2"
            case "$repeats" in
                ''|*[!0-9]*|0)
                    echo "error: --repeats needs a positive integer, got '$repeats'" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        --help|-h)
            printf '%s\n' \
                "Usage: scripts/repro/benchmark_all.sh [output.md] [--quick] [--repeats N]" \
                "  --quick     tiny and small tiers only (medium is most of the wall time)" \
                "  --repeats N readings per cell, median reported (default 3)"
            exit 0
            ;;
        --*)
            echo "error: unknown option '$1'" >&2
            exit 2
            ;;
        *)
            if [ "$output_set" -ne 0 ]; then
                echo "error: more than one output file supplied" >&2
                exit 2
            fi
            output="$1"
            output_set=1
            shift
            ;;
    esac
done

# Resolved to an absolute path BEFORE anything runs. run_config() cd's into the
# source tree, so a relative output path stops resolving partway through - and
# because the report is written last, that failure discards every measurement
# taken up to that point. It did: a full five-configuration run was lost to
# exactly this. realpath -m does not require the file to exist yet.
output="$(realpath -m "$output")"
mkdir -p "$(dirname "$output")"

# Per-run dumps are kept, but OUTSIDE the repository. They exist for one reason -
# a measurement that took minutes to make should survive a parsing bug in the
# report step, which has already cost one full run - and that reason does not
# justify committing fifteen near-identical text files into docs/. The directory
# is printed at the end so it can be inspected, and it is left behind rather than
# cleaned up so a bad column can still be traced afterwards.
raw_dir="${DRANZER_BENCH_RAW_DIR:-${TMPDIR:-/tmp}/dranzer-bench-raw-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$raw_dir"

TIERS="tiny small medium"
[ "$quick" -eq 1 ] && TIERS="tiny small"

# --- one configuration ------------------------------------------------------
# $1 label, $2 CC, $3 make args, $4 env, $5 extra bench args
run_config() {
    local label="$1" cc="$2" make_args="$3" env_spec="$4" bench_args="$5"
    local dir="$work_dir/$(echo "$label" | tr -c 'a-zA-Z0-9' '_')"
    mkdir -p "$dir"

    echo "  building: $label" >&2
    cd "$src_dir" || return 1
    make clean >/dev/null 2>&1
    if ! make bench -j2 CC="$cc" $make_args >"$dir/build.log" 2>&1; then
        echo "$label|BUILD FAILED||||||||" >> "$work_dir/rows"
        echo "  BUILD FAILED: $label (see $dir/build.log)" >&2
        failures=1
        return 0
    fi
    cp bench.out "$dir/bench.out"

    for tier in $TIERS; do
        for rep in $(seq 1 "$repeats"); do
            echo "    $tier (reading $rep/$repeats)" >&2
            # Run from its own directory so the CSV each run appends to does not
            # mix configurations into one file.
            if ! ( cd "$dir" && env $env_spec ./bench.out --tier "$tier" $bench_args \
                    > "$dir/$tier-$rep.txt" 2>&1 ); then
                echo "$label|$tier|RUN FAILED|||||||" >> "$work_dir/rows"
                echo "  RUN FAILED: $label/$tier reading $rep (see $dir/$tier-$rep.txt)" >&2
                failures=1
                cp "$dir/$tier-$rep.txt" \
                   "$raw_dir/$(basename "$dir")-$tier-$rep.txt" 2>/dev/null
                continue
            fi
            # Copied out of the temp tree immediately, not at the end: these are
            # the only record of a measurement that took minutes to make, and the
            # parse can be redone from them if a column comes out wrong.
            cp "$dir/$tier-$rep.txt" "$raw_dir/$(basename "$dir")-$tier-$rep.txt" 2>/dev/null
            parse_tier "$label" "$tier" "$dir/$tier-$rep.txt"
        done
    done
}

# Pull every CPU/GPU section out of one tier's report into a pipe-delimited row.
# A GPU-capable invocation prints a CPU section followed by a GPU section. The
# old parser kept the first inference value but overwrote the later training and
# cache values, producing a row that described no run that actually happened.
# Emitting at each section's peak-RSS line keeps every value attached to the
# mode that produced it.
parse_tier() {
    local label="$1" tier="$2" file="$3"
    # Field positions are pinned to bench.out's exact report layout, e.g.
    #   0.460 ms/step    (2173.9 steps/sec, 69563.3 context-tokens/sec)
    #   full-prefix 0.073 ms/token   KV-cache 0.010 ms/token   speedup 7.66x
    #   process peak RSS after this config: 2.40 MB (delta from process start: ...)
    # A layout change here silently produces the wrong column rather than an
    # error, so the report prints "-" for anything that did not parse and a
    # missing value is visible.
    awk -v label="$label" -v tier="$tier" '
        /^--- / {
            mode=$0
            sub(/^.* mode=/, "", mode)
            sub(/ ---$/, "", mode)
            params=infer=step=ctx=kv=kvspeed=rss=""
        }
        /parameters:/                  { params=$2 }
        /ms\/token/ && /tokens\/sec\)/ && infer=="" { infer=$1 }
        /ms\/step/                     { step=$1; ctx=$5 }
        /KV-cache/                     { kv=$5; kvspeed=$8 }
        /peak RSS after this config/ {
            rss=$7
            printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n",
                   label, tier, mode, params, infer, step, ctx, kv, kvspeed, rss
        }
    ' "$file" >> "$work_dir/rows"
}

echo "collecting provenance" >&2
cd "$src_dir" || exit 1
cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')
cores=$(nproc)
host_ram=$(awk '/MemTotal/{printf "%.1f GB", $2/1048576}' /proc/meminfo)
gcc_ver=$(gcc --version | head -1)
clang_ver=$(clang --version | head -1)
make bench -j2 CC=clang >/dev/null 2>&1
# Probed in its own throwaway run: gpu_matmul_available() dlopen()s libcuda and
# adds ~99 MB of RSS, so it must not happen inside a run whose memory is reported.
gpu_line=$(cd "$work_dir" && "$src_dir/bench.out" --tier tiny 2>/dev/null | grep -E 'CUDA GPU detected|No CUDA GPU' | head -1)

: > "$work_dir/rows"

echo "measuring" >&2
# 1. The default build, exactly as `make` produces it: no OpenMP at all.
run_config "clang, default (no OpenMP)" clang "" "" "--cpu-only"
# 2. Same source, same flags, other compiler. Isolates the compiler.
run_config "gcc, default (no OpenMP)" gcc "" "" "--cpu-only"
# 3. OpenMP compiled in but pinned to one thread. Isolates the cost of the
#    OpenMP machinery itself from the benefit of using it.
run_config "gcc, OpenMP, 1 thread" gcc "OMP=1" "OMP_NUM_THREADS=1" "--cpu-only"
# 4. OpenMP with every core. Isolates threading.
run_config "gcc, OpenMP, $cores threads" gcc "OMP=1" "OMP_NUM_THREADS=$cores" "--cpu-only"
# 5. GPU pass included. bench.out runs each config on CPU and GPU when a device
#    is usable; parse_tier() keeps those as two labelled rows.
run_config "gcc, OpenMP + GPU" gcc "OMP=1" "OMP_NUM_THREADS=$cores" ""

# --- report -----------------------------------------------------------------
{
    echo "# Consolidated benchmark"
    echo
    echo "Generated by \`scripts/repro/benchmark_all.sh\` on $(date -u '+%Y-%m-%d %H:%M UTC')."
    echo
    echo "## Machine"
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| CPU | $cpu_model |"
    echo "| Cores visible | $cores |"
    echo "| RAM visible | $host_ram |"
    echo "| Kernel | $(uname -sr) |"
    echo "| gcc | $gcc_ver |"
    echo "| clang | $clang_ver |"
    echo "| GPU | ${gpu_line:-unknown} |"
    echo
    echo "## Results"
    echo
    echo "Inference is ms/token with a full context each step. Training is ms/step"
    echo "(forward + backward + optimizer). KV is ms/token with the cache, and its"
    echo "speedup is against re-running the full prefix. RSS is process peak."
    echo
    echo "**Every cell is the median of $repeats repeated run(s)**, and the training column"
    echo "carries the observed range. Repeating one configuration back to back on"
    echo "this machine spans about 1.37x, so two configurations whose ranges overlap"
    echo "are not distinguishable by this benchmark - read the spread, not the"
    echo "third digit. Differences smaller than that need a paired interleaved"
    echo "comparison (see \`include/tools/timing_spread.h\`)."
    echo
    python3 - "$work_dir/rows" "$repeats" <<'PYTHON'
import sys, statistics as st, collections
rows_path, repeats = sys.argv[1], int(sys.argv[2])
cells = collections.OrderedDict()
for line in open(rows_path):
    f = line.rstrip("\n").split("|")
    if len(f) < 10:
        continue
    label, tier, mode = f[0], f[1], f[2]
    if tier == "BUILD FAILED" or mode == "RUN FAILED":
        cells.setdefault((label, tier, mode or "-"), None)
        continue
    entry = cells.setdefault((label, tier, mode), collections.defaultdict(list))
    for name, idx in (("params",3), ("infer",4), ("step",5), ("ctx",6),
                      ("kv",7), ("kvspeed",8), ("rss",9)):
        try:
            entry[name].append(float(f[idx].rstrip("x")))
        except (ValueError, IndexError):
            pass

def med(values):
    return st.median(values) if values else None

print("| configuration | tier | mode | params | infer ms/tok | train ms/step | train spread | train tok/s | KV ms/tok | KV speedup | peak RSS |")
print("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for (label, tier, mode), e in cells.items():
    if e is None:
        failure = "build failed" if tier == "BUILD FAILED" else "run failed"
        print(f"| {label} | {tier if tier != 'BUILD FAILED' else '-'} | {mode} | {failure} | | | | | | | |")
        continue
    steps = e["step"]
    spread = (f"{min(steps):.3f}-{max(steps):.3f}"
              if len(steps) > 1 else "single reading")
    def fmt(name, digits=3):
        v = med(e[name])
        return "-" if v is None else f"{v:.{digits}f}"
    print(f"| {label} | {tier} | {mode} | {int(med(e['params']) or 0)} | {fmt('infer')} | "
          f"{fmt('step')} | {spread} | {fmt('ctx',1)} | {fmt('kv')} | "
          f"{fmt('kvspeed',2)}x | {fmt('rss',2)} MB |")
PYTHON
    echo
    echo "Full per-tier reports for every configuration and every reading were"
    echo "written to \`$raw_dir\` (outside the repository), so a number above can be"
    echo "checked against the run that produced it. Set \`DRANZER_BENCH_RAW_DIR\` to"
    echo "put them somewhere else."
} > "$output"

if ! (cd "$src_dir" && make clean >/dev/null 2>&1 && make -j2 >/dev/null 2>&1); then
    echo "error: could not restore the default build" >&2
    exit 1
fi
echo "wrote $output" >&2
echo "raw per-run reports: $raw_dir" >&2
exit "$failures"
