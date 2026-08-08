#!/usr/bin/env bash
#
# The claim: a seed identifies a set of weights on any C library, not just on
# the one that produced them.
#
# The direct way to check that would be to build against glibc and musl and
# compare. That is a real check and it belongs in CI, but it cannot run on a
# machine with one libc installed, and it also proves less than it appears to -
# it compares two builds rather than testing the property.
#
# So this tests the property instead. It replaces rand(), srand(), random(), and
# rand_r() with implementations that return deliberate garbage, loads them ahead
# of libc with LD_PRELOAD, and requires the weight fingerprint and the generated
# text to be unchanged. If any path that affects model state or generated tokens
# still reaches the C library's generator, the numbers move and this fails.
#
# That is stronger than a grep for "rand(" - it covers the whole linked program,
# including anything a dependency does - and it runs anywhere, on one libc.
#
# The two checks together are what docs/reproducibility.md's libc row rests on:
# this one for the property, the pinned draws in tests/core/test_rng.c for the
# values.
set -uo pipefail

src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="$(cd "$src_dir/.." && pwd)"
work_dir="$(mktemp -d /tmp/dranzer-libc-independence.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

fingerprint_bin="$src_dir/fingerprint_model.out"
app_bin="$src_dir/app.out"
for binary in "$fingerprint_bin" "$app_bin"; do
    if [ ! -x "$binary" ]; then
        echo "LIBC INDEPENDENCE CHECK SKIPPED: $(basename "$binary") not built" >&2
        echo "  build it with: make -C src fingerprint all" >&2
        exit 0
    fi
done

# A hostile libc RNG. Every entry point returns something a real one would not:
# a constant far from any plausible sequence, so a caller that still draws from
# libc produces visibly different weights rather than subtly different ones.
cat > "$work_dir/hostile_rand.c" <<'EOF'
#include <stdlib.h>
static unsigned long counter = 0;
int rand(void) { return (int)((counter++ * 2654435761u) % 32768u); }
void srand(unsigned int seed) { counter = (unsigned long)seed * 7919u + 13u; }
long random(void) { return (long)rand(); }
void srandom(unsigned int seed) { srand(seed); }
int rand_r(unsigned int *state) { (void)state; return rand(); }
EOF

if ! cc -shared -fPIC -O1 -o "$work_dir/hostile_rand.so" \
        "$work_dir/hostile_rand.c" 2>"$work_dir/build.log"; then
    echo "LIBC INDEPENDENCE CHECK SKIPPED: cannot build a preload shim here" >&2
    sed 's/^/  /' "$work_dir/build.log" >&2
    exit 0
fi

# Confirm the shim actually takes effect in this environment before drawing any
# conclusion from it. A statically linked binary, a hardened loader, or a
# sanitizer build can ignore LD_PRELOAD - and then the check below would pass
# for the wrong reason and report a guarantee nobody verified.
cat > "$work_dir/probe.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
int main(void) { srand(1); printf("%d\n", rand()); return 0; }
EOF
cc -O1 -o "$work_dir/probe" "$work_dir/probe.c" 2>/dev/null || {
    echo "LIBC INDEPENDENCE CHECK SKIPPED: no working C compiler for the probe" >&2
    exit 0
}
probe_plain="$("$work_dir/probe")"
probe_preloaded="$(LD_PRELOAD="$work_dir/hostile_rand.so" "$work_dir/probe")"
if [ "$probe_plain" = "$probe_preloaded" ]; then
    echo "LIBC INDEPENDENCE CHECK SKIPPED: LD_PRELOAD does not override rand()" >&2
    echo "  here (both runs printed $probe_plain), so the check below would" >&2
    echo "  prove nothing. Run it on a platform where preloading works." >&2
    exit 0
fi
echo "preload shim verified: rand() moved from $probe_plain to $probe_preloaded"

failed=0

# --- 1. Initial weights, across several architectures and seeds ---
for spec in "42:260:16:2:2:32" "7:80:24:3:1:16" "0:64:8:2:3:8"; do
    IFS=: read -r seed vocab emb heads layers max_seq <<< "$spec"
    args=(--seed "$seed" --vocab "$vocab" --embedding-dim "$emb"
          --heads "$heads" --layers "$layers" --max-seq-len "$max_seq" --quiet)
    plain="$("$fingerprint_bin" "${args[@]}")"
    hostile="$(LD_PRELOAD="$work_dir/hostile_rand.so" "$fingerprint_bin" "${args[@]}")"
    if [ "$plain" != "$hostile" ]; then
        echo "FAIL seed $seed (${vocab}x${emb}, ${layers}L): weights depend on libc's rand()"
        echo "     normal run  $plain"
        echo "     hostile run $hostile"
        failed=1
    else
        echo "  seed $seed (${vocab}x${emb}, ${layers}L): $plain unchanged"
    fi
done

# --- 2. Generated text, which is the other thing a seed is supposed to name ---
# Sampling is the remaining consumer of randomness. Temperature and top-k are
# set so the run genuinely samples rather than falling through to greedy, since
# greedy decoding would be unaffected by any RNG and would pass vacuously.
generate() {
    cd "$work_dir" || exit 1
    "$app_bin" train --input "$repo_root/test.txt" --model libc.pth \
        --vocab-size 260 --embedding-dim 8 --num-heads 2 --num-layers 1 \
        --max-seq-len 16 --train-window 8 --epochs 1 --batch-size 4 \
        --learning-rate 0.002 --seed 42 >/dev/null 2>&1 || return 1
    "$app_bin" generate --model libc.pth --prompt "the" \
        --length 24 --temperature 0.9 --top-k 8 --sampling topk \
        --seed 1234 2>/dev/null
}
plain_text="$(generate)"
generate_status=$?
# Exported rather than prefixed onto the call: the function runs two child
# processes, and both have to inherit the shim.
export LD_PRELOAD="$work_dir/hostile_rand.so"
hostile_text="$(generate)"
hostile_status=$?
unset LD_PRELOAD

if [ $generate_status -ne 0 ] || [ $hostile_status -ne 0 ]; then
    echo "FAIL could not complete a train+generate pair (exit $generate_status/$hostile_status)"
    failed=1
elif [ -z "$plain_text" ]; then
    echo "FAIL the generate run produced no output to compare"
    failed=1
elif [ "$plain_text" != "$hostile_text" ]; then
    echo "FAIL generated text depends on libc's rand()"
    diff <(printf '%s' "$plain_text") <(printf '%s' "$hostile_text") | head -20
    failed=1
else
    echo "  sampled generation: $(printf '%s' "$plain_text" | wc -c) bytes unchanged"
fi

if [ $failed -eq 0 ]; then
    echo "LIBC INDEPENDENCE CHECK PASSED"
else
    echo "LIBC INDEPENDENCE CHECK FAILED"
fi
exit $failed
