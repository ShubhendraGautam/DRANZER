#!/usr/bin/env bash
set -euo pipefail

# Executable frontends and probe/report modules may print. Embeddable runtime
# code may format into caller/file buffers, but must never choose stdout or
# stderr on an application's behalf.
paths=(
    core
    api
    ../libs/src
    backends/gpu/gpu_cuda.c
    backends/gpu/gpu_matmul.c
)

if rg -n '\b(printf|fprintf|puts|putchar|perror)[[:space:]]*\(' "${paths[@]}"; then
    echo "library-silence-check: terminal output call found in embeddable code" >&2
    exit 1
fi

echo "library-silence-check: embeddable code has no terminal output calls"
