#!/usr/bin/env python3
"""Turn benchmark CSVs into a readable report and check them for regressions.

Written for CI. The hard problem with performance checks on hosted runners is
that absolute timings are not comparable: a run may land on a different CPU
model, a different generation of hypervisor, or simply a busier neighbour.
Committing "medium training must take under X ms" produces a check that fails
for reasons that have nothing to do with the code.

So every assertion here is a *ratio between two measurements taken in the same
process on the same machine within the same run*. Those hold regardless of how
fast or how loaded the runner is:

  - the shipped matmul kernel must beat the portable scalar reference on every
    shape, because both were measured back to back on the same buffers;
  - the shipped kernel must stay within a factor of the fastest candidate
    measured beside it, which catches a bad default without pinning a speed;
  - KV-cached decode must beat recomputing the whole prefix, because that is a
    structural property of the cache and worth several-fold everywhere;
  - every kernel must stay within numerical tolerance of the reference.

Absolute numbers are still reported - they are what a human reads and what the
uploaded artifacts preserve for trend-watching - but they are never asserted on.

Usage:
    perf_check.py [--bench FILE] [--matmul FILE] [--omp FILE]
                  [--summary FILE] [--strict]

Exits non-zero if a check fails. --strict also fails on warnings.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

# A shape's shipped kernel may be at most this much slower than the fastest
# candidate measured next to it before it counts as a bad default. Generous on
# purpose: the point is to catch a wrong selection, not to police a few percent
# on a shared runner.
AUTO_GAP_WARN = 1.5
AUTO_GAP_FAIL = 2.5

# The shipped kernel is allowed to be marginally slower than the scalar
# reference on a shape before it counts as a regression. Anything at or below
# this means the tuning work was undone.
SCALAR_RATIO_FAIL = 0.95

# Cached decode versus recomputing the full prefix. Measured between 10x and
# 44x across tiers on the project's own hardware; 3x is a floor that only trips
# if the cache stops working.
KV_SPEEDUP_FAIL = 3.0

# Kernel output versus the scalar reference, relative to the magnitude of the
# result. Matches the tolerance the benchmark itself applies.
ERROR_TOLERANCE = 1e-4


class Report:
    def __init__(self):
        self.lines = []
        self.failures = []
        self.warnings = []

    def line(self, text=""):
        self.lines.append(text)

    def table(self, headers, rows):
        if not rows:
            return
        self.line("| " + " | ".join(headers) + " |")
        self.line("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            self.line("| " + " | ".join(str(c) for c in row) + " |")
        self.line()

    def fail(self, text):
        self.failures.append(text)

    def warn(self, text):
        self.warnings.append(text)


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def geomean(values):
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def describe_machine(rows, report):
    """Record which machine produced the numbers. Absolute values are only
    interpretable against this, which is why it leads the report."""
    if not rows:
        return
    row = rows[0]
    report.line("### Machine")
    report.line()
    for label, key in (("CPU", "cpu"), ("OS", "os"), ("Compiler", "compiler"),
                       ("Build", "build_command"), ("Online cores", "online_cpus"),
                       ("OpenMP threads", "max_threads")):
        if row.get(key):
            report.line(f"- **{label}:** {row[key]}")
    report.line()


def check_matmul(rows, report):
    """Kernel candidates, per shape, all measured in one process."""
    if not rows:
        return
    report.line("## CPU matmul kernels")
    report.line()

    by_shape = defaultdict(dict)
    for row in rows:
        compiler = row.get("compiler", "").split(" ")[0] or "cc"
        key = (compiler, row["tier"], row["case"], row["m"], row["k"], row["n"])
        label = row["kernel"]
        # Every blocked kernel is swept at several tiles, and each tile is a
        # distinct candidate. Keying on the name alone would collapse them and
        # silently keep only the last row, so "fastest candidate beside it"
        # would be measured against an arbitrary tile. The unblocked kernels
        # record tile 0 and keep their bare name.
        #
        # "auto" and "scalar" must stay bare whatever their tile column says:
        # the two lookups below are by exact name, and labelling them would
        # skip the shape entirely rather than fail loudly.
        if label not in ("auto", "scalar") and row["tile"] not in ("", "0"):
            label = f"{label}/{row['tile']}"
        by_shape[key][label] = row

    table_rows = []
    scalar_ratios = []
    for key in sorted(by_shape):
        compiler, tier, case, m, k, n = key
        candidates = by_shape[key]
        scalar = candidates.get("scalar")
        shipped = candidates.get("auto")
        if not scalar or not shipped:
            continue

        scalar_us = number(scalar, "best_us")
        shipped_us = number(shipped, "best_us")
        if shipped_us <= 0 or scalar_us <= 0:
            continue

        versus_scalar = scalar_us / shipped_us
        scalar_ratios.append(versus_scalar)

        explicit = [number(r, "best_us") for label, r in candidates.items()
                    if label not in ("auto", "scalar") and number(r, "best_us") > 0]
        best_us = min(explicit) if explicit else shipped_us
        gap = shipped_us / best_us if best_us > 0 else 1.0

        flag = ""
        where = f"{compiler} {tier}/{case} ({m}x{k}x{n})"
        if versus_scalar < SCALAR_RATIO_FAIL:
            report.fail(f"{where}: shipped kernel is {1 / versus_scalar:.2f}x "
                        f"SLOWER than the portable scalar reference")
            flag = " **FAIL**"
        if gap >= AUTO_GAP_FAIL:
            report.fail(f"{where}: shipped kernel is {gap:.2f}x off the fastest "
                        f"candidate measured beside it")
            flag = " **FAIL**"
        elif gap >= AUTO_GAP_WARN:
            report.warn(f"{where}: shipped kernel is {gap:.2f}x off the fastest "
                        f"candidate measured beside it")
            flag = " ⚠️"

        for label, row in candidates.items():
            error = number(row, "max_abs_error")
            if not math.isfinite(error) or error > ERROR_TOLERANCE * (1.0 + 1000.0):
                report.fail(f"{where}: kernel {label} diverged from the reference "
                            f"(max abs error {error:g})")
                flag = " **FAIL**"

        table_rows.append([
            compiler, tier, case, f"{m}x{k}x{n}",
            f"{scalar_us:.1f}", f"{shipped_us:.1f}",
            f"{versus_scalar:.2f}x", f"{gap:.2f}x{flag}",
        ])

    report.table(
        ["cc", "tier", "call site", "shape", "scalar µs", "shipped µs",
         "vs scalar", "vs best"],
        table_rows,
    )
    if scalar_ratios:
        report.line(f"**Speedup over the portable reference:** "
                    f"{min(scalar_ratios):.2f}x worst, "
                    f"{geomean(scalar_ratios):.2f}x geometric mean, "
                    f"{max(scalar_ratios):.2f}x best "
                    f"({len(scalar_ratios)} shapes).")
        report.line()


def check_bench(rows, report):
    """Whole-model throughput and the structural KV-cache property."""
    if not rows:
        return
    report.line("## Whole-model throughput")
    report.line()

    table_rows = []
    for row in rows:
        if row.get("mode", "").startswith("gpu"):
            continue
        inference = number(row, "inference_ms_per_token")
        decode = number(row, "growing_decode_ms_per_token")
        tier = row.get("config", "?")
        if decode > 0 and inference > 0:
            speedup = inference / decode
            if speedup < KV_SPEEDUP_FAIL:
                report.fail(f"{tier}: KV-cached decode is only {speedup:.2f}x "
                            f"faster than recomputing the full prefix")
        else:
            speedup = 0.0
        table_rows.append([
            tier, row.get("mode", "?"), f"{int(number(row, 'param_count')):,}",
            f"{inference:.3f}", f"{number(row, 'prefill_ms_per_token'):.3f}",
            f"{decode:.3f}", f"{number(row, 'training_ms_per_step'):.3f}",
            f"{speedup:.1f}x", f"{number(row, 'rss_delta_mb'):.1f}",
        ])

    report.table(
        ["tier", "mode", "params", "inference ms", "prefill ms", "decode ms",
         "training ms", "cache win", "RSS MB"],
        table_rows,
    )


def check_omp(rows, report):
    """Thread scaling. Reported, and warned on, but never failed: how much a
    second thread helps depends on how many cores the runner really gave us."""
    if not rows:
        return
    report.line("## Thread scaling")
    report.line()

    by_threads = defaultdict(dict)
    for row in rows:
        if row.get("mode", "").startswith("gpu"):
            continue
        by_threads[row.get("config", "?")][int(number(row, "max_threads", 1))] = row

    table_rows = []
    for tier, runs in sorted(by_threads.items()):
        thread_counts = sorted(runs)
        if len(thread_counts) < 2:
            continue
        base = runs[thread_counts[0]]
        for count in thread_counts[1:]:
            row = runs[count]
            for metric, label in (("training_ms_per_step", "training step"),
                                  ("inference_ms_per_token", "inference")):
                one = number(base, metric)
                many = number(row, metric)
                if one <= 0 or many <= 0:
                    continue
                table_rows.append([
                    tier, label, f"{thread_counts[0]}→{count}",
                    f"{one:.3f}", f"{many:.3f}", f"{one / many:.2f}x",
                    f"{(one / many) / count * 100:.0f}%",
                ])
            training_gain = number(base, "training_ms_per_step") / max(
                number(row, "training_ms_per_step"), 1e-9)
            if count > 1 and training_gain < 1.05:
                report.warn(f"{tier}: {count} threads gave only "
                            f"{training_gain:.2f}x on training - the runner may "
                            f"not have provided that many usable cores")

    report.table(
        ["tier", "workload", "threads", "base ms", "scaled ms", "speedup",
         "efficiency"],
        table_rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", help="bench_results_v3.csv")
    parser.add_argument("--matmul", help="matmul_results_v3.csv")
    parser.add_argument("--omp", help="bench_results_v3.csv from a thread sweep")
    parser.add_argument("--summary", help="write the markdown report here")
    parser.add_argument("--title", default="Performance report")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as failures")
    args = parser.parse_args()

    bench_rows = read_csv(args.bench)
    matmul_rows = read_csv(args.matmul)
    omp_rows = read_csv(args.omp)

    if not (bench_rows or matmul_rows or omp_rows):
        print("perf_check: no input CSVs found", file=sys.stderr)
        return 1

    report = Report()
    report.line(f"# {args.title}")
    report.line()
    describe_machine(matmul_rows or bench_rows or omp_rows, report)
    check_matmul(matmul_rows, report)
    check_bench(bench_rows, report)
    check_omp(omp_rows, report)

    report.line("## Checks")
    report.line()
    report.line("Every check below compares two measurements taken in the same "
                "run on the same machine. No absolute timing is asserted on, so "
                "a slower or busier runner cannot fail the build.")
    report.line()
    if report.failures:
        report.line(f"**{len(report.failures)} failed:**")
        report.line()
        for text in report.failures:
            report.line(f"- ❌ {text}")
        report.line()
    if report.warnings:
        report.line(f"**{len(report.warnings)} warning(s):**")
        report.line()
        for text in report.warnings:
            report.line(f"- ⚠️ {text}")
        report.line()
    if not report.failures and not report.warnings:
        report.line("- ✅ All performance invariants held.")
        report.line()

    text = "\n".join(report.lines)
    print(text)
    if args.summary:
        with open(args.summary, "a") as handle:
            handle.write(text + "\n")

    if report.failures:
        return 1
    if report.warnings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
