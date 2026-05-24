#!/usr/bin/env python3
"""Fail a CUDA benchmark run when correctness or performance regressions exceed policy.

Input: benchmarks/results/bench-dbs-cuda-direct.csv produced by
benchmarks/bench_dbs_cuda_direct.py.
"""
import argparse
import csv
import os
import sys
from pathlib import Path


def fenv(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def parse_benchmark_row(row, index):
    try:
        return {
            "B": int(row["B"]),
            "T": int(row["T"]),
            "K": int(row["K"]),
            "V": int(row["V"]),
            "cpu_cuda": float(row.get("max_abs_cpu_cuda_diff", "inf")),
            "cuda_ref": float(row.get("max_abs_cuda_torch_ref_diff", "inf")),
            "peak": float(row.get("peak_cuda_mb", "inf")),
            "speedup": float(row.get("cuda_vs_cpu_speedup", "0")),
            "torch_speedup": float(row.get("cuda_vs_torch_speedup", "0")),
        }, None
    except Exception as exc:
        return None, f"row {index}: parse error: {exc}"


def row_label(row, index):
    return f"row {index} B={row['B']} T={row['T']} K={row['K']} V={row['V']}"


def check_common_thresholds(row, index, args):
    label = row_label(row, index)
    failures = []
    if row["cpu_cuda"] > args.max_diff:
        failures.append(f"{label}: CPU/CUDA diff {row['cpu_cuda']} > {args.max_diff}")
    if row["cuda_ref"] > args.max_diff:
        failures.append(f"{label}: CUDA/reference diff {row['cuda_ref']} > {args.max_diff}")
    if row["peak"] > args.max_peak_mb:
        failures.append(f"{label}: peak CUDA memory {row['peak']}MB > {args.max_peak_mb}MB")
    return failures


def is_large_case(row):
    return row["V"] >= 32000 and row["T"] >= 16 and row["K"] >= 4


def check_large_case_thresholds(row, index, args):
    label = row_label(row, index)
    failures = []
    min_cpu_ratio = args.min_large_cpu_speedup
    if min_cpu_ratio is None:
        min_cpu_ratio = 1.0 / args.max_cuda_vs_cpu_slowdown
    if row["speedup"] < min_cpu_ratio:
        failures.append(f"{label}: DBS CUDA is too slow vs CPU ratio={row['speedup']}")
    # cuda_vs_torch_speedup = torch_ms / dbs_cuda_ms. Values < 0.5 mean DBS is >2x slower than torch ref.
    if row["torch_speedup"] < 1.0 / args.max_cuda_vs_torch_slowdown:
        failures.append(f"{label}: DBS CUDA is too slow vs torch reference ratio={row['torch_speedup']}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default="benchmarks/results/bench-dbs-cuda-direct.csv")
    ap.add_argument("--max-cuda-vs-cpu-slowdown", type=float, default=fenv("DBS_MAX_CUDA_VS_CPU_SLOWDOWN", 2.0))
    ap.add_argument("--min-large-cpu-speedup", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--max-cuda-vs-torch-slowdown", type=float, default=fenv("DBS_MAX_CUDA_VS_TORCH_SLOWDOWN", 2.0))
    ap.add_argument("--max-diff", type=float, default=fenv("DBS_MAX_PARITY_DIFF", 1.0e-5))
    ap.add_argument("--max-peak-mb", type=float, default=fenv("DBS_MAX_PEAK_CUDA_MB", 4096.0))
    args = ap.parse_args()
    if args.max_cuda_vs_cpu_slowdown <= 0.0:
        print("--max-cuda-vs-cpu-slowdown must be positive", file=sys.stderr)
        return 2
    if args.max_cuda_vs_torch_slowdown <= 0.0:
        print("--max-cuda-vs-torch-slowdown must be positive", file=sys.stderr)
        return 2
    if args.min_large_cpu_speedup is not None and args.min_large_cpu_speedup <= 0.0:
        print("--min-large-cpu-speedup must be positive", file=sys.stderr)
        return 2

    path = Path(args.csv_path)
    if not path.exists():
        print(f"missing benchmark csv: {path}", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(path.open()))
    if not rows:
        print("benchmark csv has no rows", file=sys.stderr)
        return 2

    failures = []
    large_rows = 0
    for i, row in enumerate(rows, 1):
        parsed, error = parse_benchmark_row(row, i)
        if error:
            failures.append(error)
            continue

        failures.extend(check_common_thresholds(parsed, i, args))

        if is_large_case(parsed):
            large_rows += 1
            failures.extend(check_large_case_thresholds(parsed, i, args))

    if large_rows == 0:
        failures.append("no large CUDA benchmark rows were found")

    if failures:
        print("performance thresholds failed:", file=sys.stderr)
        for msg in failures:
            print(" - " + msg, file=sys.stderr)
        return 1

    print(f"performance thresholds passed for {len(rows)} rows; large rows={large_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
