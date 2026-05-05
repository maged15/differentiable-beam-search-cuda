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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default="benchmarks/results/bench-dbs-cuda-direct.csv")
    ap.add_argument("--min-large-cpu-speedup", type=float, default=fenv("DBS_MIN_LARGE_CPU_SPEEDUP", 2.0))
    ap.add_argument("--max-cuda-vs-torch-slowdown", type=float, default=fenv("DBS_MAX_CUDA_VS_TORCH_SLOWDOWN", 2.0))
    ap.add_argument("--max-diff", type=float, default=fenv("DBS_MAX_PARITY_DIFF", 1.0e-5))
    ap.add_argument("--max-peak-mb", type=float, default=fenv("DBS_MAX_PEAK_CUDA_MB", 4096.0))
    args = ap.parse_args()

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
        try:
            B = int(row["B"]); T = int(row["T"]); K = int(row["K"]); V = int(row["V"])
            cpu_cuda = float(row.get("max_abs_cpu_cuda_diff", "inf"))
            cuda_ref = float(row.get("max_abs_cuda_torch_ref_diff", "inf"))
            peak = float(row.get("peak_cuda_mb", "inf"))
            speedup = float(row.get("cuda_vs_cpu_speedup", "0"))
            torch_speedup = float(row.get("cuda_vs_torch_speedup", "0"))
        except Exception as exc:
            failures.append(f"row {i}: parse error: {exc}")
            continue

        if cpu_cuda > args.max_diff:
            failures.append(f"row {i} B={B} T={T} K={K} V={V}: CPU/CUDA diff {cpu_cuda} > {args.max_diff}")
        if cuda_ref > args.max_diff:
            failures.append(f"row {i} B={B} T={T} K={K} V={V}: CUDA/reference diff {cuda_ref} > {args.max_diff}")
        if peak > args.max_peak_mb:
            failures.append(f"row {i} B={B} T={T} K={K} V={V}: peak CUDA memory {peak}MB > {args.max_peak_mb}MB")

        # Require useful CUDA acceleration only for meaningful large cases.
        if V >= 32000 and T >= 16 and K >= 4:
            large_rows += 1
            if speedup < args.min_large_cpu_speedup:
                failures.append(f"row {i} B={B} T={T} K={K} V={V}: CUDA/CPU speedup {speedup} < {args.min_large_cpu_speedup}")
            # cuda_vs_torch_speedup = torch_ms / dbs_cuda_ms. Values < 0.5 mean DBS is >2x slower than torch ref.
            if torch_speedup < 1.0 / args.max_cuda_vs_torch_slowdown:
                failures.append(f"row {i} B={B} T={T} K={K} V={V}: DBS CUDA is too slow vs torch reference ratio={torch_speedup}")

    if large_rows == 0:
        failures.append("no large CUDA benchmark rows were found")

    if failures:
        print("performance gate failed:", file=sys.stderr)
        for msg in failures:
            print(" - " + msg, file=sys.stderr)
        return 1

    print(f"performance gate passed for {len(rows)} rows; large rows={large_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
