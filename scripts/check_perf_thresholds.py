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


LARGE_VOCAB_THRESHOLD = 32000
LARGE_STEP_THRESHOLD = 16
LARGE_BEAM_THRESHOLD = 4


def fenv(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default="benchmarks/results/bench-dbs-cuda-direct.csv")
    ap.add_argument("--min-large-cpu-speedup", type=float, default=fenv("DBS_MIN_LARGE_CPU_SPEEDUP", 2.0))
    ap.add_argument("--max-cuda-vs-torch-slowdown", type=float, default=fenv("DBS_MAX_CUDA_VS_TORCH_SLOWDOWN", 2.0))
    ap.add_argument("--max-diff", type=float, default=fenv("DBS_MAX_PARITY_DIFF", 1.0e-5))
    ap.add_argument("--max-peak-mb", type=float, default=fenv("DBS_MAX_PEAK_CUDA_MB", 4096.0))
    return ap.parse_args()


def read_rows(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        print(f"missing benchmark csv: {path}", file=sys.stderr)
        return None

    rows = list(csv.DictReader(path.open()))
    if not rows:
        print("benchmark csv has no rows", file=sys.stderr)
        return None
    return rows


def parse_row(row: dict[str, str]) -> dict[str, float | int]:
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
    }


def row_label(row_index: int, parsed: dict[str, float | int]) -> str:
    return f"row {row_index} B={parsed['B']} T={parsed['T']} K={parsed['K']} V={parsed['V']}"


def is_large_row(parsed: dict[str, float | int]) -> bool:
    return (
        parsed["V"] >= LARGE_VOCAB_THRESHOLD
        and parsed["T"] >= LARGE_STEP_THRESHOLD
        and parsed["K"] >= LARGE_BEAM_THRESHOLD
    )


def evaluate_row(row_index: int, parsed: dict[str, float | int], args: argparse.Namespace) -> tuple[list[str], bool]:
    failures = []
    label = row_label(row_index, parsed)
    if parsed["cpu_cuda"] > args.max_diff:
        failures.append(f"{label}: CPU/CUDA diff {parsed['cpu_cuda']} > {args.max_diff}")
    if parsed["cuda_ref"] > args.max_diff:
        failures.append(f"{label}: CUDA/reference diff {parsed['cuda_ref']} > {args.max_diff}")
    if parsed["peak"] > args.max_peak_mb:
        failures.append(f"{label}: peak CUDA memory {parsed['peak']}MB > {args.max_peak_mb}MB")

    large = is_large_row(parsed)
    if large:
        if parsed["speedup"] < args.min_large_cpu_speedup:
            failures.append(f"{label}: CUDA/CPU speedup {parsed['speedup']} < {args.min_large_cpu_speedup}")
        # cuda_vs_torch_speedup = torch_ms / dbs_cuda_ms. Values < 0.5 mean DBS is >2x slower than torch ref.
        if parsed["torch_speedup"] < 1.0 / args.max_cuda_vs_torch_slowdown:
            failures.append(f"{label}: DBS CUDA is too slow vs torch reference ratio={parsed['torch_speedup']}")
    return failures, large


def report_failures(failures: list[str]) -> int:
    print("performance gate failed:", file=sys.stderr)
    for msg in failures:
        print(" - " + msg, file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()

    path = Path(args.csv_path)
    rows = read_rows(path)
    if rows is None:
        return 2

    failures = []
    large_rows = 0
    for i, row in enumerate(rows, 1):
        try:
            row_failures, is_large = evaluate_row(i, parse_row(row), args)
        except Exception as exc:
            failures.append(f"row {i}: parse error: {exc}")
            continue

        failures.extend(row_failures)
        if is_large:
            large_rows += 1

    if large_rows == 0:
        failures.append("no large CUDA benchmark rows were found")

    if failures:
        return report_failures(failures)

    print(f"performance gate passed for {len(rows)} rows; large rows={large_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
