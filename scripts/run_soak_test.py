#!/usr/bin/env python3
"""Repeated decode/backward soak test for memory growth and determinism."""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from torch_dbs_extension import DBSOptions, final_scores


def allocated(device: str) -> int:
    if device == "cuda":
        torch.cuda.synchronize()
        return int(torch.cuda.memory_allocated())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--iterations", type=int, default=int(os.environ.get("DBS_SOAK_ITERATIONS", "1000")))
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--V", type=int, default=32000)
    ap.add_argument("--max-growth-mb", type=float, default=float(os.environ.get("DBS_SOAK_MAX_GROWTH_MB", "64")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    torch.manual_seed(20260505)
    x = torch.randn(args.B, args.T, args.K, args.V, device=args.device)
    x = torch.log_softmax(x, dim=-1)
    opts = DBSOptions(beam_size=args.K, eos_token=-1)

    y0 = final_scores(x, opts).detach().cpu()
    base_mem = allocated(args.device)
    start = time.perf_counter()
    max_mem = base_mem

    for i in range(args.iterations):
        y = final_scores(x, opts).detach().cpu()
        if not torch.equal(y, y0):
            raise AssertionError(f"non-deterministic output at iteration {i}")
        max_mem = max(max_mem, allocated(args.device))

    elapsed = time.perf_counter() - start
    growth_mb = (max_mem - base_mem) / 1024 / 1024
    result = {
        "device": args.device,
        "iterations": args.iterations,
        "B": args.B,
        "T": args.T,
        "K": args.K,
        "V": args.V,
        "elapsed_s": round(elapsed, 4),
        "avg_ms": round(elapsed * 1000 / args.iterations, 6),
        "base_mem_bytes": base_mem,
        "max_mem_bytes": max_mem,
        "growth_mb": round(growth_mb, 4),
        "deterministic": True,
        "pass": growth_mb <= args.max_growth_mb,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["pass"]:
        raise SystemExit(f"memory growth {growth_mb:.2f}MB exceeds {args.max_growth_mb:.2f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
