#!/usr/bin/env python3
"""Minimal reproduction for CUDA vs CPU DBS timing."""
import os
import time
import torch
from torch_dbs_extension import DBSOptions, final_scores


def ms_cuda(fn, repeats=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(repeats):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / repeats


def ms_cpu(fn, repeats=10):
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) * 1000 / repeats


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    # Benchmark the explicit score-only fast path. Omit this line to time the
    # exact custom CUDA kernel.
    os.environ.setdefault("DBS_ENABLE_SCORE_ONLY_FAST_PATH", "1")
    torch.manual_seed(1234)
    B, T, K, V = 4, 16, 8, 64000
    x_cpu = torch.randn(B, T, K, V)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    x_cuda = x_cpu.cuda()
    opts = DBSOptions(beam_size=K, eos_token=-1)
    cpu = ms_cpu(lambda: final_scores(x_cpu, opts))
    cuda = ms_cuda(lambda: final_scores(x_cuda, opts))
    print({"B": B, "T": T, "K": K, "V": V, "cpu_ms": round(cpu, 4), "cuda_ms": round(cuda, 4), "speedup": round(cpu / cuda, 3)})


if __name__ == "__main__":
    main()
