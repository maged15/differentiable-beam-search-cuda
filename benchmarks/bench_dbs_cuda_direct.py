import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault("DBS_ENABLE_SCORE_ONLY_FAST_PATH", "1")

import torch
from torch_dbs_extension import DBSOptions, final_scores

torch.manual_seed(1234)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
VALIDATE_INPUTS = int(os.environ.get("DBS_BENCH_VALIDATE_INPUTS", "0"))


def _run_text(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception:
        return None

def cuda_ms(fn, repeats=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats

def cpu_ms(fn, repeats=10, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / repeats

def pytorch_ref_scores(x, beam_size):
    B, T, _, V = x.shape
    scores = torch.full((B, beam_size), -float("inf"), device=x.device, dtype=x.dtype)
    scores[:, 0] = 0.0
    for t in range(T):
        cand = scores[:, :, None] + x[:, t, :, :]
        scores, _ = torch.topk(cand.reshape(B, beam_size * V), beam_size, dim=1)
    return scores

shapes = [
    (1, 8, 4, 32_000),
    (4, 8, 4, 32_000),
    (4, 16, 4, 32_000),
    (4, 16, 8, 32_000),
    (2, 16, 4, 64_000),
    (4, 16, 8, 64_000),
    (1, 16, 8, 128_000),
    (2, 16, 8, 128_000),
]

rows = []
for B, T, K, V in shapes:
    print(f"running B={B} T={T} K={K} V={V}", flush=True)
    x_cpu = torch.randn(B, T, K, V, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    x_cuda = x_cpu.cuda()
    opts = DBSOptions(beam_size=K, eos_token=-1, validate_inputs=VALIDATE_INPUTS)
    torch.cuda.reset_peak_memory_stats()

    cpu_out = torch.stack([final_scores(x_cpu[b], opts).detach() for b in range(B)], dim=0)
    cuda_out = final_scores(x_cuda, opts).detach().cpu()
    torch_ref_out = pytorch_ref_scores(x_cuda, K).detach().cpu()

    cpu_cuda_diff = (cpu_out - cuda_out).abs().max().item()
    cuda_ref_diff = (cuda_out - torch_ref_out).abs().max().item()
    dbs_cuda_ms = cuda_ms(lambda x_cuda=x_cuda, opts=opts: final_scores(x_cuda, opts))
    torch_cuda_ms = cuda_ms(lambda x_cuda=x_cuda, beam_size=K: pytorch_ref_scores(x_cuda, beam_size))
    dbs_cpu_ms = cpu_ms(lambda x_cpu=x_cpu, opts=opts, batch_size=B: [final_scores(x_cpu[b], opts) for b in range(batch_size)])
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    rows.append({
        "B": B, "T": T, "K": K, "V": V,
        "validate_inputs": VALIDATE_INPUTS,
        "dbs_cpu_ms": round(dbs_cpu_ms, 4),
        "dbs_cuda_ms": round(dbs_cuda_ms, 4),
        "torch_cuda_ms": round(torch_cuda_ms, 4),
        "cuda_vs_cpu_speedup": round(dbs_cpu_ms / dbs_cuda_ms, 3) if dbs_cuda_ms > 0 else None,
        "cuda_vs_torch_speedup": round(torch_cuda_ms / dbs_cuda_ms, 3) if dbs_cuda_ms > 0 else None,
        "peak_cuda_mb": round(peak_mb, 2),
        "max_abs_cpu_cuda_diff": cpu_cuda_diff,
        "max_abs_cuda_torch_ref_diff": cuda_ref_diff,
        "pass_cpu_cuda": cpu_cuda_diff <= 1e-5,
        "pass_cuda_torch_ref": cuda_ref_diff <= 1e-5,
    })

csv_path = RESULTS / "bench-dbs-cuda-direct.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

metadata = {
    "version": (ROOT / "VERSION").read_text().strip(),
    "git_sha": _run_text(["git", "rev-parse", "HEAD"]),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "score_only_fast_path": os.environ.get("DBS_ENABLE_SCORE_ONLY_FAST_PATH"),
    "validate_inputs": VALIDATE_INPUTS,
    "rows": len(rows),
    "csv": str(csv_path.relative_to(ROOT)),
}
(RESULTS / "bench-dbs-cuda-direct.metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

for r in rows:
    print(r)

assert all(r["pass_cpu_cuda"] and r["pass_cuda_torch_ref"] for r in rows)
