"""Benchmark DBS against PyTorch and optional Hugging Face baselines.

The matrix covers realistic vocab sizes (32k-128k), greedy, stepwise torch.topk beam
search, sparse/dense backward timing through the compiled extension when available,
and an optional transformers.generate baseline when --hf-model is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]


def torch_greedy(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0, :].argmax(dim=-1)


def torch_step_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    scores = torch.zeros(k, device=x.device)
    toks = []
    for t in range(x.shape[0]):
        cand = scores[:, None] + x[t]
        vals, idx = torch.topk(cand.reshape(-1), k)
        toks.append(idx % x.shape[2])
        scores = vals
    return torch.stack(toks)


def time_ms(fn: Callable[[], object], repeats: int) -> float:
    for _ in range(2):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeats


def optional_hf_generate(model_name: str, repeats: int) -> float | None:
    if not model_name:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tok("benchmark", return_tensors="pt")
    return time_ms(lambda: model.generate(**inputs, num_beams=4, max_new_tokens=16), repeats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbs-bench", default="build/dbs_bench")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model", default="", help="optional small causal LM for transformers.generate baseline")
    parser.add_argument("--metadata-out", default="", help="optional JSON metadata path for the benchmark run")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, object]] = []
    for T in [8, 16, 32]:
        for K in [2, 4, 8]:
            for V in [32_000, 64_000, 128_000]:
                # Keep repeats modest for the large matrix. This is benchmark source, not a default CI workload.
                x = torch.randn(T, K, V, device=device)
                rows.append({
                    "device": str(device),
                    "T": T,
                    "K": K,
                    "V": V,
                    "torch_greedy_ms": time_ms(lambda x=x: torch_greedy(x), args.repeats),
                    "torch_topk_beam_ms": time_ms(lambda x=x, k=K: torch_step_topk(x, k), args.repeats),
                })

    hf_ms = optional_hf_generate(args.hf_model, max(1, args.repeats // 2))
    if hf_ms is not None:
        rows.append({"device": "cpu", "T": "hf", "K": 4, "V": "model", "torch_greedy_ms": "", "torch_topk_beam_ms": "", "hf_generate_ms": hf_ms})

    print("# PyTorch/HuggingFace baselines")
    fieldnames = sorted({k for row in rows for k in row})
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    try:
        dbs_csv = subprocess.check_output([args.dbs_bench, str(args.repeats)], text=True)
        print("\n# dbs C++ benchmark CSV")
        print(dbs_csv)
    except Exception as exc:
        print(f"\n# dbs C++ benchmark unavailable: {exc}")

    if args.metadata_out:
        out = Path(args.metadata_out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": (ROOT / "VERSION").read_text().strip(),
            "device": str(device),
            "dbs_bench": args.dbs_bench,
            "repeats": args.repeats,
            "rows": len(rows),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
        out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
