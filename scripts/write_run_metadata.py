#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception:
        return None


def first_line(text: str | None) -> str | None:
    if not text:
        return None
    return text.splitlines()[0] if text.splitlines() else text


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_isa() -> dict[str, object]:
    info: dict[str, object] = {"machine": platform.machine()}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("flags") or line.startswith("Features"):
                _, _, value = line.partition(":")
                flags = sorted(value.strip().split())
                info["flags"] = flags
                info["summary"] = {
                    "sse4_2": "sse4_2" in flags,
                    "avx2": "avx2" in flags,
                    "avx512f": "avx512f" in flags,
                    "neon": "asimd" in flags or "neon" in flags,
                }
                break
    return info


def torch_info() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }


def artifact_record(path_text: str) -> dict[str, object]:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--status", default="passed")
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()

    version_path = ROOT / "VERSION"
    git_sha = run_text(["git", "rev-parse", "HEAD"])
    git_status = run_text(["git", "status", "--short"])
    metadata = {
        "schema_version": 1,
        "kind": args.kind,
        "status": args.status,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "version": version_path.read_text(encoding="utf-8").strip() if version_path.exists() else None,
        "git": {
            "sha": git_sha,
            "dirty": bool(git_status),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": cpu_isa(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "tools": {
            "cmake": first_line(run_text(["cmake", "--version"])),
            "cc": first_line(run_text([os.environ.get("CC", "cc"), "--version"])) if shutil.which(os.environ.get("CC", "cc")) else None,
            "cxx": first_line(run_text([os.environ.get("CXX", "c++"), "--version"])) if shutil.which(os.environ.get("CXX", "c++")) else None,
            "nvcc": first_line(run_text(["nvcc", "--version"])) if shutil.which("nvcc") else None,
            "nvidia_smi": run_text(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]) if shutil.which("nvidia-smi") else None,
        },
        "torch": torch_info(),
        "environment": {
            "DBS_REQUIRE_CUDA": os.environ.get("DBS_REQUIRE_CUDA"),
            "DBS_REQUIRE_TORCH": os.environ.get("DBS_REQUIRE_TORCH"),
            "DBS_REQUIRE_LONG_FUZZ": os.environ.get("DBS_REQUIRE_LONG_FUZZ"),
            "DBS_CUDA_LARGE_SCATTER_TEST": os.environ.get("DBS_CUDA_LARGE_SCATTER_TEST"),
            "DBS_BUILD_TORCH_CUDA": os.environ.get("DBS_BUILD_TORCH_CUDA"),
        },
        "artifacts": [artifact_record(p) for p in args.artifact],
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
