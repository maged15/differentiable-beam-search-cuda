#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REQUIRED = [
    "cuda_parity", "torch_wheel_cpu", "torch_wheel_cuda", "large_vocab_benchmarks",
    "sanitizers", "fuzzing", "abi_compatibility", "zero_allocation_hot_path",
    "mixed_precision_parity", "hardware_matrix",
]
path = sys.argv[1] if len(sys.argv) > 1 else "validation/production_gate_manifest.required.json"
manifest = Path(path)
repo_root = Path(__file__).resolve().parents[1]
with manifest.open("r", encoding="utf-8") as f:
    data = json.load(f)


def artifact_paths(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if "paths" in value:
            return value["paths"]
        if "path" in value:
            return [value["path"]]
    return []


def resolve_artifact(path_text):
    p = Path(path_text)
    if p.is_absolute():
        return p
    root_candidate = repo_root / p
    if root_candidate.exists():
        return root_candidate
    return manifest.parent / p


failures = []
missing = [k for k in REQUIRED if data.get(k) is not True]
if missing:
    failures.append("missing true evidence flags: " + ", ".join(missing))

artifacts = data.get("artifacts", {})
if not isinstance(artifacts, dict):
    failures.append("artifacts must be an object keyed by evidence flag")
    artifacts = {}

for key in REQUIRED:
    if data.get(key) is not True:
        continue
    paths = artifact_paths(artifacts.get(key))
    if not paths:
        failures.append(f"{key} has no artifact paths")
        continue
    for path_text in paths:
        p = resolve_artifact(path_text)
        if not p.exists():
            failures.append(f"{key} artifact missing: {path_text}")
        elif p.is_file() and p.stat().st_size == 0:
            failures.append(f"{key} artifact is empty: {path_text}")

if failures:
    raise SystemExit("production gate manifest failed:\n- " + "\n- ".join(failures))
print("production gate manifest passed")
