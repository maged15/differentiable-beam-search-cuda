#!/usr/bin/env python3
import json, sys
REQUIRED = [
    "cuda_parity", "torch_wheel_cpu", "torch_wheel_cuda", "large_vocab_benchmarks",
    "sanitizers", "fuzzing", "abi_compatibility", "zero_allocation_hot_path",
    "mixed_precision_parity", "hardware_matrix",
]
path = sys.argv[1] if len(sys.argv) > 1 else "validation/production_gate_manifest.required.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
missing = [k for k in REQUIRED if data.get(k) is not True]
if missing:
    raise SystemExit("production gate manifest failed: " + ", ".join(missing))
print("production gate manifest passed")
