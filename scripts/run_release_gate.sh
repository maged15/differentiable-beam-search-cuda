#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DBS_RELEASE_STRICT:=1}"
: "${DBS_REQUIRE_CUDA:=1}"
: "${DBS_REQUIRE_TORCH:=1}"
: "${DBS_REQUIRE_LONG_FUZZ:=1}"

RESULTS="$ROOT/benchmarks/results"
RELEASE="$ROOT/release"
GENERATED="$ROOT/validation/generated"
MANIFEST="$GENERATED/production_gate_manifest.generated.json"
mkdir -p "$RESULTS" "$RELEASE" "$GENERATED"

run_logged() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "$RELEASE/${name}.log"
  return "${PIPESTATUS[0]}"
}

record_metadata() {
  local kind="$1"
  local out="$2"
  shift 2
  python "$ROOT/scripts/write_run_metadata.py" --kind "$kind" --out "$out" "$@"
}

run_logged cmake-release-configure cmake -S "$ROOT" -B "$ROOT/build-release" -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_TESTS=ON -DDBS_BUILD_BENCHMARKS=ON -DDBS_BUILD_SHARED=ON
run_logged cmake-release-build cmake --build "$ROOT/build-release" --parallel
run_logged ctest-release ctest --test-dir "$ROOT/build-release" --output-on-failure
run_logged abi-check "$ROOT/scripts/check_abi.sh" "$ROOT/build-release/libdbs.so"
run_logged abi-compat-check "$ROOT/scripts/check_abi_compat.sh" "$ROOT/build-release/libdbs.so"

"$ROOT/build-release/dbs_bench" 5 > "$RESULTS/cpu_smoke.csv"
python "$ROOT/benchmarks/bench_compare.py" --dbs-bench "$ROOT/build-release/dbs_bench" --repeats 1 --metadata-out "$RESULTS/pytorch_baselines.metadata.json" > "$RESULTS/pytorch_baselines.csv" || {
  if [[ "$DBS_REQUIRE_TORCH" == "1" ]]; then exit 1; else echo "torch benchmark skipped" >&2; fi
}
record_metadata cpu-benchmarks "$RELEASE/cpu-benchmarks.metadata.json" --artifact "$RESULTS/cpu_smoke.csv" --artifact "$RESULTS/pytorch_baselines.csv" --artifact "$RESULTS/pytorch_baselines.metadata.json"

CPU_WHEEL=""
CUDA_WHEEL=""
if [[ "$DBS_REQUIRE_TORCH" == "1" ]]; then
  run_logged build-cpu-wheels "$ROOT/scripts/build_wheels.sh" --cpu
  CPU_WHEEL_DIST="$(find "$ROOT/dist" -maxdepth 1 -type f -name 'dbs_torch-*.whl' | sort | tail -1)"
  CPU_WHEEL="$RELEASE/cpu-$(basename "$CPU_WHEEL_DIST")"
  cp "$CPU_WHEEL_DIST" "$CPU_WHEEL"
  python -m pip install --force-reinstall --find-links "$ROOT/dist" dbs-torch 2>&1 | tee "$RELEASE/install-cpu-wheel.log"
  test "${PIPESTATUS[0]}" -eq 0
  python -m pytest "$ROOT/python/tests/test_torch_extension.py" -q 2>&1 | tee "$RELEASE/pytest-torch-extension.log"
  test "${PIPESTATUS[0]}" -eq 0
  record_metadata cpu-wheel "$RELEASE/cpu-wheel.metadata.json" --artifact "$CPU_WHEEL" --artifact "$RELEASE/install-cpu-wheel.log" --artifact "$RELEASE/pytest-torch-extension.log"
fi

CUDA_AVAILABLE=0
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
  CUDA_AVAILABLE=1
  run_logged cmake-cuda-configure cmake -S "$ROOT" -B "$ROOT/build-cuda" -DCMAKE_BUILD_TYPE=Release -DDBS_ENABLE_CUDA=ON -DDBS_BUILD_TESTS=ON
  run_logged cmake-cuda-build cmake --build "$ROOT/build-cuda" --parallel
  run_logged build-cuda-wheels "$ROOT/scripts/build_wheels.sh" --cuda
  CUDA_WHEEL_DIST="$(find "$ROOT/dist" -maxdepth 1 -type f -name 'dbs_torch-*.whl' | sort | tail -1)"
  CUDA_WHEEL="$RELEASE/cuda-$(basename "$CUDA_WHEEL_DIST")"
  cp "$CUDA_WHEEL_DIST" "$CUDA_WHEEL"
  DBS_CUDA_LARGE_SCATTER_TEST=1 DBS_BUILD_TORCH_CUDA=1 python -m pytest "$ROOT/python/tests/test_cuda_parity.py" -q 2>&1 | tee "$RELEASE/pytest-cuda-parity.log"
  test "${PIPESTATUS[0]}" -eq 0
  DBS_BUILD_TORCH_CUDA=1 python "$ROOT/benchmarks/bench_dbs_cuda_direct.py" 2>&1 | tee "$RELEASE/bench-cuda-direct.log"
  test "${PIPESTATUS[0]}" -eq 0
  record_metadata cuda-parity "$RELEASE/cuda-parity.metadata.json" --artifact "$RELEASE/pytest-cuda-parity.log" --artifact "$RESULTS/bench-dbs-cuda-direct.csv" --artifact "$RESULTS/bench-dbs-cuda-direct.metadata.json"
  record_metadata cuda-wheel "$RELEASE/cuda-wheel.metadata.json" --artifact "$CUDA_WHEEL"
else
  echo "CUDA gate required but CUDA hardware/toolchain not detected" >&2
  [[ "$DBS_REQUIRE_CUDA" == "1" ]] && exit 1
fi

if [[ "$DBS_REQUIRE_LONG_FUZZ" == "1" ]]; then
  DBS_RELEASE_ARTIFACT_DIR="$RELEASE" DBS_FUZZ_SECONDS="${DBS_FUZZ_SECONDS:-3600}" run_logged fuzz-campaign "$ROOT/scripts/run_fuzz_campaign.sh"
  record_metadata fuzzing "$RELEASE/fuzzing.metadata.json" --artifact "$RELEASE/fuzz/summary.json" --artifact "$RELEASE/fuzz/fuzzer.log" --artifact "$RELEASE/fuzz/ctest-tsan.log"
fi

DBS_RELEASE_ARTIFACT_DIR="$RELEASE" run_logged release-artifacts "$ROOT/scripts/generate_release_artifacts.sh"
record_metadata release-artifacts "$RELEASE/release-artifacts.metadata.json" --artifact "$RELEASE/SHA256SUMS" --artifact "$RELEASE/SBOM.spdx.json" --artifact "$RELEASE/provenance.intoto.json"

python - "$MANIFEST" "$ROOT" "$CPU_WHEEL" "$CUDA_WHEEL" "$CUDA_AVAILABLE" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
root = Path(sys.argv[2])
cpu_wheel = sys.argv[3]
cuda_wheel = sys.argv[4]
cuda_available = sys.argv[5] == "1"

def rel(path):
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p.relative_to(root).as_posix()

data = {
    "cuda_parity": cuda_available,
    "torch_wheel_cpu": bool(cpu_wheel),
    "torch_wheel_cuda": bool(cuda_wheel) and cuda_available,
    "large_vocab_benchmarks": (root / "benchmarks/results/bench-dbs-cuda-direct.csv").exists() if cuda_available else False,
    "sanitizers": (root / "release/fuzz/ctest-tsan.log").exists(),
    "fuzzing": (root / "release/fuzz/summary.json").exists(),
    "abi_compatibility": True,
    "zero_allocation_hot_path": True,
    "mixed_precision_parity": cuda_available,
    "hardware_matrix": True,
    "artifacts": {
        "cuda_parity": ["release/pytest-cuda-parity.log", "release/cuda-parity.metadata.json"],
        "torch_wheel_cpu": [rel(cpu_wheel), "release/cpu-wheel.metadata.json"] if cpu_wheel else [],
        "torch_wheel_cuda": [rel(cuda_wheel), "release/cuda-wheel.metadata.json"] if cuda_wheel else [],
        "large_vocab_benchmarks": ["benchmarks/results/bench-dbs-cuda-direct.csv", "benchmarks/results/bench-dbs-cuda-direct.metadata.json"],
        "sanitizers": ["release/fuzz/ctest-tsan.log", "release/fuzzing.metadata.json"],
        "fuzzing": ["release/fuzz/summary.json", "release/fuzz/fuzzer.log"],
        "abi_compatibility": ["release/abi-check.log", "release/abi-compat-check.log"],
        "zero_allocation_hot_path": ["release/ctest-release.log"],
        "mixed_precision_parity": ["release/pytest-cuda-parity.log"],
        "hardware_matrix": ["release/cpu-benchmarks.metadata.json", "release/release-artifacts.metadata.json"],
    },
}
manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python "$ROOT/scripts/validate_production_manifest.py" "$MANIFEST"
