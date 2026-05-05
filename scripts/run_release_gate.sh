#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DBS_RELEASE_STRICT:=1}"
: "${DBS_REQUIRE_CUDA:=1}"
: "${DBS_REQUIRE_TORCH:=1}"
: "${DBS_REQUIRE_LONG_FUZZ:=1}"
cmake -S "$ROOT" -B "$ROOT/build-release" -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_TESTS=ON -DDBS_BUILD_BENCHMARKS=ON -DDBS_BUILD_SHARED=ON
cmake --build "$ROOT/build-release" --parallel
ctest --test-dir "$ROOT/build-release" --output-on-failure
"$ROOT/scripts/check_abi.sh" "$ROOT/build-release/libdbs.so"
"$ROOT/scripts/check_abi_compat.sh" "$ROOT/build-release/libdbs.so"
"$ROOT/build-release/dbs_bench" 5 > "$ROOT/benchmarks/results/cpu_smoke.csv"
python "$ROOT/benchmarks/bench_compare.py" --dbs-bench "$ROOT/build-release/dbs_bench" --repeats 1 > "$ROOT/benchmarks/results/pytorch_baselines.csv" || {
  if [[ "$DBS_REQUIRE_TORCH" == "1" ]]; then exit 1; else echo "torch benchmark skipped" >&2; fi
}
if [[ "$DBS_REQUIRE_TORCH" == "1" ]]; then
  "$ROOT/scripts/build_wheels.sh"
  python -m pip install --force-reinstall --find-links "$ROOT/dist" dbs-torch
  python -m pytest "$ROOT/python/tests/test_torch_extension.py" -q
fi
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
  cmake -S "$ROOT" -B "$ROOT/build-cuda" -DCMAKE_BUILD_TYPE=Release -DDBS_ENABLE_CUDA=ON -DDBS_BUILD_TESTS=ON
  cmake --build "$ROOT/build-cuda" --parallel
  DBS_BUILD_TORCH_CUDA=1 "$ROOT/scripts/build_wheels.sh"
  DBS_BUILD_TORCH_CUDA=1 python -m pytest "$ROOT/python/tests/test_cuda_parity.py" -q
else
  echo "CUDA gate required but CUDA hardware/toolchain not detected" >&2
  [[ "$DBS_REQUIRE_CUDA" == "1" ]] && exit 1
fi
if [[ "$DBS_REQUIRE_LONG_FUZZ" == "1" ]]; then
  DBS_FUZZ_SECONDS="${DBS_FUZZ_SECONDS:-3600}" "$ROOT/scripts/run_fuzz_campaign.sh"
fi
python "$ROOT/scripts/validate_production_manifest.py" "$ROOT/validation/production_gate_manifest.required.json"
