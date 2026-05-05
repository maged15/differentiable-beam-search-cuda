#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DBS_REQUIRE_CUDA:=0}"
: "${DBS_REQUIRE_SIMD_BENCH:=0}"
cmake -S "$ROOT" -B "$ROOT/build-release" -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_TESTS=ON -DDBS_BUILD_BENCHMARKS=ON -DDBS_BUILD_SHARED=ON
cmake --build "$ROOT/build-release" --parallel
ctest --test-dir "$ROOT/build-release" --output-on-failure
"$ROOT/scripts/check_abi.sh" "$ROOT/build-release/libdbs.so"
"$ROOT/scripts/check_abi_compat.sh" "$ROOT/build-release/libdbs.so"
"$ROOT/build-release/dbs_bench" 16 8 32768 4 3 > "$ROOT/benchmarks/results/hardware_smoke.csv"
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
  cmake -S "$ROOT" -B "$ROOT/build-cuda" -DCMAKE_BUILD_TYPE=Release -DDBS_ENABLE_CUDA=ON -DDBS_BUILD_TESTS=ON
  cmake --build "$ROOT/build-cuda" --parallel
  DBS_BUILD_TORCH_CUDA=1 python -m pip install -e "$ROOT"
  python -m pytest "$ROOT/python/tests/test_torch_extension.py" "$ROOT/python/tests/test_cuda_parity.py" -q
else
  echo "CUDA hardware/toolchain not detected" >&2
  [[ "$DBS_REQUIRE_CUDA" == "1" ]] && exit 1
fi
if [[ "$DBS_REQUIRE_SIMD_BENCH" == "1" ]]; then
  python "$ROOT/benchmarks/bench_compare.py" --dbs-bench "$ROOT/build-release/dbs_bench" --repeats 3 > "$ROOT/benchmarks/results/simd_baseline.csv"
fi
