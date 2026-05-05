#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PYTHON:=python}"
BUILD_CUDA=0
NO_ISOLATION=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) BUILD_CUDA=1 ;;
    --cpu) BUILD_CUDA=0 ;;
    --no-isolation) NO_ISOLATION=1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

CMAKE_BUILD_DIR="$ROOT/build"
if [[ "$BUILD_CUDA" == "1" ]]; then
  CMAKE_BUILD_DIR="$ROOT/build-cuda"
  cmake -S "$ROOT" -B "$CMAKE_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_SHARED=ON -DDBS_BUILD_TESTS=ON -DDBS_ENABLE_CUDA=ON
else
  cmake -S "$ROOT" -B "$CMAKE_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_SHARED=ON -DDBS_BUILD_TESTS=ON
fi
cmake --build "$CMAKE_BUILD_DIR" --parallel

rm -rf "$ROOT/dist" "$ROOT/build_wheel_tmp" "$ROOT"/*.egg-info
"$PYTHON" -m pip install --upgrade build wheel "packaging>=24.2" >/dev/null
BUILD_ARGS=("$ROOT" --wheel)
if [[ "$NO_ISOLATION" == "1" ]]; then
  BUILD_ARGS+=(--no-isolation)
fi
if [[ "$BUILD_CUDA" == "1" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "--cuda requested but nvcc was not found on PATH" >&2
    exit 1
  fi
  DBS_BUILD_TORCH_CUDA=1 "$PYTHON" -m build "${BUILD_ARGS[@]}"
else
  "$PYTHON" -m build "${BUILD_ARGS[@]}"
fi

WHEEL_COUNT=$(find "$ROOT/dist" -name '*.whl' | wc -l | tr -d ' ')
if [[ "$WHEEL_COUNT" -lt 1 ]]; then
  echo "wheel was not produced" >&2
  exit 1
fi
printf 'wheels:\n'
find "$ROOT/dist" -name '*.whl' -print
