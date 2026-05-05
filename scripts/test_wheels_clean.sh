#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PYTHON:=python3}"
REQUIRE_CUDA=0
ISOLATED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) REQUIRE_CUDA=1 ;;
    --cpu) REQUIRE_CUDA=0 ;;
    --isolated|--clean) ISOLATED=1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
WHEEL="$(find "$ROOT/dist" -name '*.whl' | sort | tail -1)"
if [[ -z "$WHEEL" ]]; then
  echo "no wheel found in $ROOT/dist" >&2
  exit 1
fi

run_smoke() {
  python -m pip install --force-reinstall --no-deps "$WHEEL"
  python - <<'PY'
import torch
import torch_dbs_extension as dbs
x = torch.randn(3, 2, 8, dtype=torch.float32, requires_grad=True)
y = dbs.final_scores(x, dbs.DBSOptions(beam_size=2, eos_token=-1)).sum()
y.backward()
assert x.grad is not None and x.grad.shape == x.shape
print('CPU wheel smoke test passed:', y.detach().cpu().tolist())
PY
  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    python - <<'PY'
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is false'
import dbs_torch_cuda_ext as cuda_ext
assert cuda_ext.cuda_available(), 'dbs CUDA extension reports unavailable'
import torch_dbs_extension as dbs
x = torch.randn(1, 3, 2, 16, device='cuda', dtype=torch.float32)
y = dbs.final_scores(x, dbs.DBSOptions(beam_size=2, eos_token=-1))
assert y.is_cuda and y.shape == (1, 2)
print('CUDA wheel smoke test passed:', y.detach().cpu().tolist())
PY
  fi
}

if [[ "$ISOLATED" == "1" ]]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  "$PYTHON" -m venv "$TMP/env"
  source "$TMP/env/bin/activate"
  python -m pip install --upgrade pip wheel setuptools "packaging>=24.2" >/dev/null
  TORCH_SPEC="$($PYTHON - <<'PY'
try:
    import torch
    print('torch==' + torch.__version__.split('+')[0])
except Exception:
    print('torch')
PY
)"
  python -m pip install "$TORCH_SPEC" >/dev/null
  run_smoke
else
  run_smoke
fi
