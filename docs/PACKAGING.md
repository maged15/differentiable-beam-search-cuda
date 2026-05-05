# Python packaging

The Python extension is packaged as `dbs-torch`.

## CPU wheel

The default build produces a CPU wheel with the native C++/PyTorch extension:

```bash
python -m pip install --upgrade build wheel setuptools packaging
python -m build --wheel --no-isolation
```

## CUDA wheel

CUDA wheels are intentionally opt-in because they must match the local CUDA toolkit, GPU architecture, and PyTorch CUDA ABI:

```bash
DBS_BUILD_TORCH_CUDA=1 python -m build --wheel --no-isolation
```

Use `--no-isolation` when building in a controlled PyTorch environment so the extension compiles against the same `torch` installation that will import it. For release distribution, publish CPU wheels and CUDA wheels as separate artifacts with explicit CUDA/PyTorch/toolkit metadata.

## Public tensor contract

The public Python API accepts both unbatched and batched tensors on CPU and CUDA:

- `[T,K,V] -> [K]`
- `[B,T,K,V] -> [B,K]`

CUDA autograd backward is not implemented. CUDA tensors are forward-only; CPU tensors support the sparse surrogate backward.
