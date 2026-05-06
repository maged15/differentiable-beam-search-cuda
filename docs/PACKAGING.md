# Python packaging

The Python extension is packaged as `dbs-torch`.

## CPU wheel

The default build produces a CPU wheel with the native C++/PyTorch extension. This extension is self-contained because it compiles the C ABI implementation directly into `dbs_torch_ext`.

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

CUDA tensors support the same public `final_scores()` options and gradients as CPU tensors. Native CUDA kernels accelerate supported hard-forward cases; unsupported native CUDA options and CUDA backward use CPU semantic fallback internally and return CUDA tensors. CPU tensors support the sparse surrogate backward across the full CPU option set.

## ctypes wrapper

`python/torch_dbs.py` and `python/jax_dbs.py` are ctypes validation wrappers. They load `libdbs.so` from `DBS_LIBRARY` or the platform loader path, so pip users should prefer `torch_dbs_extension` unless they also install or build the shared C library and configure the loader path explicitly.
