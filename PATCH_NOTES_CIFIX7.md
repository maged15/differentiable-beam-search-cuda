# cifix7

- Explicitly normalizes CUDA `[T,K,V]` inputs in `torch_dbs_extension.py` before calling the native rank-4 CUDA extension path.
- Keeps the native CUDA binding accepting both `[T,K,V]` and `[B,T,K,V]` with `INT_MAX` checks as a defensive API boundary.
- Adds build-tree `dbs::dbs` and `dbs::dbs_cuda` aliases and retains installed/exported `dbs_cuda` stub target for non-CUDA CMake builds.
- Documents async CUDA launch error behavior and `DBS_CUDA_SYNC_CHECK`/`DBS_CUDA_DEBUG_SYNC`.
- Normalizes release metadata to Python `1.0.0rc7`, CMake library `1.0.0`, C ABI `10`.
