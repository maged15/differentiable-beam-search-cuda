# CIFIX8

- Uses `VERSION` as the Python package version source of truth via dynamic `pyproject.toml` metadata.
- Hardens native CPU/CUDA PyTorch extension validation before integer casts.
- Checks sparse-gradient index bounds in native CPU backward.
- Documents and retains opt-in CUDA stream synchronization checks.
- Adds CUDA public API tests for invalid shapes, unsupported options, non-contiguous tensors, and half input promotion.
