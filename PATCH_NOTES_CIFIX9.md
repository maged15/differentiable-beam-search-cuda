# cifix9

- Normalized visible version metadata to `1.0.0rc7` while keeping CMake/shared library version `1.0.0` and C ABI version `10`.
- Replaced the CUDA placeholder test in `test_torch_extension.py` with a real optional CUDA parity check when CUDA and the CUDA extension are available.
- Added explicit CUDA public API contract tests for unbatched `[T,K,V]` inputs and unsupported CUDA options.
- Added `docs/CUDA_VALIDATION.md` documenting forward-only CUDA support, current-stream/device semantics, and debug synchronization flags.
