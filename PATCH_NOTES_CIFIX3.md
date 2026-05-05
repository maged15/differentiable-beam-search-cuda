# CIFIX3

- Fixed CUDA PyTorch exact-kernel stream/device semantics with `CUDAGuard` and PyTorch current CUDA stream.
- Made the score-only ATen CUDA fast path explicit opt-in via `DBS_ENABLE_SCORE_ONLY_FAST_PATH=1`; exact kernel is now the default.
- Added adversarial CUDA parity tests for ties/near-ties and a current-stream test.
- Hardened CPU PyTorch backward shape checks and sparse-index bounds validation.
- Fixed Windows static/shared export definitions with `DBS_STATIC`, `DBS_BUILD_SHARED`, and `DBS_COMPILING_LIBRARY`.
- Synced README ABI version with `DBS_ABI_VERSION=10`.
