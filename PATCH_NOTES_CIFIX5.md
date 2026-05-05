# cifix5 patch notes

This patch addresses static review findings after `1.0.0rc7`:

- Confirms and tests the public PyTorch tensor API contract: `[T,K,V] -> [K]` and `[B,T,K,V] -> [B,K]` on CPU and CUDA.
- Adds stricter Python/C++ dimension checks, non-empty shape checks, `beam_size == K`, and `INT_MAX` bounds before native casts.
- Adds optional custom CUDA launch synchronization checks via `DBS_CUDA_SYNC_CHECK=1` or `DBS_CUDA_DEBUG_SYNC=1`.
- Exports a stable `dbs::dbs_cuda` target for non-CUDA CMake builds by installing a stub implementation under the same target name.
- Reconciles README/license text with the MIT license and adds SPDX headers to key source files.
- Documents versioning as three explicit surfaces: Python package prerelease `1.0.0rc7`, CMake/shared-library version `1.0.0`, and C ABI version `10`.
- Adds packaging notes for CPU wheels, CUDA wheels, and controlled PyTorch/CUDA environments.
