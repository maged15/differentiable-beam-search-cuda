# CUDA validation

CUDA support in the public PyTorch API is hard-forward first. CPU tensors support the full sparse surrogate autograd path; CUDA tensors support final-score forward/parity validation and limited selected-path sparse surrogate backward for `beam_size <= 32`.

Public tensor contract:

- `[T,K,V] -> [K]`
- `[B,T,K,V] -> [B,K]`

The Python wrapper normalizes unbatched CUDA inputs to the native rank-4 kernel path and squeezes the result back to `[K]`. The native CUDA extension also accepts both ranks defensively.

The exact CUDA kernel uses `c10::cuda::CUDAGuard` and PyTorch's current CUDA stream. Set `DBS_CUDA_SYNC_CHECK=1` or `DBS_CUDA_DEBUG_SYNC=1` during validation to synchronize after launches and surface asynchronous device failures at the call site.

Public GitHub-hosted CI does not provide an NVIDIA GPU. CUDA parity is therefore required on self-hosted CUDA runners before stable release, even if hosted CPU CI is green.
