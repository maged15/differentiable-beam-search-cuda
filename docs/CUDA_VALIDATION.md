# CUDA validation

CUDA support in the public PyTorch API is semantic-parity first. CPU tensors support the sparse surrogate autograd path; CUDA tensors use native forward where equivalent and CPU semantic fallback for unsupported native options/backward, returning CUDA outputs and gradients.

Public tensor contract:

- `[T,K,V] -> [K]`
- `[B,T,K,V] -> [B,K]`

The Python wrapper normalizes unbatched CUDA inputs to the native rank-4 kernel path and squeezes the result back to `[K]`. The native CUDA extension also accepts both ranks defensively.

The exact CUDA kernel uses `c10::cuda::CUDAGuard` and PyTorch's current CUDA stream. Set `DBS_CUDA_SYNC_CHECK=1` or `DBS_CUDA_DEBUG_SYNC=1` during validation to synchronize after launches and surface asynchronous device failures at the call site.

Public GitHub-hosted CI does not provide an NVIDIA GPU. CUDA parity is therefore required on self-hosted CUDA runners before publishing CUDA performance or hardware-support claims, even if hosted CPU CI is green.
