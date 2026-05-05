# CUDA validation

CUDA support is forward-only in the public PyTorch API. CPU tensors support surrogate autograd; CUDA tensors support final-score forward/parity validation only.

Public tensor contract:

- `[T,K,V] -> [K]`
- `[B,T,K,V] -> [B,K]`

The Python wrapper normalizes unbatched CUDA inputs to the native rank-4 kernel path and squeezes the result back to `[K]`. The native CUDA extension also accepts both ranks defensively.

The exact CUDA kernel uses `c10::cuda::CUDAGuard` and PyTorch's current CUDA stream. CUDA calls synchronize that stream by default so device-side failures surface at the call site. C/CUDA callers can opt into asynchronous launch semantics with `dbs_cuda_set_synchronization(0)`; `DBS_CUDA_SYNC_CHECK=1` or `DBS_CUDA_DEBUG_SYNC=1` still forces synchronization in validation runs.

With `validate_inputs=1`, the public CUDA wrapper rejects NaN and `+Inf` log-probabilities to match the CPU validation contract; `-Inf` remains allowed as an impossible token score. Public CUDA `final_scores()` supports `beam_size`, `eos_token`, and `min_length`. Other CPU shaping/surrogate options remain rejected on CUDA instead of being silently ignored.

Public GitHub-hosted CI does not provide an NVIDIA GPU. CUDA parity is therefore required on self-hosted CUDA runners before stable release, even if hosted CPU CI is green.
