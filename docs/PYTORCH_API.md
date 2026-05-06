# PyTorch API contract

`torch_dbs_extension.final_scores(log_probs, options)` accepts the same public shapes on CPU and CUDA:

- `[T, K, V] -> [K]`
- `[B, T, K, V] -> [B, K]`

`K` must equal `options.beam_size`. All dimensions must be non-empty. Inputs are promoted to FP32 internally; gradients are cast back to the original input dtype where autograd is supported.

## Autograd support

CPU supports sparse surrogate autograd for both public shapes.

CUDA supports hard forward and limited selected-path sparse surrogate backward for `beam_size <= 32` when `dbs_torch_cuda_ext` is built. Larger CUDA beams run scores-only forward and raise on backward.

## CUDA fast path

The exact custom CUDA path is the default. The score-only ATen top-k fast path is opt-in with `DBS_ENABLE_SCORE_ONLY_FAST_PATH=1`; it is intended for no-EOS final-score workloads and should not be used when parent/token trace semantics or model-state reordering are required unless separately validated.


## CUDA option support

CUDA supports the public tensor shapes `[T,K,V] -> [K]` and `[B,T,K,V] -> [B,K]`.
The CUDA extension implements final-score decoding for `beam_size`, `eos_token`, and `min_length`; options that change CPU decoding semantics, such as `length_penalty_alpha`, temperature fields, relaxed-pool sizing, and soft-top-k iteration settings, are rejected on CUDA instead of being silently ignored. CPU tensors continue to accept the full `DBSOptions` tuple and support surrogate autograd.
