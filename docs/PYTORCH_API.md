# PyTorch API contract

`torch_dbs_extension.final_scores(log_probs, options)` accepts the same public shapes on CPU and CUDA:

- `[T, K, V] -> [K]`
- `[B, T, K, V] -> [B, K]`

`K` must equal `options.beam_size`. All dimensions must be non-empty. Inputs are promoted to FP32 internally; gradients are cast back to the original input dtype where autograd is supported.

## Autograd support

CPU supports sparse surrogate autograd for both public shapes.

CUDA tensors have the same public `final_scores()` semantics as CPU tensors. Native CUDA kernels accelerate forward decoding only when the requested options are semantically supported; otherwise the wrapper runs the CPU semantic implementation internally and moves the output/gradient back to CUDA.

## CUDA fast path

The exact custom CUDA path is used for native-supported hard-forward options when `dbs_torch_cuda_ext` is available and `beam_size <= 32`. Larger CUDA beam sizes use CPU semantic fallback in the public tensor API to avoid the serial direct-C CUDA correctness kernel. The score-only ATen top-k fast path is opt-in with `DBS_ENABLE_SCORE_ONLY_FAST_PATH=1`; it is intended for no-EOS final-score workloads and should not be used when parent/token trace semantics or model-state reordering are required unless separately validated.


## CUDA option support

CUDA supports the public tensor shapes `[T,K,V] -> [K]` and `[B,T,K,V] -> [B,K]`.
The public CUDA tensor API accepts the full `DBSOptions` tuple. Native CUDA forward handles `beam_size <= 32`, `eos_token`, `min_length`, and input validation; options that require CPU-only semantics, such as `length_penalty_alpha`, temperature fields, relaxed-pool sizing, and soft-top-k iteration settings, use CPU semantic fallback instead of being silently ignored. The debugging-only `decode()` token-trace API remains native-CUDA-only and fail-closed for unsupported native options.
