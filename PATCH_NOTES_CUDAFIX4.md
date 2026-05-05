# CUDAFIX4: Fast no-EOS CUDA forward path

This patch fixes the observed slow `dbs_forward_fast_kernel` benchmark path for the common `eos_token < 0` case used in large-vocabulary training/benchmark workloads.

## Changes

- `dbs_torch_cuda_ext.final_scores_forward_cuda()` now routes no-EOS CUDA tensors through ATen CUDA `topk` with DBS-correct initialization (`beam 0 = 0`, all other beams `-inf`).
- The exact custom CUDA kernel remains available for EOS/tie-sensitive validation and can be forced with `DBS_FORCE_EXACT_CUDA_KERNEL=1`.
- Added `final_scores_forward_cuda_aten_topk()` and `final_scores_forward_cuda_exact_kernel()` symbols for A/B profiling.
- Added `benchmarks/bench_dbs_cuda_direct.py` using a corrected PyTorch reference with DBS initialization semantics.

## Expected result

The direct CUDA benchmark should move from the old custom-kernel timings toward the PyTorch CUDA `topk` baseline while preserving CPU/CUDA final-score parity for `eos_token=-1`.

## Remaining work

This is a performance fix for the PyTorch CUDA extension path, not a replacement for a future fused EOS-aware custom kernel. The C ABI `dbs_cuda_decode_forward_fast()` still exposes the exact custom kernel.
