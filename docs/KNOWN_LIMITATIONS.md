# Known Limitations

Forward decoding uses hard beam selection, which is discontinuous. The backward API exposes a surrogate gradient. It is useful for experiments but is not a mathematically exact derivative through top-k beam search.

The optimized CUDA fast path is for `eos_token = -1` no-EOS scoring. EOS-aware CUDA decoding and constraint-heavy decoding use the exact fallback path or CPU behavior depending on the API. Treat EOS/constraint performance as correctness-first until dedicated fused kernels are added.

The CUDA PyTorch extension uses ATen CUDA `topk` for the common no-EOS fast path. This gives strong practical performance and parity, but it is not yet a single fused custom kernel.

Production readiness still requires long fuzz/soak validation, multi-platform CI evidence, signed artifacts, SBOM publication, and performance thresholds on the deployment hardware.


## CUDA fast path semantics

The CUDA extension defaults to the exact custom kernel. The optimized ATen `topk` path is opt-in via `DBS_ENABLE_SCORE_ONLY_FAST_PATH=1` and is intended for no-EOS, rank-aligned, score-only workloads. Do not use the fast path when parent/token trace semantics, EOS carry-forward, or model-state reordering are required unless your workload has its own parity tests.


## Package framing

This is a CPU-autograd/CUDA-forward release candidate. It is not a full CUDA differentiable beam-search training package until CUDA sparse surrogate backward is implemented and validated.
