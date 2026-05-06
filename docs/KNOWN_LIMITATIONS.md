# Known Limitations

Forward decoding uses hard beam selection, which is discontinuous. The backward API exposes a surrogate gradient. It is useful for experiments but is not a mathematically exact derivative through top-k beam search.

The CUDA backend is correctness-first. EOS-aware decoding is supported, but constraint-heavy decoding still belongs on the CPU path. Treat EOS/constraint performance as validation-oriented until dedicated fused kernels are added.

The CUDA PyTorch extension can use ATen CUDA `topk` for the common no-EOS fast path when `DBS_ENABLE_SCORE_ONLY_FAST_PATH=1` is set. This gives strong practical performance and parity for score-only workloads, but it is not a single fused custom kernel.

Production readiness still requires long fuzz/soak validation, multi-platform CI evidence, signed artifacts, SBOM publication, and performance thresholds on the deployment hardware.


## CUDA fast path semantics

The CUDA extension defaults to the exact custom kernel. The optimized ATen `topk` path is opt-in via `DBS_ENABLE_SCORE_ONLY_FAST_PATH=1` and is intended for no-EOS, rank-aligned, score-only workloads. Do not use the fast path when parent/token trace semantics, EOS carry-forward, or model-state reordering are required unless your workload has its own parity tests.


## Package framing

This is a CPU-autograd/CUDA-limited-backward release. It is not a full CUDA differentiable beam-search training package until CUDA surrogate backward has CPU/GPU parity evidence across the full option matrix and large-beam trace support.
