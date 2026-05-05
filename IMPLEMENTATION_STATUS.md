# Implementation status for v1.0.0rc7

Status: late pre-production validation package.

Implemented and locally CPU-validated:
- Sparse backward remains the default path; dense backward is explicit and capped.
- CPU C ABI, deterministic tie-breaking, batch decode, variable batch decode, typed FP32/FP16/BF16 decode, model-step callback decode, reusable workspace APIs, advanced constraints, statistics JSON, allocator counters, and deterministic seed recording.
- C++ tests for sparse gradients, EOS/ties, variable batch shape, constraints, mixed precision conversion, allocator counters, workspace reuse, and golden outputs.
- ABI symbol check script and Linux export map.
- Benchmark harnesses for DBS, PyTorch greedy/top-k beam, optional Hugging Face generate, and large vocabulary dimensions.

Implemented as source but requiring hardware validation before release claims:
- CUDA serial correctness kernel, cooperative CUDA fast kernel for K <= 32, and sparse scatter kernel.
- Optional CUDA PyTorch extension build path.
- CUDA/CPU parity pytest suite.
- Distributed/self-hosted CI definitions for CUDA, ARM64/NEON, AVX2/AVX-512, macOS, Windows, and sanitizer/fuzzer campaigns.

Not yet production-certified:
- CUDA kernels have not been runtime-validated in this environment on NVIDIA hardware.
- CUDA backward through PyTorch autograd is intentionally blocked until sparse-gradient parity is proven.
- SIMD paths require measured hardware parity/throughput on the target CPUs.
- ABI compatibility needs continuous checks across released versions, not a one-time symbol diff.
