# Implementation Status For v1.0.0

Status: v1.0.0 research release. Production certification still depends on the release gate and hardware validation evidence described in `docs/PRODUCTION_READINESS.md`.

Implemented and locally CPU-validated:

- Sparse backward remains the default CPU path; dense backward is explicit and capped.
- CPU C ABI, deterministic tie-breaking, batch decode, variable batch decode, typed FP32/FP16/BF16 decode, model-step callback decode, reusable workspace APIs, advanced constraints, statistics JSON, allocator counters, and deterministic seed recording.
- C++ tests for sparse gradients, EOS/ties, variable batch shape, constraints, mixed precision conversion, allocator counters, workspace reuse, and golden outputs.
- ABI exact and compatibility symbol checks.
- Benchmark harnesses for DBS, PyTorch greedy/top-k beam, optional Hugging Face generate, and large-vocabulary dimensions.

Implemented with hardware-gated validation:

- CUDA serial correctness kernel, cooperative CUDA fast kernel for `K <= 32`, sparse scatter kernel, and sparse surrogate backward builder.
- Optional CUDA PyTorch extension build path with CUDA forward and surrogate backward.
- CUDA/CPU parity pytest suite and self-hosted CUDA smoke workflow.
- SIMD dispatch paths for AVX-512, AVX2, SSE4.2, and NEON where supported by compiler/runtime hardware.

Not production-certified until release evidence is present:

- CUDA parity and benchmark logs on the target NVIDIA hardware.
- SIMD parity/throughput logs on target CPUs.
- Sanitizer and fuzzing campaign artifacts.
- Wheel provenance, checksums, and release artifacts under the ignored `release/` directory.
