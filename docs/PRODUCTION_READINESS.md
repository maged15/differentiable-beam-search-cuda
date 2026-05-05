# Production readiness gate

v1.0.0rc9 is designed to fail closed. A production release requires all of the following evidence to be true in `validation/production_gate_manifest.required.json` and backed by logs:

- CUDA parity on real NVIDIA hardware for forward decode, EOS, constraints, variable batch lengths, sparse scatter, FP16/BF16, deterministic ties, and golden outputs.
- CPU and CUDA PyTorch wheels built and tested with native autograd and an end-to-end training loop.
- Large-vocabulary benchmarks at `V=32768`, `65536`, and `131072` with `K=4..32`, CPU/GPU split, sparse/dense backward, and PyTorch/Hugging Face comparisons.
- ASAN, UBSAN, TSAN, MSAN where available, plus libFuzzer/AFL runs for malformed constraints, NaNs, infinities, overflow sizes, ties, EOS, and huge dimensions.
- ABI exact check for the current release and backwards-compatible symbol check against the previous public symbol manifest.
- Zero-allocation hot-path evidence from allocator counters and repeated decode/backward loops.
- Hardware matrix logs for Linux, macOS, Windows, AVX-512, AVX2, non-AVX x86, ARM64/NEON, and CUDA.

Run:

```bash
DBS_RELEASE_STRICT=1 DBS_REQUIRE_CUDA=1 DBS_REQUIRE_TORCH=1 DBS_REQUIRE_LONG_FUZZ=1 ./scripts/run_release_gate.sh
```

The script exits non-zero if required hardware/toolchains or validation results are missing.
