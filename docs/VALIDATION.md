# Validation gates

A production release must pass the following gates on target hardware.

## CPU gate

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_TESTS=ON -DDBS_BUILD_BENCHMARKS=ON
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
scripts/check_abi.sh build-release/libdbs.so
build-release/dbs_bench 16 8 32768 4 3
```

## CUDA parity gate

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DDBS_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
DBS_BUILD_TORCH_CUDA=1 pip install -e .
python -m pytest python/tests/test_torch_extension.py python/tests/test_cuda_parity.py -q
```

This gate compares CUDA final scores against the CPU golden path for EOS and no-EOS cases. CUDA autograd is disabled until a fused CUDA surrogate-gradient kernel passes the same parity standard.

## Fuzz/sanitizer gate

```bash
DBS_FUZZ_SECONDS=1800 scripts/run_fuzz_campaign.sh
```

Recommended matrix: ASAN+UBSAN, TSAN, MSAN with Clang, libFuzzer for invalid dimensions and malformed constraints, and AFL++ as a separate long-running campaign.

## SIMD gate

Run `dbs_bench` and golden-output tests on scalar-only x86, SSE4.2, AVX2, AVX-512, and ARM64/NEON machines. The acceptance criterion is bit-stable output for deterministic cases and a measured throughput improvement over scalar for the selected kernel.
