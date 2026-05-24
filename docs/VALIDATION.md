# Validation

Use these checks to validate changes on target hardware. They are not a production certification system.

## CPU checks

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_TESTS=ON -DDBS_BUILD_BENCHMARKS=ON
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
scripts/check_abi.sh build-release/libdbs.so
build-release/dbs_bench 16 8 32768 4 3
```

## CUDA parity checks

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DDBS_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
DBS_BUILD_TORCH_CUDA=1 pip install -e .
python -m pytest python/tests/test_torch_extension.py python/tests/test_cuda_parity.py -q
```

These tests compare CUDA final scores against the CPU golden path for EOS and no-EOS cases.

## Fuzz/sanitizer checks

```bash
DBS_FUZZ_SECONDS=1800 scripts/run_fuzz_campaign.sh
```

Recommended matrix: ASAN+UBSAN, TSAN, MSAN with Clang, libFuzzer for invalid dimensions and malformed constraints, and AFL++ as a separate long-running campaign.

## SIMD checks

Run `dbs_bench` and golden-output tests on scalar-only x86, SSE4.2, AVX2, AVX-512, and ARM64/NEON machines. The acceptance criterion is bit-stable output for deterministic cases and a measured throughput improvement over scalar for the selected kernel.
