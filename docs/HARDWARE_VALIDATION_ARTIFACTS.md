# Hardware validation artifacts

A production/stable release should attach CI-produced artifacts, not local screenshots:

- CUDA wheel build log and clean-install test log
- CUDA parity log for `[T,K,V]` and `[B,T,K,V]` public API shapes
- adversarial tie/near-tie CUDA parity log
- CUDA benchmark CSV from `benchmarks/bench_dbs_cuda_direct.py`
- exact-kernel fallback benchmark/parity log
- ASAN/UBSAN logs
- fuzz campaign seeds, coverage, and crash reproducers
- soak-test JSON with peak memory and determinism results
- CPU ISA benchmark logs for AVX-512, AVX2, scalar x86, and ARM64/NEON where supported

Self-hosted jobs may be skipped in public CI. A release should not claim hardware validation for skipped jobs.
