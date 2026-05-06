# Changelog

## 1.0.0

- Wired public CUDA `final_scores()` through the exact CUDA kernel's `min_length` support.
- Added CUDA NaN/+Inf validation when `validate_inputs=1` to match CPU validation semantics.
- Made setup.py CUDA fast-math opt-in through `DBS_CUDA_USE_FAST_MATH=1`.
- Added sparse-gradient index guards to the ctypes and JAX wrappers.
- Added a public CUDA `decode()` helper returning token traces plus final scores.
- Added direct CUDA C API boundary checks for launch grid size and decode element-count overflow.
- Added explicit CUDA synchronization policy APIs and removed the environment-only async toggle.
- Hardened ctypes wrapper cleanup paths around native handle/result ownership.
- Fixed the AVX-512 exponential range-reduction constant used by sigmoid/softmax-style gradient helpers.
- Added internal scalar-vs-AVX-512 parity coverage for exp, selected softmax weights, and relaxed top-k weights.
- Enabled GCC target-attribute builds for AVX2/FMA and SSE4.2 helper paths.
- Made CUDA sparse backward scatter use a capped grid-stride launch to avoid grid-size truncation for large `nnz`.
- Aligned shared-library `SOVERSION` with C ABI `10` and expanded metadata checks for ABI/SOVERSION/license consistency.
- Added package metadata for MIT license/readme and documented the ctypes wrapper's shared-library requirement.

## 1.0.0rc8

- Fixed CUDA public API shape handling at the native extension boundary, including direct native CUDA calls with `[T,K,V]`.
- Rejected unsupported non-default CUDA decoder options instead of silently ignoring them.
- Corrected README C API backward/free ordering.
- Reconfirmed MIT license text and explicit Python/CMake/C ABI version separation.

## 1.0.0rc8

- Added cooperative CUDA fast forward kernel API (`dbs_cuda_decode_forward_fast`) alongside the serial correctness CUDA kernel and sparse scatter kernel.
- Added optional CUDA PyTorch extension source and CUDA/CPU parity pytest gate.
- Added allocator counters, deterministic seed recording APIs, and expanded JSON stats.
- Added ABI symbol check, hardware validation script, fuzz campaign script, and package manifest.
- Added CI hooks for hardware validation and sanitizer/fuzzer campaigns.
- Added tests for allocator counters, deterministic seed recording, and JSON stats fields.
- Updated docs to distinguish locally validated CPU features from hardware-dependent CUDA/SIMD validation gates.

## 0.4.0

- Added CUDA source kernels, AVX2/SSE4.2/NEON helper kernels, typed FP16/BF16 decode APIs, native `torch.ops.dbs` source, toy training example, benchmark scaffolds, export map, and extended docs.

## 0.3.0

- Added sparse-default backward, model-step callback API, variable batch decode, reusable workspace API, advanced constraints, observability stats, CPU feature reporting, PyTorch/JAX wrapper files, expanded tests, and CI/docs.

## 1.0.0rc8
- Added exported result observability APIs: EOS count, deterministic-order validation, and JSON result summaries.
- Added production gate manifest validation API and required/template manifests.
- Added fail-closed release gate script covering CPU tests, ABI exact/compat checks, wheels, CUDA parity, benchmarks, and fuzz/sanitizer campaigns.
- Added ABI symbol manifests for v0.5 and v1.0 and backwards-compatible symbol checks.
- Added FP16 decode, dense backward memory-cap, summary JSON, and gate-manifest tests.

## 1.0.0rc8

- Added fail-closed production approval gate requiring CI-produced evidence only.
- Added independent review attestations for API/ABI, security, numerical correctness, performance, and ML integration.
- Added mandatory release artifacts for signed wheels/libraries, SBOM, provenance, vulnerability/license scans, CUDA raw logs, fuzz/sanitizer artifacts, and soak/SLO artifacts.
- Added frozen supported platform matrix lock and canary/rollback guidance.

## v1.0.0-rc1 validation polish

- Added `make cuda-validate`, `make soak`, `make long-fuzz`, `make perf-gate`, and `make release-artifacts`.
- Added CUDA scheduled/self-hosted benchmark workflow.
- Added fuzz/soak workflow and release artifact workflow.
- Added performance threshold validation for CUDA parity and speed regressions.
- Added soak-test script for determinism and memory-growth checks.
- Added runnable CPU, CUDA, and benchmark reproduction examples.
- Added supported-platforms, known-limitations, and performance-threshold docs.
- Added issue templates, pull request template, contributing guide, code of conduct, and security policy.
- Added extra constraint tests for banned-token and forced-token behavior.
