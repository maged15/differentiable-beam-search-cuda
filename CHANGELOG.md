# Changelog

Entries are ordered oldest to newest.

## 0.3.0

- Made `dbs_backward()` sparse by default. Dense gradients are opt-in through `dbs_backward_dense()` and remain protected by `max_dense_gradient_elements`.
- Added model-step callback APIs: `dbs_decode_model_steps()` and `dbs_decode_model_steps_with_workspace()`.
- Added reusable workspace APIs for callback decoding: `dbs_workspace_create()`, `dbs_workspace_reserve()`, and `dbs_workspace_allocated_bytes()`.
- Added variable batch decode with per-example steps, beam sizes, EOS tokens, min lengths, banned tokens, and forced tokens.
- Added advanced constraints: repetition penalty, no-repeat n-gram, and token-filter callback via `DBSAdvancedConstraintsC`.
- Added observability through `DBSStatsC`: selected kernel, sparse/dense backward mode, model-step usage, timing, allocation estimate, result sizes, sparse nnz, and error category.
- Added CPU feature discovery plus AVX2/SSE4.2/NEON helper kernels for dot/softmax normalization paths.
- Added PyTorch C++ extension packaging plus a `torch.autograd.Function` wrapper.
- Added JAX `custom_vjp` host-callback wrapper for CPU validation.
- Expanded C++ tests, benchmark coverage, CI, and docs.

## 0.4.0

- Replaced the CUDA scaffold with real CUDA source kernels for batched hard forward decode and sparse backward scatter.
- Added typed FP16/BF16 decode APIs.
- Added native `torch.ops.dbs` source.
- Added the toy training example, benchmark scaffolds, export map, and extended docs.

## 1.0.0rc1

- Added `make cuda-validate`, `make soak`, `make long-fuzz`, `make perf-gate`, and `make release-artifacts`.
- Added CUDA scheduled/self-hosted benchmark workflow.
- Added fuzz/soak workflow and release artifact workflow.
- Added performance threshold validation for CUDA parity and speed regressions.
- Added soak-test script for determinism and memory-growth checks.
- Added runnable CPU, CUDA, and benchmark reproduction examples.
- Added supported-platforms, known-limitations, and performance-threshold docs.
- Added issue templates, pull request template, contributing guide, code of conduct, and security policy.
- Added extra constraint tests for banned-token and forced-token behavior.

## 1.0.0rc8

- Added cooperative CUDA fast forward kernel API (`dbs_cuda_decode_forward_fast`) alongside the serial correctness CUDA kernel and sparse scatter kernel.
- Added optional CUDA PyTorch extension source and CUDA/CPU parity pytest gate.
- Added allocator counters, deterministic seed recording APIs, and expanded JSON stats.
- Added ABI symbol check, hardware validation script, fuzz campaign script, and package manifest.
- Added CI hooks for hardware validation and sanitizer/fuzzer campaigns.
- Added tests for allocator counters, deterministic seed recording, and JSON stats fields.
- Added exported result observability APIs: EOS count, deterministic-order validation, and JSON result summaries.
- Added production gate manifest validation API and required/template manifests.
- Added fail-closed release gate script covering CPU tests, ABI exact/compat checks, wheels, CUDA parity, benchmarks, and fuzz/sanitizer campaigns.
- Added ABI symbol manifests for v0.5 and v1.0 and backwards-compatible symbol checks.
- Added FP16 decode, dense backward memory-cap, summary JSON, and gate-manifest tests.
- Added fail-closed production approval gate requiring CI-produced evidence only.
- Added independent review attestations for API/ABI, security, numerical correctness, performance, and ML integration.
- Added mandatory release artifacts for signed wheels/libraries, SBOM, provenance, vulnerability/license scans, CUDA raw logs, fuzz/sanitizer artifacts, and soak/SLO artifacts.
- Added frozen supported platform matrix lock and canary/rollback guidance.
- Fixed CUDA public API shape handling at the native extension boundary, including direct native CUDA calls with `[T,K,V]`.
- Rejected unsupported non-default CUDA decoder options instead of silently ignoring them.
- Corrected README C API backward/free ordering.
- Reconfirmed MIT license text and explicit Python/CMake/C ABI version separation.
- Updated docs to distinguish locally validated CPU features from hardware-dependent CUDA/SIMD validation gates.

## 1.0.0

- Hardened AVX-512 sigmoid for very negative inputs and added scalar parity coverage for the denormal-range behavior.
- Fixed the AVX-512 exponential range-reduction constant used by sigmoid/softmax-style gradient helpers.
- Added internal scalar-vs-AVX-512 parity tests for exp, selected softmax weights, relaxed top-k weights, and sigmoid weights.
- Enabled GCC target-attribute builds for AVX2/FMA and SSE4.2 helper paths.
- Added an SSE4.2 vocabulary row scan path and dispatch branch.
- Capped CUDA sparse scatter launches with a grid-stride loop to avoid int grid overflow on large sparse gradients.
- Changed CUDA async finish handling so the non-synchronizing path no longer consumes CUDA error state with `cudaGetLastError()`.
- Merged CUDA fast-kernel EOS carry-forward candidates outside the per-thread local vocabulary lists, and added fast-vs-serial C API parity coverage.
- Documented the variable-batch CUDA log-prob layout as dense `[B, max_steps, max_beam_size, vocab_size]` with max-beam stride.
- Added direct CUDA C API boundary checks for launch grid size and decode element-count overflow.
- Wired public CUDA `final_scores()` through the exact CUDA kernel's `min_length` support.
- CUDA unbatched `[T,K,V]` inputs are normalized in the public Python wrapper before calling the rank-4 native CUDA extension.
- CUDA Python forward now supports `min_length`, rejects NaN/+Inf when `validate_inputs=1`, and exposes an optional `decode()` helper for CUDA token traces.
- CUDA kernels synchronize by default for correctness-oriented status reporting; use `dbs_cuda_set_synchronization(0)` only when async launch semantics are explicitly desired.
- Added explicit CUDA synchronization policy APIs and removed the environment-only async toggle.
- Made CUDA fast-math opt-in for CMake and setup.py builds; correctness/release validation builds leave it disabled.
- Added sparse-gradient index guards to the ctypes and JAX wrappers.
- Vectorized the ctypes wrapper's final-score copy and sparse gradient scatter.
- Hardened ctypes wrapper cleanup paths around native handle/result ownership.
- Added a public CUDA `decode()` helper returning token traces plus final scores.
- Public tests now treat `[B,T,K,V]` with `B=1` as valid and reject only true beam-size/rank/dimension errors.
- Non-CUDA CMake builds now export a stable `dbs::dbs_cuda` stub target.
- Aligned shared-library `SOVERSION` with C ABI `10`, and expanded the metadata gate to check ABI/SOVERSION/license consistency.
- License text, SPDX headers, README, and package metadata consistently use MIT.
- Version metadata uses `VERSION` as the source of truth for Python packaging; `pyproject.toml` declares the version as dynamic.
- CMake library `VERSION` remains `1.0.0`, and shared-library `SOVERSION` follows C ABI `10`.
- Versioning docs explicitly separate Python package version, CMake library version, and C ABI version.
