# v1.0 Production Approval Requirements

This candidate is not approved for production unless `scripts/run_v10_production_gate.sh --strict` passes on the supported production matrix with no skipped targets.

Required evidence must be produced by CI and attached to the release report:

- CPU and CUDA PyTorch wheels built by CI, then installed and tested in clean containers.
- CUDA parity against immutable checked-in golden fixtures, including raw logs, output hashes, and tolerance metadata.
- 24–72 hour fuzz/sanitizer campaign artifacts: ASAN, UBSAN, TSAN, MSAN where supported, libFuzzer/AFL seeds, coverage, logs, and crash reproducers.
- Soak and SLO tests under deployment-like load, including p50/p95/p99 latency, max memory, allocation counts, fallback rates, CUDA/SIMD timing, and error categories.
- Signed release artifacts: wheels, shared libraries, source tarball, SBOM, checksum manifest, vulnerability scan, license scan, and provenance attestations.
- Frozen production matrix: OS, compiler, CMake, Python, PyTorch, CUDA, CPU ISA, GPU model, driver version, and container image digest.
- Canary and failure-injection evidence for CUDA failure, unsupported ISA, bad inputs, memory pressure, fallback paths, rollback criteria, and diagnostics correctness.
- Independent review approvals for API/ABI, security, numerical correctness, performance, and ML integration.

A release with skipped hardware targets, missing CI provenance, or manually supplied evidence must be treated as failed.
