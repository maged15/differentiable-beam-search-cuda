# Release policy

The public ABI is the C header under `include/dbs.h`. Each release must preserve existing exported `dbs_*` symbols or intentionally bump `DBS_ABI_VERSION` and document the break.

Before tagging a release:

1. Run CPU tests, ABI symbol check, sanitizer tests, and benchmark smoke tests.
2. Run CUDA parity tests on NVIDIA hardware when CUDA artifacts are included.
3. Run SIMD parity/benchmark tests on every supported CPU family.
4. Publish benchmark CSVs with compiler, CPU/GPU model, OS, and library versions.
5. Update `VERSION`, `CHANGELOG.md`, `IMPLEMENTATION_STATUS.md`, and migration notes.

This project does not claim production certification. Release notes should link the raw validation logs and benchmark CSVs that were actually run.
