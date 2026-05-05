# CI Fix 1

This patch fixes the first public GitHub Actions failures after the CUDA RC push.

Changes:
- Adds `docs/abi/v1.0.symbols` so `scripts/check_abi.sh` has a v1.0 exact baseline.
- Propagates sanitizer compile/link flags through the `dbs` target with `PUBLIC` usage requirements so test/benchmark executables link ASAN/UBSAN runtimes correctly.
- Uses MSVC release flags (`/O2 /DNDEBUG`) instead of GCC-style `-O3 -DNDEBUG` on Windows.
- Marks internal C ABI declarations with `DBS_EXPORT` so MSVC sees matching linkage between prototypes and exported definitions.
- Replaces several raw handle allocations with `std::unique_ptr` until success, preventing leaks in exception/fail-closed paths.
- Adds `requirements.txt`, `requirements-dev.txt`, and `.gitignore`.
- Replaces the GitHub Actions workflow with explicit GCC, Clang, macOS, Windows, sanitizer, ABI, Python wheel, and optional self-hosted CUDA jobs.

Local syntax checks run in this package:
- `g++ -std=c++17 -Iinclude -DDBS_BUILD_SHARED -fsyntax-only src/differentiable_beam.cpp`
- `clang++ -std=c++17 -Iinclude -DDBS_BUILD_SHARED -fsyntax-only src/differentiable_beam.cpp`
