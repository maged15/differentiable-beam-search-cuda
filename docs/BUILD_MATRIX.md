# Build matrix

Tier 1: Linux x86_64 GCC/Clang, scalar and AVX-512-capable hosts when available.

Tier 2: macOS Clang, Windows MSVC, Linux ARM64/NEON, CUDA self-hosted runner.

Sanitizer campaigns: ASAN/UBSAN, TSAN, MSAN with Clang where available, plus libFuzzer smoke and long-running AFL/libFuzzer campaigns outside default CI.

CUDA builds require `-DDBS_ENABLE_CUDA=ON`, a CUDA compiler, and a visible GPU for runtime tests.
