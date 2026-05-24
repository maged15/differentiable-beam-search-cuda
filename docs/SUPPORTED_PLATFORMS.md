# Supported Platforms

This repository is a research library. The CPU C ABI is intended to build on Linux, macOS, and Windows. The CUDA/PyTorch extension is validated on Linux with NVIDIA CUDA and PyTorch matching the local build environment.

## Tier 1

- Linux x86_64, GCC and Clang, CPU C++ library.
- Linux x86_64 with NVIDIA CUDA, PyTorch extension, no-EOS CUDA fast path.

## Tier 2

- Windows MSVC CPU library.
- macOS CPU library.
- Linux ARM64/NEON CPU library when a self-hosted runner is available.

## Validation Artifacts

Useful validation artifacts include CPU and CUDA wheel logs, CUDA parity CSVs, performance threshold reports, sanitizer/fuzzer artifacts, soak-test JSON, ABI reports, SBOMs, checksums, and provenance attestations. They are evidence for a specific environment, not a blanket production claim.
