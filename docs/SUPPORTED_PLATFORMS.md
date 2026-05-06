# Supported Platforms

This repository is a v1.0 research library. The CPU C ABI is intended to build on Linux, macOS, and Windows. The CUDA/PyTorch extension is validated on Linux with NVIDIA CUDA and PyTorch matching the local build environment.

## Tier 1

- Linux x86_64, GCC and Clang, CPU C++ library.
- Linux x86_64 with NVIDIA CUDA, PyTorch extension, no-EOS CUDA fast path.

## Tier 2

- Windows MSVC CPU library.
- macOS CPU library.
- Linux ARM64/NEON CPU library when a self-hosted runner is available.

## Release gates

A stable release requires the following artifacts: CPU and CUDA wheel logs, CUDA parity CSV, performance threshold report, sanitizer/fuzzer artifacts, soak-test JSON, ABI report, SBOM, checksums, and provenance attestation.
