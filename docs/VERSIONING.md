# Versioning

This repository has three related version numbers:

- Python package version: `1.0.0` in `VERSION`. This uses PEP 440 because it is consumed by Python packaging tools.
- CMake/shared-library version: `1.0.0` in `CMakeLists.txt`. CMake package-version files require numeric versions, so prerelease labels are not encoded there.
- C ABI version: `10` in `include/dbs.h` as `DBS_ABI_VERSION`. This is the runtime binary compatibility contract and the shared-library `SOVERSION`.

Pre-release builds may change the Python prerelease suffix without changing the C ABI version. ABI-breaking changes must increment `DBS_ABI_VERSION`, update `docs/ABI.md`, and regenerate `docs/abi/v*.symbols`.


`VERSION` is the source of truth for Python packaging. `pyproject.toml` declares the version as dynamic and setuptools reads `VERSION`. CMake uses `project(dbs VERSION 1.0.0)` for the library `VERSION`, while shared-library `SOVERSION` follows `DBS_ABI_VERSION`.
