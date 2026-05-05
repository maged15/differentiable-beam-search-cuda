# Versioning

This repository has three related version numbers:

- Python package version: `1.0.0rc8` in `VERSION` and `pyproject.toml`. This uses PEP 440 because it is consumed by Python packaging tools.
- CMake/shared-library version: `1.0.0` in `CMakeLists.txt`. CMake package-version files require numeric versions, so prerelease labels are not encoded there.
- C ABI version: `10` in `include/dbs.h` as `DBS_ABI_VERSION`. This is the runtime binary compatibility contract.

Compatible release-candidate updates may change the Python prerelease suffix without changing the C ABI version. ABI-breaking changes must increment `DBS_ABI_VERSION`, update `docs/ABI.md`, and regenerate `docs/abi/v*.symbols`.


`VERSION` is the source of truth for Python packaging. `pyproject.toml` is kept in sync for static metadata readers. CMake uses `project(dbs VERSION 1.0.0)` for the shared-library/SOVERSION line, while `DBS_ABI_VERSION` remains `10` for the C ABI.
