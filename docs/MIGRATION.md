# Migration and versioning policy

Version `0.4.0` uses C ABI version `4`. Consumers should check `dbs_abi_version()` at startup.

Compatible releases keep existing fields, symbols, ownership rules, and memory layout stable. ABI-breaking changes must increment `DBS_ABI_VERSION`, preserve source compatibility where possible, and document migration steps in `CHANGELOG.md`.
