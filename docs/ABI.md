# ABI policy

The public C ABI is `include/dbs.h`. Internal implementation details live under `src/`, `cuda/`, and `python/` and are not ABI-stable.

Rules:

- `DBS_ABI_VERSION` increments on ABI-breaking changes.
- Public handles remain opaque.
- Memory ownership stays explicit: every result/backward/batch/workspace handle returned by the library has a matching free/destroy function.
- Dense backward is never implicit; callers must opt in through `dbs_backward_dense()`.

## v1.0 ABI Checks

`docs/abi/v1.0.symbols` is the exact exported-symbol manifest for this release. `docs/abi/v0.5.symbols` is used as the backwards-compatibility baseline.

Run:

```bash
./scripts/check_abi.sh build-release/libdbs.so docs/abi/v1.0.symbols
./scripts/check_abi_compat.sh build-release/libdbs.so docs/abi/v0.5.symbols
```

The first check detects accidental symbol additions/removals for the current release. The second check prevents removal of previously exported public symbols.
