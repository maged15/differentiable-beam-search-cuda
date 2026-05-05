# cifix6

- Fixed native CUDA extension public API to accept both `[T,K,V]` and `[B,T,K,V]` directly.
- Kept Python wrapper shape normalization consistent with the native CUDA extension.
- Rejected non-default CUDA options that would otherwise be silently ignored (`min_length`, length penalty, temperature fields, soft-top-k settings, relaxed-pool sizing, vocab block).
- Corrected the README C API example so backward runs before `dbs_free_result()` and `dbs_destroy()`.
- Confirmed MIT license text and bumped Python package metadata to `1.0.0rc7` while keeping CMake library version `1.0.0` and C ABI version `10` explicitly separate.
- Kept `dbs::dbs_cuda` exported as a stable target in CUDA and non-CUDA CMake builds.
