# CIFIX2

Fixes the remaining MSVC linkage mismatch for `dbs_batch_result_at` by making the internal forward declaration use the same exported C ABI declaration form as the exported definition.
