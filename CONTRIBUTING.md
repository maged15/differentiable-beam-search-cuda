# Contributing

Before opening a pull request, run:

```bash
make test
make abi
python -m pytest python/tests -q
```

For CUDA changes, run:

```bash
make cuda-validate
DBS_FORCE_EXACT_CUDA_KERNEL=1 python3 benchmarks/bench_dbs_cuda_direct.py
```

For changes that support public performance or hardware claims, attach the raw benchmark CSV files, validation logs, and sanitizer/fuzz/soak artifacts that were actually run.
