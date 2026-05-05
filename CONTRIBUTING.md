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

For release-candidate changes, attach benchmark CSV files and sanitizer/fuzz/soak artifacts when relevant. Do not claim production readiness unless the production gate evidence is available.
