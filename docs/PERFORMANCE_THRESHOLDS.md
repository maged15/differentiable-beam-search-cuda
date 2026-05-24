# Performance Thresholds

`make cuda-validate` runs CUDA wheel build, smoke/parity tests, direct CUDA benchmarks, and `scripts/check_perf_thresholds.py`.

Default threshold policy:

- CPU/CUDA final-score parity must be `<= 1e-5`.
- CUDA/PyTorch reference parity must be `<= 1e-5`.
- Large cases, defined as `V >= 32000`, `T >= 16`, `K >= 4`, must not be more than `2.0x` slower than CPU DBS.
- DBS CUDA must not be more than `2.0x` slower than the PyTorch reference top-k baseline.
- Peak CUDA memory must stay below `4096 MB` unless explicitly overridden.

Override knobs:

```bash
DBS_MAX_CUDA_VS_CPU_SLOWDOWN=1.5 \
DBS_MAX_CUDA_VS_TORCH_SLOWDOWN=1.5 \
DBS_MAX_PARITY_DIFF=1e-5 \
DBS_MAX_PEAK_CUDA_MB=4096 \
make cuda-validate
```

`benchmarks/bench_dbs_cuda_direct.py` sets `DBS_BENCH_VALIDATE_INPUTS=0`
by default so timings measure the decode hot path instead of the optional
input-validation scan. Set `DBS_BENCH_VALIDATE_INPUTS=1` to benchmark the
default safe public API path.

These thresholds are benchmark alerts. They are not production certification
or a claim that DBS CUDA beats a framework baseline on every shape.
