# Benchmark methodology

Run C++ microbenchmarks with:

```bash
./build/dbs_bench 16 8 32000 4 20
```

Run PyTorch/Hugging Face comparisons with:

```bash
python benchmarks/bench_compare.py --dbs-bench build/dbs_bench --repeats 20 --hf-model gpt2
```

Report T, K, V, batch size, forward latency, sparse backward latency, dense backward latency when enabled, sparse nnz, selected kernel, CPU feature flags, and hardware/compiler metadata.

## Realistic-scale benchmark matrix

Use the Python benchmark for PyTorch/Hugging Face comparisons:

```bash
python benchmarks/bench_compare.py --device cpu --repeats 5 --dbs-bench build/dbs_bench
python benchmarks/bench_compare.py --device cuda --repeats 5 --dbs-bench build/dbs_bench
```

The matrix includes vocabulary sizes 32k, 64k, and 128k. Do not summarize benchmark claims without the hardware, compiler, and dependency versions.

For the direct CUDA benchmark, `DBS_BENCH_VALIDATE_INPUTS=0` is the default so
the timing measures the decode hot path. Set `DBS_BENCH_VALIDATE_INPUTS=1` when
you want to include the public API finite-value validation scan in the timing.

## Benchmark evidence

Benchmark artifacts should live under `benchmarks/results/`. Useful evidence includes:

- `cpu_smoke.csv` from the C++ benchmark.
- `pytorch_baselines.csv` from PyTorch greedy/top-k beam baselines.
- CUDA timing logs when a CUDA runner is present.
- Large-vocabulary rows for 32k, 64k, and 128k vocabulary sizes.

Benchmark output is useful only when collected with compiler, CPU model, GPU model, driver, and thread settings recorded.
