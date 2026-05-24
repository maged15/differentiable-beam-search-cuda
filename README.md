# Differentiable Beam Search (CUDA)

[![CI](https://github.com/maged15/differentiable-beam-search-cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/maged15/differentiable-beam-search-cuda/actions/workflows/ci.yml)
[![CUDA smoke](https://github.com/maged15/differentiable-beam-search-cuda/actions/workflows/cuda-smoke.yml/badge.svg)](https://github.com/maged15/differentiable-beam-search-cuda/actions/workflows/cuda-smoke.yml)

Experimental C++17 / CUDA library for hard top-k beam search with surrogate gradients. Forward decoding uses standard discrete top-k selection. Backward computes a sparse surrogate gradient over selected beams plus a relaxed candidate pool, intended for use in surrogate-gradient training experiments.

This is research code, not a production library. Use it to experiment, not to ship.

## What it does

- Hard beam search forward pass (CPU and CUDA)
- Sparse surrogate gradients via selected-beam softmax + relaxed-pool sigmoid weights
- PyTorch extension with autograd support for CPU and CUDA tensors
- Optional JAX wrapper (host callback, CPU only — not for XLA deployment)

The forward selection is discrete and not actually differentiable. The "differentiable" part is the surrogate gradient defined over the selected trace and a relaxed pool, similar in spirit to Gumbel-softmax tricks but specialized for beam search. These gradients work for training experiments but are not exact derivatives of top-k.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_SHARED=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

CUDA build:

```bash
cmake -S . -B build-cuda -DDBS_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
```

Sanitizer build:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DDBS_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel
```

## PyTorch usage

```python
import torch
from torch_dbs_extension import DBSOptions, final_scores

T, K, V = 4, 3, 100
log_probs = torch.randn(T, K, V, requires_grad=True)
scores = final_scores(log_probs, DBSOptions(beam_size=K))  # [K]
(-scores[0]).backward()
```

Batched: `[B, T, K, V] -> [B, K]`. CUDA tensors work the same way:

```python
x = torch.randn(2, T, K, V, device="cuda", requires_grad=True)
y = final_scores(x, DBSOptions(beam_size=K, length_penalty_alpha=0.2))
y.sum().backward()
```

Native CUDA kernels handle the forward pass for `beam_size <= 32` with default options. Larger beams or options like `length_penalty_alpha`, custom temperatures, or relaxed-pool tuning fall back to the CPU semantic implementation and return CUDA tensors. Backward currently uses CPU-equivalent semantics on both.

## C API

```c
#include "dbs.h"

DBSOptionsC opt = {0};
opt.beam_size = 4;
opt.eos_token = -1;

DBSDecoderHandle* dec = NULL;
dbs_create_ex(opt, &dec);

DBSResultHandle* result = NULL;
dbs_decode(dec, log_probs, steps, vocab_size, &result);

const int32_t* tokens = dbs_result_tokens(result);          // [T * K]
const float* scores  = dbs_result_final_scores(result);     // [K]

dbs_free_result(result);
dbs_destroy(dec);
```

Backward (sparse, default):

```c
float grad_final[4] = {1.0f, 0, 0, 0};
DBSBackwardHandle* bwd = NULL;
dbs_backward(dec, result, NULL, NULL, grad_final, &bwd);
// dbs_backward_sparse_logprob_indices(bwd) / _values(bwd) / _count(bwd)
dbs_free_backward(bwd);
```

Dense backward is gated by `max_dense_gradient_elements` and explicit (`dbs_backward_dense`).

## Constraints

`dbs_decode_constrained_ex()` supports banned tokens, forced-token schedules, minimum EOS length, repetition penalty, no-repeat n-gram, and a custom filter callback. Simple cases (banned/forced/min-length) stay on the optimized scan path; advanced constraints use the scalar scan for correctness.

## Variable-length batching

`dbs_decode_batch_variable()` accepts `[B, max_T, max_K, V]` with per-example steps, beam sizes, EOS tokens, and constraints. Uniform batches use `dbs_decode_batch()` with contiguous `[B, T, K, V]`.

## Benchmarks

Measured locally on **AMD Ryzen 7 7800X3D + RTX 4080 SUPER**, Release build, 5 repeats:

| T  | K | V     | B | fwd (µs) | sparse bwd (µs) |
| -- | - | ----- | - | -------- | --------------- |
| 4  | 2 | 32000 | 1 |          |                 |
| 4  | 4 | 32000 | 4 |          |                 |
| 8  | 8 | 32000 | 1 |          |                 |
| 16 | 8 | 32000 | 1 |          |                 |

(Fill in once you re-run `./build/dbs_bench 10` and capture the numbers.)

```bash
./build/dbs_bench 10                       # full matrix
./build/dbs_bench 16 8 32000 4 20          # single case
python benchmarks/bench_compare.py --dbs-bench build/dbs_bench --repeats 20
```

Sparse backward is essentially free relative to forward — it only scatters `T*K` gradient entries. Dense backward returns `-1` when `T*K*V > max_dense_gradient_elements`.

## Examples

```bash
python examples/cpu_decode.py
python examples/cuda_decode.py
python examples/training_toy.py
python examples/benchmark_repro.py
```

## Limitations

- "Differentiable" here means surrogate-differentiable. The forward is hard top-k.
- ROCm is not supported.
- CUDA backward uses CPU-equivalent semantics; native CUDA sparse backward is a limited utility, not the primary path.
- Model-step decoding recomputes partial prefixes; a fused incremental decoder would be needed for high-throughput serving.
- No NUMA-aware scheduling, no hardware counter profiling, no production sanitizer/fuzz campaigns. If you need those, this isn't the library yet.
- TensorFlow and ONNX Runtime are not included.

## Versioning and ABI

Package version: **1.0.0**. C ABI version: **10** (`DBS_ABI_VERSION`), aligned with the shared-library `SOVERSION`. Treat `DBS_ABI_VERSION` as the binary compatibility contract.

## License

MIT. See [`LICENSE`](LICENSE).
