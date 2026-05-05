# Differentiable Beam Search (DBS) v1.0.0rc9

This package is a release-candidate research library for hard beam search with sparse surrogate gradients. The public PyTorch API is CPU-autograd/CUDA-forward: CUDA tensors are forward-only; CPU tensors support surrogate backward, while CUDA tensors currently support forward/parity validation only. v1.0 adds hardware validation gates, allocator/reproducibility counters, an optional CUDA fast-kernel path, optional CUDA PyTorch extension source, ABI symbol checks, and clearer release criteria.

Production status: not production-certified until `scripts/run_hardware_validation.sh`, CUDA parity tests, SIMD parity tests, ABI checks, and sanitizer/fuzzer campaigns have passed on the intended deployment hardware.

# dbs: sparse-surrogate differentiable beam search

`dbs` is a C++17 beam-search decoder with deterministic hard beam output and sparse-first surrogate gradients. The forward pass still uses hard top-k beam selection, so the discrete operation is not exactly differentiable; backward computes an explicit surrogate gradient over selected beams and a relaxed candidate pool.

## What changed in 1.0.0rc9

- Fixed the AVX-512 exponential range-reduction constant used by sigmoid/softmax-style gradient helpers.
- Added internal scalar-vs-AVX-512 parity tests for exp, selected softmax weights, and relaxed top-k weights.
- Enabled GCC target-attribute builds for AVX2/FMA and SSE4.2 helper paths.
- Capped CUDA sparse scatter launches with a grid-stride loop to avoid int grid overflow on large sparse gradients.
- Aligned shared-library `SOVERSION` with C ABI `10`, and expanded the metadata gate to check ABI/SOVERSION/license consistency.
- CUDA unbatched `[T,K,V]` inputs are normalized in the public Python wrapper before calling the rank-4 native CUDA extension.
- CUDA kernels support debug launch synchronization with `DBS_CUDA_SYNC_CHECK=1` or `DBS_CUDA_DEBUG_SYNC=1`.
- Non-CUDA CMake builds now export a stable `dbs::dbs_cuda` stub target.
- License text is consistently MIT.
- Version metadata uses `VERSION` as the source of truth for Python packaging; CMake library `VERSION` remains `1.0.0` and shared-library `SOVERSION` follows C ABI `10`.
- Versioning docs explicitly separate Python package prerelease, CMake library version, and C ABI version.

- Public tests now treat `[B,T,K,V]` with `B=1` as valid and reject only true beam-size/rank/dimension errors.
- Version metadata is checked from one source: `VERSION`; `pyproject.toml` declares the version as dynamic.


- `dbs_backward()` is now sparse by default. Dense gradients are opt-in through `dbs_backward_dense()` and remain protected by `max_dense_gradient_elements`.
- Added a model-step callback API: `dbs_decode_model_steps()` and `dbs_decode_model_steps_with_workspace()`.
- Added reusable workspace APIs for callback decoding: `dbs_workspace_create()`, `dbs_workspace_reserve()`, and `dbs_workspace_allocated_bytes()`.
- Added variable batch decode: per-example steps, beam sizes, EOS tokens, min lengths, banned tokens, and forced tokens.
- Added advanced constraints: repetition penalty, no-repeat n-gram, and token-filter callback via `DBSAdvancedConstraintsC`.
- Added observability through `DBSStatsC`: selected kernel, sparse/dense backward mode, model-step usage, timing, allocation estimate, result sizes, sparse nnz, and error category.
- Added CPU feature discovery plus real AVX2/SSE4.2/NEON kernels for dot/softmax normalization paths. AVX-512 remains the most complete optimized path; top-k candidate scanning still uses AVX-512 or scalar correctness paths.
- Added PyTorch C++ extension packaging plus a `torch.autograd.Function` wrapper.
- Added JAX `custom_vjp` host-callback wrapper for CPU validation.
- Replaced the CUDA scaffold with real CUDA source kernels for batched hard forward decode and sparse backward scatter. The CUDA implementation is intentionally conservative: one device thread decodes one batch example, so it is a correctness backend, not yet a high-throughput fused GPU decoder.
- Expanded C++ tests and benchmark matrix.
- Expanded CI matrix for GCC, Clang, MSVC, Linux, macOS, Windows, sanitizer builds, and Python wheel build.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_SHARED=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix install
```

Sanitizer build:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DDBS_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

Optional CUDA backend build:

```bash
cmake -S . -B build-cuda -DDBS_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
```

## C API example

```c
#include "dbs.h"

DBSOptionsC opt = {0};
opt.beam_size = 4;
opt.eos_token = -1;
opt.selected_temperature = 1.0f;
opt.soft_topk_temperature = 0.25f;
opt.relaxed_pool_multiplier = 8;
opt.vocab_block = 4096;
opt.soft_topk_tolerance = 1.0e-4f;
opt.soft_topk_max_iters = 48;
opt.validate_inputs = 1;
opt.max_dense_gradient_elements = 100000000;

DBSDecoderHandle* dec = NULL;
if (dbs_create_ex(opt, &dec) != 0) {
    puts(dbs_last_global_error());
    return 1;
}

DBSResultHandle* result = NULL;
int rc = dbs_decode(dec, log_probs, steps, vocab_size, &result);
if (rc != 0) {
    puts(dbs_last_error(dec));
    dbs_destroy(dec);
    return 1;
}

const int32_t* tokens = dbs_result_tokens(result);            /* [T * K] */
const float* final_scores = dbs_result_final_scores(result);  /* [K] */

/* Optional backward must run before freeing result or destroying dec. */
float grad_final[4] = {1.0f, 0.0f, 0.0f, 0.0f};
DBSBackwardHandle* backward = NULL;
rc = dbs_backward(dec, result, NULL, NULL, grad_final, &backward);
if (rc == 0) {
    const int64_t nnz = dbs_backward_sparse_logprob_count(backward);
    const int64_t* indices = dbs_backward_sparse_logprob_indices(backward);
    const float* values = dbs_backward_sparse_logprob_values(backward);
    (void)nnz;
    (void)indices;
    (void)values;
    dbs_free_backward(backward);
}

dbs_free_result(result);
dbs_destroy(dec);
```

Sparse backward is the default. Dense backward is explicit and must also run before
`dbs_free_result(result)` and `dbs_destroy(dec)`.

```c
DBSBackwardHandle* dense_backward = NULL;
rc = dbs_backward_dense(dec, result, NULL, NULL, grad_final, &dense_backward);
if (rc == 0) dbs_free_backward(dense_backward);
```

## Model-step callback API

`dbs_decode_model_steps()` calls user code once per decoding step. The callback receives the previous selected tokens and scores, fills a `[K, V]` log-prob row, and can manage external KV cache or recurrent state through `user_data`.

```c
int step_fn(void* user_data, int batch_index, int step,
            const int32_t* prev_tokens, const float* prev_scores,
            int beam_size, int vocab_size, float* out_log_probs);

DBSResultHandle* result = NULL;
dbs_decode_model_steps(dec, step_fn, user_data, 0, steps, vocab_size, &result);
```

Use `dbs_decode_model_steps_with_workspace()` to reuse buffers across calls.

## Batch semantics

Uniform batch input is contiguous `[B, T, K, V]` and decoded with `dbs_decode_batch()`.

Variable batch input uses `[B, max_T, max_K, V]` and `dbs_decode_batch_variable()`. It supports per-example steps, beam sizes, EOS tokens, min lengths, banned-token masks, and forced-token schedules.

## Constraints

`dbs_decode_constrained_ex()` supports:

- banned tokens: `[V]`
- forced tokens: `[T]`, with `-1` meaning unforced
- minimum EOS length
- repetition penalty
- no-repeat n-gram
- custom token filter callback

Advanced constraints use the scalar scan path for correctness. Simple banned/forced/min-length constraints can still use the optimized AVX-512 scan when available.

## PyTorch extension

The package includes both a ctypes wrapper (`python/torch_dbs.py`) and a compiled extension wrapper (`python/torch_dbs_extension.py`). The public tensor API is standardized across CPU and CUDA:

- `[T, K, V] -> [K]` for one example
- `[B, T, K, V] -> [B, K]` for batched examples

CPU tensors support surrogate autograd for both shapes. CUDA tensors support forward only; CUDA autograd backward intentionally raises until sparse-gradient CPU/GPU parity is implemented and validated.

Build the extension from the package root after building `libdbs`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_SHARED=ON
cmake --build build --parallel
LD_LIBRARY_PATH=$PWD/build pip install .
```

```python
import torch
from torch_dbs_extension import DBSOptions, final_scores

T, K, V = 4, 3, 100
log_probs = torch.randn(T, K, V, dtype=torch.float32, requires_grad=True)
scores = final_scores(log_probs, DBSOptions(beam_size=K))
loss = -scores[0]
loss.backward()  # CPU surrogate autograd

batched = torch.randn(2, T, K, V, dtype=torch.float32, requires_grad=True)
batched_scores = final_scores(batched, DBSOptions(beam_size=K))  # [2, K]
```

CUDA forward example:

```python
x = torch.randn(2, T, K, V, device="cuda")
y = final_scores(x, DBSOptions(beam_size=K))  # forward only
```

End-to-end CPU toy training example:

```bash
LD_LIBRARY_PATH=$PWD/build PYTHONPATH=$PWD/python python examples/training_toy.py
```

## JAX wrapper

`python/jax_dbs.py` exposes a CPU host-callback `jax.custom_vjp`. It is intended for validation and research workflows, not XLA deployment. Production JAX integration should replace it with an XLA custom call.

## Benchmarks

C++ benchmark matrix:

```bash
./build/dbs_bench 10
```

Single case:

```bash
./build/dbs_bench 16 8 32000 4 20
```

PyTorch comparison helper:

```bash
python benchmarks/bench_compare.py --dbs-bench build/dbs_bench --repeats 20
```

The benchmark reports forward latency, sparse backward latency, dense backward latency when safe, batch forward latency, sparse nnz, and CPU dispatch metadata.

## Gradient semantics

Forward selection is hard and discontinuous. Backward is a surrogate: selected-beam weights use softmax over selected scores, and relaxed-pool weights use a sigmoid-bisection k-hot relaxation. These gradients are useful for surrogate-gradient training experiments, but they are not exact gradients of hard top-k beam selection.

Sparse gradients are flattened `[T * K * V]` indices plus values. Dense gradients are available only through `dbs_backward_dense()` and guarded by `max_dense_gradient_elements`.

## Current limitations

- CUDA forward is wired into the optional PyTorch extension, but CUDA autograd backward is not implemented. This is a CPU-autograd/CUDA-forward package until GPU sparse surrogate backward passes parity tests. ROCm is not included.
- AVX2/SSE4.2/NEON kernels cover dot and softmax normalization paths; top-k scanning is still AVX-512-or-scalar.
- PyTorch and JAX integrations are CPU-first. TensorFlow and ONNX Runtime are not included.
- Model-step decoding is functional but recomputes partial prefixes to expose prior beam state; a fused incremental decoder should replace it for high-throughput production.
- No hardware-counter profiling or NUMA-aware scheduler is included.

## Packaging

See [`docs/PACKAGING.md`](docs/PACKAGING.md) for CPU/CUDA wheel build modes and install-matrix notes.

## Versioning and ABI

Python package version: `1.0.0rc9`. CMake library `VERSION`: `1.0.0`. Shared-library `SOVERSION` and C ABI version: `10` (`DBS_ABI_VERSION`).

The Python package uses PEP 440 prerelease versions. CMake intentionally uses numeric semantic versions because CMake package-version files do not support PEP 440 suffixes. Treat `DBS_ABI_VERSION` as the binary compatibility contract and keep it aligned with shared-library `SOVERSION`.

## License

MIT. See [`LICENSE`](LICENSE). SPDX identifier: `MIT`.


## v1.0 validation gates

CPU validation:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DDBS_BUILD_TESTS=ON -DDBS_BUILD_BENCHMARKS=ON
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
scripts/check_abi.sh build-release/libdbs.so
```

CUDA parity validation on NVIDIA hardware:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DDBS_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
DBS_BUILD_TORCH_CUDA=1 pip install -e .
python -m pytest python/tests/test_cuda_parity.py -q
```

The CUDA forward path has exact and opt-in score-only fast paths. CUDA autograd backward remains disabled in the PyTorch wrapper until sparse surrogate-gradient parity is validated on GPU.

## v1.0 release gate

v1.0 adds fail-closed production gates rather than assuming production readiness. The strict gate is:

```bash
DBS_RELEASE_STRICT=1 DBS_REQUIRE_CUDA=1 DBS_REQUIRE_TORCH=1 DBS_REQUIRE_LONG_FUZZ=1 ./scripts/run_release_gate.sh
```

The release gate requires CPU tests, ABI exact and compatibility checks, wheel build/tests, CUDA parity, benchmark artifacts, and long fuzz/sanitizer runs. Missing hardware or missing evidence fails the gate. Passing evidence is generated into `validation/generated/production_gate_manifest.generated.json` and backed by ignored artifacts under `release/`; the checked-in manifest under `validation/` is a non-passing template.

## Release-candidate validation targets

For local CUDA validation after installing PyTorch/CUDA dependencies:

```bash
make cuda-validate
```

This builds the CUDA wheel, installs/tests it, runs CUDA parity, runs the direct CUDA benchmark matrix, and fails if `scripts/check_perf_thresholds.py` detects correctness or performance regressions.

For soak testing:

```bash
make soak DBS_SOAK_ITERATIONS=1000
```

For fuzz/sanitizer evidence:

```bash
make long-fuzz DBS_FUZZ_SECONDS=86400
```

For release metadata:

```bash
make release-artifacts
```

## Runnable examples

```bash
python examples/cpu_decode.py
python examples/cuda_decode.py
python examples/training_toy.py
python examples/benchmark_repro.py
```

## Important limitations

The CUDA fast path is optimized for `eos_token = -1` no-EOS final-score decoding. EOS and constraint-heavy paths are still correctness-first. Gradients are surrogate gradients through hard beam selection, not exact mathematical derivatives. See `docs/CUDA_VALIDATION.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/GRADIENTS.md`, and `docs/PERFORMANCE_THRESHOLDS.md`.


## CUDA error handling

CUDA launch error handling is asynchronous by default to preserve normal PyTorch stream semantics. For validation and CI, set `DBS_CUDA_SYNC_CHECK=1` or `DBS_CUDA_DEBUG_SYNC=1` to synchronize the current stream after custom CUDA launches and surface device-side failures through the returned status/exception.
