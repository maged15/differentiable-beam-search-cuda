# Refactoring plan

`src/differentiable_beam.cpp` is intentionally still a monolith in this release candidate so ABI behavior is easy to audit. Before adding more features, split it into:

1. `decoder_core.*` — hard beam expansion, deterministic ordering, EOS behavior
2. `gradients.*` — sparse/dense surrogate backward and relaxation utilities
3. `constraints.*` — banned/forced tokens, min length, repetition penalty, no-repeat n-gram, callbacks
4. `simd_dispatch.*` — scalar/AVX/SSE/NEON feature detection and kernels
5. `c_api.*` — opaque handles, ownership, exported ABI wrappers
6. `stats_validation.*` — counters and JSON summaries

Recommended sequence: first move pure helpers with no ABI surface, then decoder internals, then gradients, leaving exported `dbs_*` symbols in `c_api.cpp` until the final stage.
